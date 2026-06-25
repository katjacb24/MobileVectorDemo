import Foundation
import Combine
import LLM

// Opaque conversation token — LLM.swift manages history in the bot instance itself.
final class LlamaConversation: @unchecked Sendable {}

@MainActor
final class LlamaEngine: ObservableObject {

    enum LoadState: Equatable {
        case idle, loading, ready
        case failed(String)
    }

    @Published private(set) var loadState:    LoadState = .idle
    @Published private(set) var isGenerating: Bool      = false

    static let shared = LlamaEngine()

    // Holds the most recent bot instance. History accumulates here between sends
    // so stream() can copy it into the next fresh bot.
    private var bot:    LLM?
    private var botURL: URL?

    // System prompt for the active conversation, stored so stream() can rebuild
    // the template on each fresh bot without needing a parameter.
    private var conversationSystemPrompt: String?

    private init() {}

    var isReady: Bool { loadState == .ready }

    static let modelResource = "llama-3.2-3b-instruct-q4_k_m"

    private static func llama3Template(systemPrompt: String? = nil) -> Template {
        Template(
            prefix: "<|begin_of_text|>",
            system: ("<|start_header_id|>system<|end_header_id|>\n\n", "<|eot_id|>"),
            user: ("<|start_header_id|>user<|end_header_id|>\n\n", "<|eot_id|>"),
            bot: ("<|start_header_id|>assistant<|end_header_id|>\n\n", "<|eot_id|>"),
            stopSequence: "<|eot_id|>",
            systemPrompt: systemPrompt
        )
    }

    // MARK: - Lifecycle

    func load() async {
        guard loadState == .idle else { return }
        loadState = .loading

        guard let url = Bundle.main.url(forResource: Self.modelResource, withExtension: "gguf") else {
            loadState = .failed("Llama model not found in app bundle.\nRun Scripts/download_llama.sh first.")
            return
        }

        botURL = url

        let loaded = await Task.detached(priority: .userInitiated) {
            LLM(from: url, historyLimit: 4, maxTokenCount: 4096)
        }.value

        bot = loaded
        loadState = bot != nil ? .ready : .failed("Llama model failed to initialise.")
    }

    // MARK: - Conversation (AssistantView)

    // Stores the system prompt and clears history so the next send() starts a fresh
    // conversation. The actual LLM instance is (re)created per send() call.
    func newConversation(systemPrompt: String) async throws -> LlamaConversation {
        guard loadState == .ready else { throw LlamaError.notLoaded }
        conversationSystemPrompt = systemPrompt
        bot?.history.removeAll()
        return LlamaConversation()
    }

    nonisolated func send(
        _ prompt: String,
        in _: LlamaConversation
    ) -> AsyncThrowingStream<String, Error> {
        stream(prompt: prompt)
    }

    // MARK: - One-shot (RecommendationSheet)

    func oneShot(systemPrompt: String, userPrompt: String) -> AsyncThrowingStream<String, Error> {
        streamFresh(systemPrompt: systemPrompt, prompt: userPrompt)
    }

    // MARK: - Errors

    enum LlamaError: LocalizedError {
        case notLoaded
        var errorDescription: String? { "Llama model is not loaded yet." }
    }

    // MARK: - Private

    // Used by send() for multi-turn conversation.
    //
    // Creates a fresh LLM instance on every call to guarantee a clean llama.cpp KV cache —
    // reusing the same instance across calls corrupts the KV cache and causes the model to
    // output "..." from the second call onwards.
    //
    // The accumulated history is copied from the previous bot so the model still sees the
    // full conversation; it just re-evaluates it from scratch rather than from a stale cache.
    nonisolated private func stream(prompt: String) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task { @MainActor [weak self] in
                guard let self, let url = botURL else {
                    continuation.finish(throwing: LlamaError.notLoaded)
                    return
                }

                while isGenerating { await Task.yield() }
                isGenerating = true

                // Snapshot history and system prompt before spinning up the fresh bot.
                let oldHistory     = bot?.history ?? []
                let systemPrompt   = conversationSystemPrompt ?? ""

                let freshBot = await Task.detached(priority: .userInitiated) {
                    LLM(from: url, historyLimit: 10, maxTokenCount: 4096)
                }.value

                guard let freshBot else {
                    isGenerating = false
                    continuation.finish(throwing: LlamaError.notLoaded)
                    return
                }

                freshBot.template = Self.llama3Template(systemPrompt: systemPrompt)
                freshBot.history  = oldHistory   // re-inject prior exchanges
                bot = freshBot                   // replace so history stays updated after respond()

                var previous = ""
                var cancellable: AnyCancellable?

                cancellable = freshBot.$output.sink { current in
                    if current.count < previous.count { previous = current }
                    let new = String(current.dropFirst(previous.count))
                    if !new.isEmpty {
                        continuation.yield(new)
                        previous = current
                    }
                }

                await freshBot.respond(to: prompt)
                cancellable?.cancel()

                let tail = String(freshBot.output.dropFirst(previous.count))
                if !tail.isEmpty { continuation.yield(tail) }

                isGenerating = false
                continuation.finish()
            }
        }
    }

    // Used by oneShot() — disposable fresh bot with no history, freed after each call.
    private func streamFresh(systemPrompt: String, prompt: String) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task { @MainActor [weak self] in
                guard let self, let url = botURL else {
                    continuation.finish(throwing: LlamaError.notLoaded)
                    return
                }

                while isGenerating { await Task.yield() }
                isGenerating = true

                let freshBot = await Task.detached(priority: .userInitiated) {
                    LLM(from: url, historyLimit: 0, maxTokenCount: 4096)
                }.value

                guard let freshBot else {
                    isGenerating = false
                    continuation.finish(throwing: LlamaError.notLoaded)
                    return
                }

                freshBot.template = Self.llama3Template(systemPrompt: systemPrompt)

                var previous = ""
                var cancellable: AnyCancellable?

                cancellable = freshBot.$output.sink { current in
                    if current.count < previous.count { previous = current }
                    let new = String(current.dropFirst(previous.count))
                    if !new.isEmpty {
                        continuation.yield(new)
                        previous = current
                    }
                }

                await freshBot.respond(to: prompt)
                cancellable?.cancel()

                let tail = String(freshBot.output.dropFirst(previous.count))
                if !tail.isEmpty { continuation.yield(tail) }

                isGenerating = false
                continuation.finish()
            }
        }
    }
}
