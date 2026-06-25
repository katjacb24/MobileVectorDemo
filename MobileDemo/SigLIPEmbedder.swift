import CoreML
import UIKit

// ── Model resources ───────────────────────────────────────────────────────────
// The two .mlpackage files are members of the target's Sources build phase, so
// Xcode compiles them to .mlmodelc inside the app bundle at build time. vocab.json
// is a bundled resource. Everything loads from Bundle.main, so this works
// identically on the simulator and on a physical device — no filesystem paths.
private enum ModelResource {
    static func url(_ name: String, _ ext: String) throws -> URL {
        guard let url = Bundle.main.url(forResource: name, withExtension: ext) else {
            throw EmbedError.resourceMissing("\(name).\(ext)")
        }
        return url
    }
}

// ── SigLIP constants ──────────────────────────────────────────────────────────
private let kImageSize  = 224   // pixels; ViT-B/16 input resolution
private let kTextMaxLen = 64    // maximum token sequence length
private let kEmbedDim   = 768   // output embedding dimensionality

// ── Public result types ───────────────────────────────────────────────────────

struct EmbeddingResult {
    let embedding: [Float]   // 768-dim, already L2-normalised by the Core ML model
    let latencyMs: Double
}

// ── SigLIPEmbedder ────────────────────────────────────────────────────────────

@MainActor
final class SigLIPEmbedder: ObservableObject {

    @Published private(set) var isLoaded  = false
    @Published private(set) var loadError: String?

    private var visionModel: MLModel?
    private var textModel:   MLModel?
    private var tokenizer:   SigLIPTokenizer?

    // MARK: - Loading

    func load() async {
        do {
            // Models are already compiled to .mlmodelc in the app bundle at build time,
            // so we load them directly — no runtime compilation. Both load concurrently
            // off the main actor.
            let visionURL = try ModelResource.url("SigLIPVision", "mlmodelc")
            let textURL   = try ModelResource.url("SigLIPText",   "mlmodelc")

            async let vm = Task.detached(priority: .userInitiated) {
                try Self.loadModel(at: visionURL)
            }.value
            async let tm = Task.detached(priority: .userInitiated) {
                try Self.loadModel(at: textURL)
            }.value

            visionModel = try await vm
            textModel   = try await tm
            tokenizer   = try SigLIPTokenizer.load(from: ModelResource.url("vocab", "json"))

            isLoaded = true
        } catch {
            loadError = error.localizedDescription
        }
    }

    // Loads a bundled, already-compiled .mlmodelc.
    private nonisolated static func loadModel(at url: URL) throws -> MLModel {
        let config = MLModelConfiguration()
        // .all enables Neural Engine on device; simulator falls back to CPU/GPU
        config.computeUnits = .all
        return try MLModel(contentsOf: url, configuration: config)
    }

    // MARK: - Image embedding

    func embed(image: UIImage) throws -> EmbeddingResult {
        guard let model = visionModel else { throw EmbedError.notLoaded }

        let features = try preprocessImage(image)

        let t0  = Date()
        let out = try model.prediction(from: features)
        let ms  = Date().timeIntervalSince(t0) * 1_000

        return EmbeddingResult(embedding: extractFloats(from: out, name: "embedding"),
                               latencyMs: ms)
    }

    // MARK: - Text embedding

    func embed(text: String) throws -> EmbeddingResult {
        guard let model = textModel   else { throw EmbedError.notLoaded }
        guard let tok   = tokenizer   else { throw EmbedError.notLoaded }

        let features = try preprocessText(text, tokenizer: tok)

        let t0  = Date()
        let out = try model.prediction(from: features)
        let ms  = Date().timeIntervalSince(t0) * 1_000

        return EmbeddingResult(embedding: extractFloats(from: out, name: "embedding"),
                               latencyMs: ms)
    }

    // MARK: - Private: image preprocessing

    private func preprocessImage(_ image: UIImage) throws -> MLFeatureProvider {
        // Resize to 224×224
        let size = CGSize(width: kImageSize, height: kImageSize)
        UIGraphicsBeginImageContextWithOptions(size, true, 1.0)
        image.draw(in: CGRect(origin: .zero, size: size))
        let resized = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        guard let cgImage = resized?.cgImage else { throw EmbedError.imagePreprocessFailed }

        // Decode to RGBA bytes
        let px = kImageSize
        var rgba = [UInt8](repeating: 0, count: px * px * 4)
        let ctx  = CGContext(data: &rgba, width: px, height: px,
                             bitsPerComponent: 8, bytesPerRow: px * 4,
                             space: CGColorSpaceCreateDeviceRGB(),
                             bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        ctx?.draw(cgImage, in: CGRect(x: 0, y: 0, width: px, height: px))

        // Build CHW float32 tensor normalised to [-1, 1]
        // SigLIP normalisation: (x/255 − 0.5) / 0.5  =  x/127.5 − 1
        let array = try MLMultiArray(shape: [1, 3, NSNumber(value: px), NSNumber(value: px)],
                                     dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: 1 * 3 * px * px)
        for y in 0..<px {
            for x in 0..<px {
                let base = (y * px + x) * 4
                ptr[0 * px * px + y * px + x] = Float(rgba[base    ]) / 127.5 - 1.0  // R
                ptr[1 * px * px + y * px + x] = Float(rgba[base + 1]) / 127.5 - 1.0  // G
                ptr[2 * px * px + y * px + x] = Float(rgba[base + 2]) / 127.5 - 1.0  // B
            }
        }
        return try MLDictionaryFeatureProvider(dictionary: ["pixel_values": array])
    }

    // MARK: - Private: text preprocessing

    private func preprocessText(_ text: String,
                                 tokenizer: SigLIPTokenizer) throws -> MLFeatureProvider {
        let (ids, mask) = tokenizer.encode(text)

        let idsArray  = try MLMultiArray(shape: [1, NSNumber(value: kTextMaxLen)], dataType: .int32)
        let maskArray = try MLMultiArray(shape: [1, NSNumber(value: kTextMaxLen)], dataType: .int32)
        let idsPtr    = idsArray.dataPointer.bindMemory(to: Int32.self,  capacity: kTextMaxLen)
        let maskPtr   = maskArray.dataPointer.bindMemory(to: Int32.self, capacity: kTextMaxLen)
        for i in 0..<kTextMaxLen {
            idsPtr[i]  = Int32(ids[i])
            maskPtr[i] = Int32(mask[i])
        }
        return try MLDictionaryFeatureProvider(dictionary: [
            "input_ids":      idsArray,
            "attention_mask": maskArray,
        ])
    }

    // MARK: - Private: output extraction

    private func extractFloats(from output: MLFeatureProvider, name: String) -> [Float] {
        guard let arr = output.featureValue(for: name)?.multiArrayValue else { return [] }
        let ptr = arr.dataPointer.bindMemory(to: Float32.self, capacity: arr.count)
        return Array(UnsafeBufferPointer(start: ptr, count: arr.count))
    }
}

// ── Errors ────────────────────────────────────────────────────────────────────

enum EmbedError: LocalizedError {
    case notLoaded, imagePreprocessFailed, inferenceFailed
    case resourceMissing(String)

    var errorDescription: String? {
        switch self {
        case .notLoaded:             return "Models not yet loaded"
        case .imagePreprocessFailed: return "Failed to preprocess image"
        case .inferenceFailed:       return "Core ML inference failed"
        case .resourceMissing(let n): return "Missing bundled resource: \(n). Did the model files get added to the target?"
        }
    }
}

// ── SigLIPTokenizer ───────────────────────────────────────────────────────────
// Minimal pure-Swift implementation of the SentencePiece Unigram tokenizer
// used by google/siglip-base-patch16-224. Reads vocab.json generated by
// Scripts/convert_siglip.py (which extracts pieces + scores from spiece.model).
//
// SigLIP tokenisation rules (from tokenizer_config.json):
//   • do_lower_case = true
//   • pad and EOS token ID = 1  (</s>)
//   • unk token ID = 2
//   • max length = 64; right-pad with EOS (1)
//   • attention_mask: all ones (SigLIP uses EOS as padding; model was traced with all-ones mask)

struct SigLIPTokenizer {

    private let pieces:    [String]       // pieces[id] = piece string
    private let scores:    [Float]        // scores[id] = log-prob from SentencePiece model
    private let pieceToId: [String: Int]  // reverse lookup
    let unkId: Int
    let eosId: Int
    let padId: Int

    // MARK: Loading

    static func load(from url: URL) throws -> SigLIPTokenizer {
        let data = try Data(contentsOf: url)
        let file = try JSONDecoder().decode(VocabFile.self, from: data)
        return SigLIPTokenizer(pieces: file.pieces,
                               scores: file.scores,
                               unkId:  file.unkId,
                               eosId:  file.eosId,
                               padId:  file.padId)
    }

    private init(pieces: [String], scores: [Float], unkId: Int, eosId: Int, padId: Int) {
        self.pieces    = pieces
        self.scores    = scores
        self.unkId     = unkId
        self.eosId     = eosId
        self.padId     = padId
        self.pieceToId = Dictionary(uniqueKeysWithValues: pieces.enumerated().map { ($1, $0) })
    }

    // MARK: Encode

    // Returns (input_ids, attention_mask) each of length kTextMaxLen.
    // attention_mask is all ones — see note above.
    func encode(_ text: String) -> ([Int], [Int]) {
        let normalised  = normalize(text)           // lowercase + ▁ prefix
        let tokenPieces = viterbi(normalised)       // segmentation
        var ids         = tokenPieces.map { pieceToId[$0] ?? unkId }
        ids.append(eosId)                           // append </s>

        // Pad to kTextMaxLen with padId
        var paddedIds  = [Int](repeating: padId, count: kTextMaxLen)
        var mask       = [Int](repeating: 1,     count: kTextMaxLen)   // all ones
        let count      = min(ids.count, kTextMaxLen)
        for i in 0..<count { paddedIds[i] = ids[i] }
        return (paddedIds, mask)
    }

    // MARK: Private: normalisation

    // SentencePiece adds ▁ (U+2581) as a word boundary marker before each word.
    // SigLIP also lowercases the input.
    private func normalize(_ text: String) -> String {
        "▁" + text.lowercased().replacingOccurrences(of: " ", with: "▁")
    }

    // MARK: Private: Viterbi segmentation

    // Standard SentencePiece Unigram Viterbi: finds the segmentation of the
    // input character sequence with the highest sum of log-prob scores.
    private func viterbi(_ text: String) -> [String] {
        let scalars = Array(text.unicodeScalars)
        let n       = scalars.count

        // best[i] = highest score for text[0..<i]; -∞ means unreachable
        var best = [Float](repeating: -.infinity, count: n + 1)
        var prev = [Int](repeating: -1,           count: n + 1)
        best[0]  = 0.0

        for end in 1...n {
            // Try every possible last piece text[start..<end]
            for start in 0..<end {
                let piece = String(String.UnicodeScalarView(scalars[start..<end]))
                guard let id = pieceToId[piece] else { continue }
                let candidate = best[start] + scores[id]
                if candidate > best[end] {
                    best[end] = candidate
                    prev[end] = start
                }
            }
            // Single-character fallback for any position still unreachable
            if best[end] == -.infinity {
                let ch  = String(scalars[end - 1])
                let id  = pieceToId[ch] ?? unkId
                best[end] = best[end - 1] + (id < scores.count ? scores[id] : -100.0)
                prev[end] = end - 1
            }
        }

        // Backtrack to recover the token sequence
        var result: [String] = []
        var pos = n
        while pos > 0 {
            let start = prev[pos]
            result.append(String(String.UnicodeScalarView(scalars[start..<pos])))
            pos = start
        }
        return result.reversed()
    }

    // MARK: Private: Decodable vocab file

    private struct VocabFile: Decodable {
        let pieces: [String]
        let scores: [Float]
        let unkId:  Int
        let eosId:  Int
        let padId:  Int
        enum CodingKeys: String, CodingKey {
            case pieces, scores
            case unkId = "unk_id"
            case eosId = "eos_id"
            case padId = "pad_id"
        }
    }
}
