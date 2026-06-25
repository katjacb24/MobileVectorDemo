import SwiftUI

// ── Colour palette ─────────────────────────────────────────────────────────
private enum LP {
    // Backgrounds
    static let ivory   = Color(red: 0.965, green: 0.945, blue: 0.912)   // main bg
    static let card    = Color.white
    // Top bar
    static let topBar  = Color(red: 0.10,  green: 0.09,  blue: 0.07)    // warm black
    // Accent
    static let gold    = Color(red: 0.78,  green: 0.63,  blue: 0.28)
    static let goldLight = Color(red: 0.90, green: 0.78, blue: 0.52)
    // Text
    static let inkDark = Color(red: 0.12,  green: 0.10,  blue: 0.08)    // near-black
    static let inkMid  = Color(red: 0.42,  green: 0.37,  blue: 0.30)    // warm gray
    // Pattern lines
    static let gridLine = Color(red: 0.83, green: 0.78, blue: 0.71)
}

// ── Diamond-grid background ───────────────────────────────────────────────────────
// Recreates the argyle/diamond grid from the template: two crossing sets of 45°
// diagonal lines drawn over the ivory base colour.
private struct DiamondBackground: View {
    private let spacing: CGFloat = 54
    var body: some View {
        Canvas { ctx, size in
            var path = Path()
            let count = Int((size.width + size.height) / spacing) + 4
            for i in -2..<count {
                let d = CGFloat(i) * spacing
                // \ lines
                path.move(to: CGPoint(x: d,              y: 0))
                path.addLine(to: CGPoint(x: d + size.height, y: size.height))
                // / lines
                path.move(to: CGPoint(x: d + size.height, y: 0))
                path.addLine(to: CGPoint(x: d,              y: size.height))
            }
            ctx.stroke(path, with: .color(LP.gridLine), lineWidth: 0.7)
        }
        .background(LP.ivory)
        .ignoresSafeArea()
    }
}

// ── Catalog screen ────────────────────────────────────────────────────────────────
@MainActor
struct StoreAssistantView: View {

    @StateObject private var embedder = SigLIPEmbedder()

    @State private var store:        InventoryStore?
    @State private var setupError:   String?
    @State private var isSeeding     = false

    @State private var products:     [Product] = []
    @State private var productsByID: [String: Product] = [:]
    @State private var thumbnails:   [String: UIImage] = [:]

    @State private var queryText     = ""
    @State private var queryImage:   UIImage?
    @State private var pickerSource: UIImagePickerController.SourceType?
    @State private var results:      [SearchResult] = []
    @State private var isSearching   = false
    @State private var searchError:  String?
    @State private var hasSearched   = false

    @State private var currentPage   = 0
    private let pageSize             = 50

    enum Tab { case catalog, schedule, assistant }
    @State private var selectedTab:      Tab = .catalog
    @State private var appointmentStore: AppointmentStore?
    @State private var scheduleRefreshID = UUID()

    private var isReady: Bool { store != nil && embedder.isLoaded && !isSeeding }
    private var isSearchActive: Bool { hasSearched }
    private var allDisplayedProducts: [Product] {
        if isSearchActive { return results.compactMap { productsByID[$0.id] } }
        return products.sorted { a, b in
            let aFull = !a.name.isEmpty && !a.description.isEmpty
            let bFull = !b.name.isEmpty && !b.description.isEmpty
            return aFull && !bFull
        }
    }
    private var displayedProducts: [Product] {
        let start = currentPage * pageSize
        let end   = min(start + pageSize, allDisplayedProducts.count)
        guard start < end else { return [] }
        return Array(allDisplayedProducts[start..<end])
    }
    private var totalPages: Int {
        max(1, Int(ceil(Double(allDisplayedProducts.count) / Double(pageSize))))
    }
    private var hasActiveQuery: Bool {
        !queryText.trimmingCharacters(in: .whitespaces).isEmpty || queryImage != nil
    }

    // MARK: - Body

    var body: some View {
        ZStack(alignment: .top) {
            DiamondBackground()

            VStack(spacing: 0) {
                topBar

                switch selectedTab {
                case .catalog:
                    if let err = setupError ?? embedder.loadError {
                        statusView(err, isError: true)
                    } else if !isReady {
                        statusView(isSeeding ? "Seeding inventory…" : "Loading models…",
                                   isError: false)
                    } else {
                        searchBar
                        totalLabel
                        catalog
                    }
                case .schedule:
                    ScheduleView(store: appointmentStore, inventoryStore: store, embedder: embedder)
                        .id(scheduleRefreshID)
                case .assistant:
                    AssistantView(
                        store: store,
                        appointmentStore: appointmentStore,
                        embedder: embedder,
                        productsByID: productsByID
                    )
                }
            }
        }
        .task { await setup() }
        .sheet(item: $pickerSource) { source in
            ImagePicker(sourceType: source) { image in runImageSearch(image) }
                .ignoresSafeArea()
        }
    }

    // MARK: - Top bar

    private var topBar: some View {
        HStack(spacing: 0) {
            // Tab buttons
            HStack(spacing: 24) {
                tabButton("Catalog",   tab: .catalog)
                tabButton("Schedule",  tab: .schedule)
                tabButton("Assistant", tab: .assistant)
            }

            Spacer()

            Menu {
                Text("Store Assistant").font(.caption)
                Divider()
                Button(role: .destructive, action: reseed) {
                    Label("Re-seed inventory", systemImage: "arrow.clockwise")
                }
                .disabled(!isReady)
                Button(role: .destructive, action: reseedAppointments) {
                    Label("Re-seed appointments", systemImage: "arrow.clockwise")
                }
                .disabled(appointmentStore == nil)
            } label: {
                Image(systemName: "person.crop.circle")
                    .font(.title2)
                    .foregroundStyle(LP.gold)
            }
        }
        .padding(.horizontal, 20)
        .padding(.top, 52)
        .padding(.bottom, 14)
        .background(LP.topBar)
    }

    private func tabButton(_ label: String, tab: Tab) -> some View {
        Button { selectedTab = tab } label: {
            VStack(spacing: 4) {
                Text(label)
                    .font(.system(size: 16, weight: .semibold, design: .serif))
                    .foregroundStyle(selectedTab == tab ? LP.goldLight : LP.inkMid)
                    .tracking(1.5)
                Rectangle()
                    .frame(height: 2)
                    .foregroundStyle(selectedTab == tab ? LP.gold : Color.clear)
            }
        }
        .buttonStyle(.plain)
    }

    // MARK: - Search bar

    private var searchBar: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(LP.gold)
                    .font(.system(size: 15, weight: .medium))

                TextField("", text: $queryText,
                          prompt: Text("Search (e.g. blue shirt)…")
                              .foregroundStyle(LP.inkMid.opacity(0.7)))
                    .foregroundStyle(LP.inkDark)
                    .tint(LP.gold)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .submitLabel(.search)
                    .onSubmit(runSearch)

                if isSearching {
                    ProgressView().tint(LP.gold).scaleEffect(0.8)
                }

                Menu {
                    if UIImagePickerController.isSourceTypeAvailable(.camera) {
                        Button { pickerSource = .camera } label: {
                            Label("Take a Photo", systemImage: "camera")
                        }
                    }
                    Button { pickerSource = .photoLibrary } label: {
                        Label("Choose Photo", systemImage: "photo.on.rectangle")
                    }
                } label: {
                    Image(systemName: "camera")
                        .foregroundStyle(LP.gold)
                        .font(.system(size: 15, weight: .medium))
                }
                .disabled(isSearching)

                if hasActiveQuery {
                    Button(action: clearSearch) {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(LP.inkMid)
                    }
                    .disabled(isSearching)
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 11)
            .background(LP.card, in: Capsule())
            .overlay(Capsule().stroke(LP.gold.opacity(0.35), lineWidth: 1))

            if let queryImage {
                HStack(spacing: 8) {
                    Image(uiImage: queryImage)
                        .resizable()
                        .scaledToFill()
                        .frame(width: 28, height: 28)
                        .clipShape(RoundedRectangle(cornerRadius: 5))
                    Text("Searching by photo")
                        .font(.caption)
                        .foregroundStyle(LP.inkMid)
                    Spacer()
                }
                .padding(.horizontal, 4)
            }

            if let err = searchError {
                Text(err).font(.caption).foregroundStyle(.red).padding(.horizontal, 4)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
    }

    // MARK: - Total label

    private var totalLabel: some View {
        let count = allDisplayedProducts.count
        let label = isSearchActive
            ? "Total: \(count) result\(count == 1 ? "" : "s")"
            : "Total: \(count) product\(count == 1 ? "" : "s")"
        return HStack {
            Text(label)
                .font(.system(size: 12, weight: .medium, design: .serif))
                .foregroundStyle(LP.inkMid)
                .tracking(0.5)
            Spacer()
        }
        .padding(.horizontal, 20)
        .padding(.bottom, 6)
    }

    // MARK: - Catalog grid

    private var columns: [GridItem] {
        [GridItem(.flexible(), spacing: 14), GridItem(.flexible(), spacing: 14)]
    }

    @ViewBuilder private var catalog: some View {
        if isSearchActive && allDisplayedProducts.isEmpty && !isSearching {
            Spacer()
            Text("No matches found")
                .font(.system(size: 15, design: .serif))
                .foregroundStyle(LP.inkMid)
                .tracking(1)
            Spacer()
        } else {
            VStack(spacing: 0) {
                ScrollView {
                    LazyVGrid(columns: columns, spacing: 14) {
                        ForEach(displayedProducts, id: \.documentID) { product in
                            ProductTile(product: product,
                                        image: thumbnails[product.documentID])
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 4)
                    .padding(.bottom, 12)
                }

                if totalPages > 1 {
                    paginationBar
                }
            }
        }
    }

    // MARK: - Pagination

    private var paginationBar: some View {
        HStack(spacing: 16) {
            Button {
                currentPage -= 1
            } label: {
                Image(systemName: "chevron.left")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(currentPage > 0 ? LP.gold : LP.inkMid.opacity(0.3))
            }
            .disabled(currentPage == 0)

            Text("Page \(currentPage + 1) of \(totalPages)")
                .font(.system(size: 12, design: .serif))
                .foregroundStyle(LP.inkMid)
                .tracking(0.5)

            Button {
                currentPage += 1
            } label: {
                Image(systemName: "chevron.right")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(currentPage < totalPages - 1 ? LP.gold : LP.inkMid.opacity(0.3))
            }
            .disabled(currentPage >= totalPages - 1)
        }
        .padding(.vertical, 12)
        .padding(.horizontal, 20)
        .background(LP.ivory.opacity(0.95))
        .overlay(
            Rectangle()
                .frame(height: 1)
                .foregroundStyle(LP.gridLine),
            alignment: .top
        )
    }

    // MARK: - Status / loading

    private func statusView(_ text: String, isError: Bool) -> some View {
        VStack(spacing: 16) {
            Spacer()
            if isError {
                Image(systemName: "exclamationmark.triangle")
                    .font(.system(size: 36))
                    .foregroundStyle(LP.gold)
                Text(text)
                    .font(.system(size: 14, design: .serif))
                    .foregroundStyle(LP.inkMid)
                    .multilineTextAlignment(.center)
            } else {
                ProgressView().tint(LP.gold)
                Text(text)
                    .font(.system(size: 14, design: .serif))
                    .foregroundStyle(LP.inkMid)
                    .tracking(1)
            }
            Spacer()
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Lifecycle

    private func setup() async {
        await embedder.load()
        guard embedder.isLoaded else {
            setupError = embedder.loadError ?? "Failed to load models"
            return
        }
        do {
            let store = try InventoryStore()
            self.store = store
            if store.isEmpty { try seed(into: store) }
            refresh(store)

            let apptStore = try AppointmentStore()
            if apptStore.isEmpty { try AppointmentSeeder(store: apptStore).seedAll() }
            try apptStore.reassignDatesIfNeeded()
            appointmentStore = apptStore
        } catch {
            setupError = error.localizedDescription
        }
    }

    private func seed(into store: InventoryStore) throws {
        isSeeding = true
        defer { isSeeding = false }
        let seeder = InventorySeeder(store: store, embedder: embedder)
        _ = try seeder.seedAll()
    }

    private func refresh(_ store: InventoryStore) {
        do {
            let loaded = try store.allProducts()
            products = loaded
            var byID: [String: Product] = [:]
            var cache: [String: UIImage] = [:]
            for product in loaded {
                byID[product.documentID] = product
                if let image = UIImage(data: product.primaryImage) {
                    cache[product.documentID] = image
                }
            }
            productsByID = byID
            thumbnails = cache
        } catch {
            setupError = error.localizedDescription
        }
    }

    // MARK: - Actions

    private func runSearch() {
        guard let store, !queryText.trimmingCharacters(in: .whitespaces).isEmpty else { return }
        queryImage = nil
        isSearching = true
        searchError = nil
        Task {
            defer { isSearching = false; hasSearched = true; currentPage = 0 }
            do {
                let queryVector = try embedder.embed(text: queryText).embedding
                results = try store.search(queryVector: queryVector, limit: 5)
            } catch {
                searchError = error.localizedDescription
                results = []
            }
        }
    }

    private func runImageSearch(_ image: UIImage) {
        guard let store else { return }
        queryText = ""
        queryImage = image
        isSearching = true
        searchError = nil
        Task {
            defer { isSearching = false; hasSearched = true; currentPage = 0 }
            do {
                let queryVector = try embedder.embed(image: image).embedding
                results = try store.search(queryVector: queryVector, limit: 5)
            } catch {
                searchError = error.localizedDescription
                results = []
            }
        }
    }

    private func clearSearch() {
        queryText   = ""
        queryImage  = nil
        results     = []
        searchError = nil
        hasSearched = false
        currentPage = 0
    }

    private func reseed() {
        guard let store else { return }
        Task {
            do {
                try store.deleteAll()
                try seed(into: store)
                refresh(store)
                clearSearch()
            } catch {
                setupError = error.localizedDescription
            }
        }
    }

    private func reseedAppointments() {
        guard let apptStore = appointmentStore else { return }
        Task {
            do {
                try apptStore.deleteAll()
                try AppointmentSeeder(store: apptStore).seedAll()
                try apptStore.reassignDatesIfNeeded()
                scheduleRefreshID = UUID()
            } catch {
                setupError = error.localizedDescription
            }
        }
    }
}

// ── Product tile ───────────────────────────────────────────────────────────────────

private struct ProductTile: View {
    let product: Product
    let image:   UIImage?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            thumbnail

            VStack(alignment: .leading, spacing: 4) {
                Text(product.name)
                    .font(.system(size: 13, weight: .semibold, design: .serif))
                    .foregroundStyle(LP.inkDark)
                    .lineLimit(1)

                Text(product.description)
                    .font(.system(size: 11))
                    .foregroundStyle(LP.inkMid)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)

                Text(priceText)
                    .font(.system(size: 13, weight: .medium, design: .serif))
                    .foregroundStyle(LP.gold)
                    .padding(.top, 4)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 10)
        }
        .background(LP.card)
        .clipShape(RoundedRectangle(cornerRadius: 4))
        .shadow(color: LP.inkDark.opacity(0.10), radius: 6, x: 0, y: 2)
    }

    @ViewBuilder private var thumbnail: some View {
        ZStack {
            if let image {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFill()
            } else {
                LP.ivory
                    .overlay(
                        Image(systemName: "photo")
                            .foregroundStyle(LP.inkMid.opacity(0.4))
                    )
            }
        }
        .frame(height: 160)
        .frame(maxWidth: .infinity)
        .clipped()
    }

    private var priceText: String {
        String(format: "%@ %.2f", product.price.currency, product.price.amount)
    }
}

// ── Assistant view ────────────────────────────────────────────────────────────────

private struct AssistantView: View {

    let store:            InventoryStore?
    let appointmentStore: AppointmentStore?
    let embedder:         SigLIPEmbedder
    let productsByID:     [String: Product]

    struct ChatMessage: Identifiable {
        let id          = UUID()
        var text:         String
        let isUser:       Bool
        var isStreaming:  Bool = false
    }

    @ObservedObject private var llamaEngine = LlamaEngine.shared
    @State private var messages: [ChatMessage] = [
        ChatMessage(text: "How can I help you today?", isUser: false)
    ]
    @State private var inputText          = ""
    @State private var isThinking         = false
    @State private var llamaConversation:  LlamaConversation?

    private static let systemPrompt = """
        You are a knowledgeable fashion retail assistant. \
        You help store staff with questions about inventory, \
        appointments, and fashion recommendations. \
        Be helpful and answer as concise as possible. \
        When inventory or appointment context is provided below, use it to answer precisely, do not hallucinate. \
        If you lack sufficient information, say so gracefully.
        """

    private var activeEngineIsReady: Bool {
        llamaEngine.isReady
    }

    // Normalises the LoadState enum into one representation for the banner.
    private enum BannerState {
        case loading(String), failed(String), hidden
    }
    private var bannerState: BannerState {
        switch llamaEngine.loadState {
        case .loading:           return .loading("Llama 3.2")
        case .failed(let msg):   return .failed(msg)
        default:                 return .hidden
        }
    }

    // MARK: - Body

    var body: some View {
        VStack(spacing: 0) {
            enginePicker
            engineStatusBanner

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        ForEach(messages) { msg in
                            messageBubble(msg).id(msg.id)
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 20)
                }
                .onChange(of: messages.count) {
                    if let last = messages.last {
                        withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                    }
                }
                .onChange(of: messages.last?.text) {
                    if let last = messages.last {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }

            inputBar
        }
        .task {
            await loadActiveEngine()
            await startConversationIfNeeded()
        }
        .onChange(of: llamaEngine.isReady) {
            Task { await startConversationIfNeeded() }
        }
    }

    // MARK: - Engine picker

    private var enginePicker: some View {
        HStack(spacing: 0) {
            Text("Llama 3.2")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(LP.topBar)
                .tracking(0.8)
                .padding(.vertical, 7)
                .padding(.horizontal, 16)
                .background(LP.gold, in: Capsule())

            Spacer()

            Button(action: clearConversation) {
                Image(systemName: "xmark")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(LP.gold)
            }
            .buttonStyle(.plain)
            .disabled(isThinking)
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(LP.ivory)
        .overlay(Rectangle().frame(height: 1).foregroundStyle(LP.gridLine), alignment: .bottom)
    }

    private func clearConversation() {
        messages = [ChatMessage(text: "How can I help you today?", isUser: false)]
        llamaConversation = nil
        Task { await startConversationIfNeeded() }
    }

    // MARK: - Load / Conversation management

    private func loadActiveEngine() async {
        if case .idle = llamaEngine.loadState { await llamaEngine.load() }
    }

    private func startConversationIfNeeded() async {
        guard llamaEngine.isReady, llamaConversation == nil else { return }
        llamaConversation = try? await llamaEngine.newConversation(systemPrompt: Self.systemPrompt)
    }

    // MARK: - Send

    private func sendMessage() {
        let text = inputText.trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty, activeEngineIsReady, !isThinking else { return }
        messages.append(ChatMessage(text: text, isUser: true))
        inputText  = ""
        isThinking = true

        Task {
            let context    = await buildRAGContext(for: text)
            let fullPrompt = context.isEmpty ? text : "\(context)\n\nQuestion: \(text)"

            let placeholder = ChatMessage(text: "", isUser: false, isStreaming: true)
            messages.append(placeholder)
            let idx = messages.count - 1

            do {
                let conv: LlamaConversation
                if let existing = llamaConversation {
                    conv = existing
                } else {
                    let fresh = try await llamaEngine.newConversation(systemPrompt: Self.systemPrompt)
                    llamaConversation = fresh
                    conv = fresh
                }
                for try await token in llamaEngine.send(fullPrompt, in: conv) {
                    messages[idx].text += token
                }
                messages[idx].isStreaming = false
            } catch {
                messages[idx].text        = "Error: \(error.localizedDescription)"
                messages[idx].isStreaming = false
            }
            isThinking = false
        }
    }

    // MARK: - RAG context builder

    private static let inventoryKeywords: [String] = [
        "product", "item", "wear", "buy", "stock", "inventory", "recommend", "suggest",
        "outfit", "style", "look", "dress", "shirt", "blouse", "jacket", "coat", "trousers",
        "pants", "skirt", "shoes", "bag", "accessory", "accessories", "scarf", "belt",
        "suit", "collection", "price", "cost", "colour", "color", "size", "fabric",
        "silk", "wool", "cotton", "leather", "cashmere", "linen",
        "catalog"
    ]

    private func looksLikeInventoryQuery(_ query: String) -> Bool {
        let lower = query.lowercased()
        return Self.inventoryKeywords.contains { lower.contains($0) }
    }

    private func buildRAGContext(for query: String) async -> String {
        var sections: [String] = []

        // Vector search against inventory — only when the query is product-related
        if looksLikeInventoryQuery(query),
           let store,
           let vector = try? embedder.embed(text: query).embedding {
            let hits   = (try? store.search(queryVector: vector, limit: 5)) ?? []
            let lines  = hits.compactMap { r -> String? in
                guard let p = productsByID[r.id] else { return nil }
                return "  • \(p.name): \(p.description) (\(p.price.currency)\(String(format: "%.2f", p.price.amount)))"
            }
            if !lines.isEmpty {
                sections.append("Relevant inventory:\n" + lines.joined(separator: "\n"))
            }
        }

        // Today's appointments
        if let appointmentStore,
           let appts = try? appointmentStore.fetchAppointments(for: Date()), !appts.isEmpty {
            let lines = appts.map { a -> String in
                var line = "  • \(a.time)  \(a.name)  (size: \(a.size))"
                if !a.previousPurchases.isEmpty {
                    line += "  |  prev: \(a.previousPurchases.joined(separator: ", "))"
                }
                if let summary = a.tasteSummary {
                    line += "  |  taste: \(summary)"
                }
                return line
            }
            sections.append("Today's appointments:\n" + lines.joined(separator: "\n"))
        }

        return sections.joined(separator: "\n\n")
    }

    // MARK: - Sub-views

    @ViewBuilder private var engineStatusBanner: some View {
        switch bannerState {
        case .loading(let label):
            HStack(spacing: 8) {
                ProgressView().tint(LP.gold).scaleEffect(0.8)
                Text("Loading \(label)…")
                    .font(.system(size: 12, design: .serif))
                    .foregroundStyle(LP.inkMid)
            }
            .padding(.vertical, 8)
            .frame(maxWidth: .infinity)
            .background(LP.card)
        case .failed(let msg):
            Text(msg)
                .font(.system(size: 11))
                .foregroundStyle(.red)
                .multilineTextAlignment(.center)
                .padding(8)
                .frame(maxWidth: .infinity)
                .background(LP.card)
        case .hidden:
            EmptyView()
        }
    }

    @ViewBuilder private func messageBubble(_ message: ChatMessage) -> some View {
        HStack(alignment: .bottom) {
            if message.isUser { Spacer(minLength: 60) }

            VStack(alignment: message.isUser ? .trailing : .leading, spacing: 4) {
                if message.isStreaming && message.text.isEmpty {
                    HStack(spacing: 8) {
                        ProgressView().tint(LP.gold).scaleEffect(0.8)
                        Text("Thinking…")
                            .font(.system(size: 13, design: .serif))
                            .foregroundStyle(LP.inkMid)
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .background(LP.card, in: RoundedRectangle(cornerRadius: 16))
                    .overlay(
                        RoundedRectangle(cornerRadius: 16)
                            .stroke(LP.gridLine, lineWidth: 1)
                    )
                } else {
                    Text(message.text)
                        .font(.system(size: 14, design: .serif))
                        .foregroundStyle(message.isUser ? LP.ivory : LP.inkDark)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                        .background(
                            message.isUser ? LP.topBar : LP.card,
                            in: RoundedRectangle(cornerRadius: 16)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 16)
                                .stroke(message.isUser ? LP.gold.opacity(0.4) : LP.gridLine,
                                        lineWidth: 1)
                        )
                }
                if message.isStreaming && !message.text.isEmpty {
                    ProgressView().tint(LP.gold).scaleEffect(0.6)
                }
            }

            if !message.isUser { Spacer(minLength: 60) }
        }
    }

    private var inputBar: some View {
        HStack(spacing: 10) {
            TextField("", text: $inputText,
                      prompt: Text("Ask about inventory, style…")
                          .foregroundStyle(LP.inkMid.opacity(0.7)))
                .foregroundStyle(LP.inkDark)
                .tint(LP.gold)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .submitLabel(.send)
                .onSubmit(sendMessage)
                .disabled(isThinking)
                .padding(.horizontal, 14)
                .padding(.vertical, 11)
                .background(LP.card, in: Capsule())
                .overlay(Capsule().stroke(LP.gold.opacity(0.35), lineWidth: 1))

            Button(action: sendMessage) {
                Image(systemName: isThinking ? "ellipsis.circle.fill" : "arrow.up.circle.fill")
                    .font(.system(size: 32))
                    .foregroundStyle(
                        (inputText.trimmingCharacters(in: .whitespaces).isEmpty || isThinking
                         || !activeEngineIsReady)
                            ? LP.inkMid.opacity(0.3) : LP.gold
                    )
            }
            .disabled(inputText.trimmingCharacters(in: .whitespaces).isEmpty
                      || isThinking || !activeEngineIsReady)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(LP.ivory)
        .overlay(Rectangle().frame(height: 1).foregroundStyle(LP.gridLine), alignment: .top)
    }
}

// ── Schedule view ─────────────────────────────────────────────────────────────────

private struct ScheduleView: View {

    let store:          AppointmentStore?
    let inventoryStore: InventoryStore?
    let embedder:       SigLIPEmbedder

    @State private var selectedDate  = Date()
    @State private var appointments: [Appointment] = []

    var body: some View {
        VStack(spacing: 0) {
            // ── Date picker header
            HStack(spacing: 8) {
                Text("Customer Appointments on")
                    .font(.system(size: 15, weight: .semibold, design: .serif))
                    .foregroundStyle(LP.inkDark)
                    .tracking(0.5)
                DatePicker("", selection: $selectedDate, displayedComponents: .date)
                    .labelsHidden()
                    .tint(LP.gold)
                Spacer()
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 14)
            .background(LP.ivory)
            .overlay(
                Rectangle().frame(height: 1).foregroundStyle(LP.gridLine),
                alignment: .bottom
            )

            // ── Appointment cards
            if appointments.isEmpty {
                Spacer()
                Text("No appointments scheduled")
                    .font(.system(size: 15, design: .serif))
                    .foregroundStyle(LP.inkMid)
                    .tracking(1)
                Spacer()
            } else {
                ScrollView {
                    VStack(spacing: 16) {
                        ForEach(appointments) { appt in
                            AppointmentCard(appointment: appt, inventoryStore: inventoryStore, embedder: embedder)
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 20)
                }
            }
        }
        .onAppear { loadAppointments() }
        .onChange(of: selectedDate) { loadAppointments() }
        .onChange(of: store == nil) { loadAppointments() }
    }

    private func loadAppointments() {
        appointments = (try? store?.fetchAppointments(for: selectedDate)) ?? []
    }
}

// ── Appointment card ──────────────────────────────────────────────────────────────

private struct AppointmentCard: View {

    let appointment:    Appointment
    let inventoryStore: InventoryStore?
    let embedder:       SigLIPEmbedder

    @State private var showingRecommendation = false

    private static let inputFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm"
        return f
    }()

    private static let displayFormatter: DateFormatter = {
        let f = DateFormatter()
        f.timeStyle = .short
        return f
    }()

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {

            // ── Time header (dark bar)
            HStack(spacing: 8) {
                Image(systemName: "clock")
                    .foregroundStyle(LP.gold)
                    .font(.system(size: 13, weight: .medium))
                Text(formattedTime)
                    .font(.system(size: 14, weight: .semibold, design: .serif))
                    .foregroundStyle(LP.goldLight)
                    .tracking(0.5)
                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 11)
            .background(LP.topBar)

            // ── Card body
            VStack(alignment: .leading, spacing: 14) {

                // Customer name + gender
                HStack(spacing: 10) {
                    Image(systemName: "person")
                        .foregroundStyle(LP.gold)
                        .font(.system(size: 14))
                        .frame(width: 18)
                    Text(appointment.name)
                        .font(.system(size: 17, weight: .semibold, design: .serif))
                        .foregroundStyle(LP.inkDark)
                        .tracking(0.5)
                    Spacer()
                    Text(appointment.gender.lowercased() == "male" ? "M" : "F")
                        .font(.system(size: 11, weight: .semibold, design: .serif))
                        .foregroundStyle(LP.gold)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(LP.ivory, in: RoundedRectangle(cornerRadius: 4))
                        .overlay(RoundedRectangle(cornerRadius: 4).stroke(LP.gridLine, lineWidth: 1))
                }

                hairline

                // Size
                infoRow(icon: "ruler") {
                    VStack(alignment: .leading, spacing: 6) {
                        sectionLabel("Size")
                        Text(appointment.size)
                            .font(.system(size: 13, weight: .semibold, design: .serif))
                            .foregroundStyle(LP.inkDark)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 4)
                            .background(LP.ivory, in: RoundedRectangle(cornerRadius: 5))
                            .overlay(
                                RoundedRectangle(cornerRadius: 5)
                                    .stroke(LP.gridLine, lineWidth: 1)
                            )
                    }
                }

                hairline

                // Previous purchases
                infoRow(icon: "bag") {
                    VStack(alignment: .leading, spacing: 5) {
                        sectionLabel("Previous Purchases")
                        ForEach(appointment.previousPurchases, id: \.self) { item in
                            HStack(spacing: 8) {
                                Circle()
                                    .fill(LP.gold)
                                    .frame(width: 4, height: 4)
                                Text(item)
                                    .font(.system(size: 13, design: .serif))
                                    .foregroundStyle(LP.inkDark)
                            }
                        }
                    }
                }

                // Taste summary (only shown when present)
                if let summary = appointment.tasteSummary {
                    hairline
                    infoRow(icon: "sparkles") {
                        VStack(alignment: .leading, spacing: 5) {
                            sectionLabel("Taste Summary")
                            Text(summary)
                                .font(.system(size: 13, design: .serif))
                                .foregroundStyle(LP.inkDark)
                                .lineSpacing(3)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }

                // Customer Insights (only shown when present)
                if let insight = appointment.customerInsight {
                    hairline
                    infoRow(icon: "lightbulb") {
                        VStack(alignment: .leading, spacing: 5) {
                            sectionLabel("Customer Insights")
                            Text(insight)
                                .font(.system(size: 13, design: .serif))
                                .foregroundStyle(LP.inkDark)
                                .lineSpacing(3)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }

                // Show Recommendation button
                Button {
                    showingRecommendation = true
                } label: {
                    HStack(spacing: 8) {
                        Spacer()
                        Image(systemName: "wand.and.stars")
                            .font(.system(size: 13, weight: .semibold))
                        Text("Show Recommendation")
                            .font(.system(size: 14, weight: .semibold, design: .serif))
                            .tracking(0.5)
                        Spacer()
                    }
                    .foregroundStyle(LP.topBar)
                    .padding(.vertical, 13)
                    .background(LP.gold, in: RoundedRectangle(cornerRadius: 7))
                }
                .buttonStyle(.plain)
                .padding(.top, 2)
                .sheet(isPresented: $showingRecommendation) {
                    RecommendationSheet(appointment: appointment, inventoryStore: inventoryStore, embedder: embedder)
                }
            }
            .padding(16)
        }
        .background(LP.card)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .shadow(color: LP.inkDark.opacity(0.10), radius: 8, x: 0, y: 2)
    }

    private var formattedTime: String {
        guard let date = Self.inputFormatter.date(from: appointment.time) else {
            return appointment.time
        }
        return Self.displayFormatter.string(from: date)
    }

    private var hairline: some View {
        Rectangle()
            .frame(height: 1)
            .foregroundStyle(LP.gridLine)
    }

    private func sectionLabel(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 10, weight: .medium))
            .foregroundStyle(LP.inkMid)
            .tracking(1.2)
            .textCase(.uppercase)
    }

    @ViewBuilder
    private func infoRow<Content: View>(icon: String, @ViewBuilder content: () -> Content) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: icon)
                .foregroundStyle(LP.gold)
                .font(.system(size: 14))
                .frame(width: 18)
                .padding(.top, 1)
            content()
        }
    }
}

// ── Recommendation sheet ──────────────────────────────────────────────────────────

private struct RecommendationSheet: View {

    let appointment:    Appointment
    let inventoryStore: InventoryStore?
    let embedder:       SigLIPEmbedder

    @Environment(\.dismiss) private var dismiss
    @ObservedObject private var engine = LlamaEngine.shared

    @State private var recommendation = ""
    @State private var isStreaming     = false
    @State private var genError:  String?
    @State private var ragProducts: [(product: Product, distance: Float)] = []
    @State private var ragImages:   [String: UIImage] = [:]
    @State private var isLoadingRAG = false

    var body: some View {
        ZStack(alignment: .top) {
            DiamondBackground()

            VStack(spacing: 0) {

                // ── Header bar
                HStack {
                    Spacer()
                    Button { dismiss() } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 24))
                            .foregroundStyle(LP.inkMid.opacity(0.6))
                    }
                    .buttonStyle(.plain)
                }
                .padding(.horizontal, 20)
                .padding(.top, 16)
                .padding(.bottom, 8)

                ScrollView {
                    VStack(alignment: .leading, spacing: 24) {

                        // Customer name
                        HStack(spacing: 12) {
                            Image(systemName: "person.crop.circle")
                                .font(.system(size: 28))
                                .foregroundStyle(LP.gold)
                            Text(appointment.name)
                                .font(.system(size: 22, weight: .semibold, design: .serif))
                                .foregroundStyle(LP.inkDark)
                                .tracking(0.5)
                        }
                        .padding(.bottom, 4)

                        hairline

                        // ── Styling recommendation
                        VStack(alignment: .leading, spacing: 10) {
                            sectionHeader(icon: "wand.and.stars", title: "Styling Recommendation")

                            if isLoadingRAG || engine.loadState == .loading || (isStreaming && recommendation.isEmpty) {
                                HStack(spacing: 8) {
                                    ProgressView().tint(LP.gold)
                                    Text(loadingMessage)
                                        .font(.system(size: 13, design: .serif))
                                        .foregroundStyle(LP.inkMid)
                                }
                                .padding(.top, 4)
                            } else if let err = genError {
                                Text(err)
                                    .font(.system(size: 13))
                                    .foregroundStyle(.red)
                            } else {
                                Text(recommendation)
                                    .font(.system(size: 14, design: .serif))
                                    .foregroundStyle(LP.inkDark)
                                    .lineSpacing(4)
                                    .fixedSize(horizontal: false, vertical: true)
                                if isStreaming {
                                    ProgressView().tint(LP.gold).scaleEffect(0.7)
                                }
                            }
                        }

                        // ── Matched product cards
                        if !ragProducts.isEmpty {
                            hairline
                            sectionHeader(icon: "tag", title: "Matched Products")
                            ScrollView(.horizontal, showsIndicators: false) {
                                HStack(spacing: 12) {
                                    ForEach(ragProducts, id: \.product.documentID) { item in
                                        RecommendationProductCard(
                                            product: item.product,
                                            image: ragImages[item.product.documentID]
                                        )
                                    }
                                }
                                .padding(.horizontal, 2)
                                .padding(.bottom, 4)
                            }
                        }
                    }
                    .padding(.horizontal, 24)
                    .padding(.vertical, 20)
                }
            }
        }
        .task {
            if case .idle = engine.loadState { await engine.load() }
            await runRAGAndGenerate()
        }
        .onChange(of: engine.isReady) { Task { await runRAGAndGenerate() } }
    }

    // MARK: - Loading state

    private var loadingMessage: String {
        if isLoadingRAG { return "Searching inventory…" }
        if engine.loadState == .loading { return "Loading Llama…" }
        return "Generating…"
    }

    // MARK: - RAG + Generation

    private func runRAGAndGenerate() async {
        guard engine.isReady, recommendation.isEmpty, !isStreaming, !isLoadingRAG else { return }

        guard let insight = appointment.customerInsight, !insight.isEmpty else {
            genError = "No customer insights available for this customer."
            return
        }

        guard let store = inventoryStore, embedder.isLoaded else {
            genError = "Inventory not ready. Please wait for setup to complete."
            return
        }

        isLoadingRAG = true
        genError = nil

        do {
            let genderPrefix = "Gender: \(appointment.gender). "
            let emb  = try embedder.embed(text: genderPrefix + insight)
            let hits = try store.search(queryVector: emb.embedding, limit: 10)

            var loaded: [(product: Product, distance: Float)] = []
            var images:  [String: UIImage] = [:]
            for hit in hits {
                if let product = try store.product(id: hit.id) {
                    loaded.append((product: product, distance: hit.distance))
                    if let img = UIImage(data: product.primaryImage) {
                        images[product.documentID] = img
                    }
                }
            }
            ragProducts = loaded
            ragImages   = images
        } catch {
            genError = error.localizedDescription
            isLoadingRAG = false
            return
        }

        isLoadingRAG = false
        await generate(products: ragProducts)
    }

    private func generate(products: [(product: Product, distance: Float)]) async {
        guard engine.isReady, recommendation.isEmpty, !isStreaming else { return }
        isStreaming = true
        genError    = nil

        let systemPrompt = """
            You are a helpful assistant for a store clerk preparing for a customer appointment. \
            You will receive the customer's preference profile and a numbered list of specific \
            products retrieved from the store inventory. \
            Write a concise recommendation explaining why those exact products suit the customer. \
            You MUST only reference products from the list below — never invent or mention any \
            other items. Do not add any preamble, introduction, postscript, or meta-commentary. \
            Output only the recommendation itself. Be polite and concise.
            """

        let userPrompt = buildUserPrompt(products: products)

        do {
            for try await token in engine.oneShot(systemPrompt: systemPrompt, userPrompt: userPrompt) {
                recommendation += token
            }
        } catch {
            genError = error.localizedDescription
        }
        isStreaming = false
    }

    private func buildUserPrompt(products: [(product: Product, distance: Float)]) -> String {
        var lines: [String] = []
        if let insight = appointment.customerInsight {
            lines.append("Customer preference profile:\n\(insight)")
        }
        if !products.isEmpty {
            lines.append("\nProducts retrieved from inventory (reference ONLY these \(products.count) items):")
            for (i, item) in products.enumerated() {
                let p = item.product
                lines.append("\(i + 1). \(p.name) — \(p.category.department) / \(p.category.type). \(p.description) Color: \(p.color). Material: \(p.material). Price: \(p.price.currency) \(String(format: "%.2f", p.price.amount)).")
            }
            lines.append("\nWrite your recommendation based solely on the \(products.count) products listed above.")
        }
        return lines.joined(separator: "\n")
    }

    // MARK: - Sub-views

    private var hairline: some View {
        Rectangle().frame(height: 1).foregroundStyle(LP.gridLine)
    }

    private func sectionHeader(icon: String, title: String) -> some View {
        HStack(spacing: 10) {
            Image(systemName: icon)
                .font(.system(size: 15, weight: .medium))
                .foregroundStyle(LP.gold)
            Text(title)
                .font(.system(size: 15, weight: .semibold, design: .serif))
                .foregroundStyle(LP.inkDark)
                .tracking(0.8)
        }
    }
}

// ── Recommendation product card ───────────────────────────────────────────────────

private struct RecommendationProductCard: View {

    let product: Product
    let image:   UIImage?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {

            // Thumbnail
            Group {
                if let img = image {
                    Image(uiImage: img)
                        .resizable()
                        .scaledToFill()
                } else {
                    LP.ivory
                        .overlay(
                            Image(systemName: "photo")
                                .foregroundStyle(LP.gridLine)
                        )
                }
            }
            .frame(width: 110, height: 130)
            .clipped()

            // Info
            VStack(alignment: .leading, spacing: 3) {
                Text(product.name)
                    .font(.system(size: 11, weight: .semibold, design: .serif))
                    .foregroundStyle(LP.inkDark)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
                Text("\(product.category.department) · \(product.category.type)")
                    .font(.system(size: 10))
                    .foregroundStyle(LP.inkMid)
                    .lineLimit(1)
                Text(priceText)
                    .font(.system(size: 11, weight: .semibold, design: .serif))
                    .foregroundStyle(LP.gold)
            }
            .padding(8)
            .frame(width: 110, alignment: .leading)
        }
        .frame(width: 110)
        .background(LP.card)
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .shadow(color: LP.inkDark.opacity(0.10), radius: 4, x: 0, y: 1)
    }

    private var priceText: String {
        String(format: "%@ %.2f", product.price.currency, product.price.amount)
    }
}

// ── Image picker ──────────────────────────────────────────────────────────────────

extension UIImagePickerController.SourceType: Identifiable {
    public var id: Int { rawValue }
}

struct ImagePicker: UIViewControllerRepresentable {
    let sourceType: UIImagePickerController.SourceType
    let onImage: (UIImage) -> Void

    @Environment(\.dismiss) private var dismiss

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = sourceType
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ controller: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    final class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        private let parent: ImagePicker
        init(_ parent: ImagePicker) { self.parent = parent }

        func imagePickerController(_ picker: UIImagePickerController,
                                   didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            if let image = info[.originalImage] as? UIImage { parent.onImage(image) }
            parent.dismiss()
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.dismiss()
        }
    }
}
