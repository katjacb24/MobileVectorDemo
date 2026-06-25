import Foundation
import CouchbaseLiteSwift

// ── Errors ──────────────────────────────────────────────────────────────────────

enum InventoryError: LocalizedError {
    case invalidEmbeddingDimension(expected: Int, actual: Int)

    var errorDescription: String? {
        switch self {
        case let .invalidEmbeddingDimension(expected, actual):
            return "image_embedding must have \(expected) elements, got \(actual)."
        }
    }
}

// ── Product domain model ────────────────────────────────────────────────────────
// The Swift representation of one inventory document, mapped to/from a Couchbase Lite
// document by InventoryStore. See README.md for the document schema.
//
// `imageEmbedding` is the single per-product 768-dim vector (from the primary image,
// L2-normalised by the SigLIP vision encoder). Images are carried as JPEG `Data` and
// stored as Couchbase Lite Blobs.
struct Product {
    var sku:            String
    var name:           String
    var description:    String
    var category:       Category
    var color:          String
    var material:       String
    var sizes:          [String]
    var price:          Price
    var tags:           [String]
    var stock:          Int
    var imageEmbedding: [Float]   // 768-d, L2-normalised, from the primary image
    var primaryImage:   Data      // JPEG
    var altImages:      [Data]    // additional JPEGs (stored, not embedded)

    // Decodable so the seeder can read them straight out of Catalog/catalog.json;
    // the JSON keys match these property names.
    struct Category: Decodable {
        var department: String    // e.g. "Men"
        var type:       String    // e.g. "Outerwear"
        var subtype:    String    // e.g. "Jackets"
    }

    struct Price: Decodable {
        var amount:   Double
        var currency: String      // ISO 4217, e.g. "USD"
    }

    // Stable document id: `inventory::<sku>`.
    var documentID: String { "inventory::\(sku)" }
}

// ── Search result ───────────────────────────────────────────────────────────────
// One row of a vector-search query: a product plus its distance to the query vector.
// `approx_vector_distance` returns a cosine *distance* (lower = closer); since the
// stored and query vectors are L2-normalised, similarity ≈ 1 − distance.
struct SearchResult: Identifiable {
    let id:            String   // document id, e.g. "inventory::TEE-CN-RED-003"
    let name:          String
    let department:    String
    let priceAmount:   Double
    let priceCurrency: String
    let distance:      Float

    var similarity: Float { 1 - distance }
}

// ── InventoryStore ──────────────────────────────────────────────────────────────
// Phase 1 database layer: opens the `brand` database, ensures the `store.inventory`
// collection exists, and exposes a small CRUD surface. Couchbase Lite database and
// collection objects are thread-safe, so this is a plain class.
//
// Requires Extension.enableVectorSearch() to have been called first (done in
// MobileDemoApp.init). The vector index itself is added in Phase 4.
final class InventoryStore {

    static let databaseName   = "brand"        // CBL Database  (demo term: brand)
    static let scopeName      = "store"        // CBL Scope     (demo term: store)
    static let collectionName = "inventory"    // CBL Collection

    // SigLIP ViT-B/16 output dimensionality. The schema requires `image_embedding`
    // to be exactly this length; the vector index is configured to match.
    static let embeddingDimensions = 768

    // Vector index over `image_embedding`.
    static let vectorIndexName = "imageEmbeddingIndex"
    // Small centroid count for a demo dataset. IVF training needs at least this many
    // vectors; below that, queries still work (full scan). Raise toward √N at scale.
    static let vectorIndexCentroids = 8

    // Document field keys (kept in one place so Phase 4/5 reference the same names).
    enum Field {
        static let type           = "type"
        static let sku            = "sku"
        static let name           = "name"
        static let description     = "description"
        static let category       = "category"
        static let color          = "color"
        static let material       = "material"
        static let sizes          = "sizes"
        static let price          = "price"
        static let tags           = "tags"
        static let stock          = "stock"
        static let imageEmbedding  = "image_embedding"  // the vector field (indexed in Phase 4)
        static let primaryImage    = "primary_image"    // primary image Blob
        static let altImages       = "alt_image"        // array of additional image Blobs
    }

    let database:  Database
    let inventory: Collection

    init() throws {
        database = try Database(name: Self.databaseName)
        // createCollection is idempotent: it returns the existing collection if present,
        // and creates the scope implicitly when it names a new one.
        inventory = try database.createCollection(name: Self.collectionName,
                                                  scope: Self.scopeName)
        try ensureVectorIndex()
    }

    // MARK: - Vector index

    // Creates the cosine vector index over `image_embedding` if it isn't already present.
    // The index can be created before any documents exist — it covers documents as they
    // are added. Requires Extension.enableVectorSearch() to have run first (App init).
    private func ensureVectorIndex() throws {
        guard try !inventory.indexes().contains(Self.vectorIndexName) else { return }
        var config = VectorIndexConfiguration(
            expression: Field.imageEmbedding,
            dimensions: UInt32(Self.embeddingDimensions),
            centroids:  UInt32(Self.vectorIndexCentroids))
        config.metric = .cosine   // .dot is equivalent for L2-normalised vectors
        try inventory.createIndex(withName: Self.vectorIndexName, config: config)
    }

    // MARK: - Create / update

    func upsert(_ product: Product) throws {
        // Enforce the schema contract: the vector must be exactly the model's dimension,
        // otherwise the Phase 4 vector index would reject or mis-handle the document.
        guard product.imageEmbedding.count == Self.embeddingDimensions else {
            throw InventoryError.invalidEmbeddingDimension(
                expected: Self.embeddingDimensions,
                actual: product.imageEmbedding.count)
        }

        let doc = MutableDocument(id: product.documentID)
        doc.setString("product",            forKey: Field.type)
        doc.setString(product.sku,          forKey: Field.sku)
        doc.setString(product.name,         forKey: Field.name)
        doc.setString(product.description,  forKey: Field.description)
        doc.setString(product.color,        forKey: Field.color)
        doc.setString(product.material,     forKey: Field.material)
        doc.setInt(product.stock,           forKey: Field.stock)
        doc.setValue(product.sizes,         forKey: Field.sizes)
        doc.setValue(product.tags,          forKey: Field.tags)

        let category = MutableDictionaryObject()
        category.setString(product.category.department, forKey: "department")
        category.setString(product.category.type,       forKey: "type")
        category.setString(product.category.subtype,    forKey: "subtype")
        doc.setDictionary(category, forKey: Field.category)

        let price = MutableDictionaryObject()
        price.setDouble(product.price.amount,   forKey: "amount")
        price.setString(product.price.currency, forKey: "currency")
        doc.setDictionary(price, forKey: Field.price)

        // Vector stored as a numeric array (the vector index reads it in Phase 4).
        doc.setValue(product.imageEmbedding.map { Double($0) }, forKey: Field.imageEmbedding)

        // Images as Blobs.
        doc.setBlob(Blob(contentType: "image/jpeg", data: product.primaryImage),
                    forKey: Field.primaryImage)
        if !product.altImages.isEmpty {
            let alt = MutableArrayObject()
            for data in product.altImages {
                alt.addBlob(Blob(contentType: "image/jpeg", data: data))
            }
            doc.setArray(alt, forKey: Field.altImages)
        }

        try inventory.save(document: doc)
    }

    // MARK: - Read

    func product(id: String) throws -> Product? {
        guard let doc = try inventory.document(id: id) else { return nil }
        return Self.product(from: doc)
    }

    func allProducts() throws -> [Product] {
        let query = QueryBuilder
            .select(SelectResult.expression(Meta.id))
            .from(DataSource.collection(inventory))

        var products: [Product] = []
        for row in try query.execute() {
            if let id = row.string(forKey: "id"), let product = try product(id: id) {
                products.append(product)
            }
        }
        return products
    }

    var count: Int { Int(inventory.count) }

    var isEmpty: Bool { count == 0 }

    // MARK: - Vector search

    // Ranks inventory products by similarity to `queryVector` (a 768-d SigLIP text or image
    // embedding) using the cosine vector index. Returns the top `limit` matches, closest first.
    func search(queryVector: [Float], limit: Int = 10) throws -> [SearchResult] {
        // `limit` is an Int we control, so inlining it (rather than binding) sidesteps
        // SQL++ restrictions on parameterised LIMIT.
        let sql = """
        SELECT meta(product).id AS id,
               product.\(Field.name) AS name,
               product.\(Field.category).department AS department,
               product.\(Field.price).amount AS amount,
               product.\(Field.price).currency AS currency,
               approx_vector_distance(product.\(Field.imageEmbedding), $vector, "cosine") AS distance
        FROM \(Self.scopeName).\(Self.collectionName) AS product
        ORDER BY distance
        LIMIT \(max(1, limit))
        """

        let query = try database.createQuery(sql)
        let params = Parameters()
        params.setValue(queryVector.map { Double($0) }, forName: "vector")
        query.parameters = params

        var hits: [SearchResult] = []
        for row in try query.execute() {
            guard let id = row.string(forKey: "id") else { continue }
            hits.append(SearchResult(
                id:            id,
                name:          row.string(forKey: "name") ?? "",
                department:    row.string(forKey: "department") ?? "",
                priceAmount:   row.double(forKey: "amount"),
                priceCurrency: row.string(forKey: "currency") ?? "USD",
                distance:      Float(row.double(forKey: "distance"))
            ))
        }
        return hits
    }

    // MARK: - Delete

    func deleteAll() throws {
        for product in try allProducts() {
            if let doc = try inventory.document(id: product.documentID) {
                try inventory.delete(document: doc)
            }
        }
    }

    // MARK: - Mapping

    private static func product(from doc: Document) -> Product? {
        guard let sku = doc.string(forKey: Field.sku),
              let name = doc.string(forKey: Field.name),
              let primaryData = doc.blob(forKey: Field.primaryImage)?.content
        else { return nil }

        let categoryDict = doc.dictionary(forKey: Field.category)
        let category = Product.Category(
            department: categoryDict?.string(forKey: "department") ?? "",
            type:       categoryDict?.string(forKey: "type") ?? "",
            subtype:    categoryDict?.string(forKey: "subtype") ?? ""
        )

        let priceDict = doc.dictionary(forKey: Field.price)
        let price = Product.Price(
            amount:   priceDict?.double(forKey: "amount") ?? 0,
            currency: priceDict?.string(forKey: "currency") ?? "USD"
        )

        var embedding: [Float] = []
        if let arr = doc.array(forKey: Field.imageEmbedding) {
            embedding.reserveCapacity(arr.count)
            for i in 0..<arr.count { embedding.append(Float(arr.double(at: i))) }
        }

        var altImages: [Data] = []
        if let altArr = doc.array(forKey: Field.altImages) {
            for i in 0..<altArr.count {
                if let data = altArr.blob(at: i)?.content { altImages.append(data) }
            }
        }

        let sizes = (doc.array(forKey: Field.sizes)?.toArray() as? [String]) ?? []
        let tags  = (doc.array(forKey: Field.tags)?.toArray()  as? [String]) ?? []

        return Product(
            sku:            sku,
            name:           name,
            description:    doc.string(forKey: Field.description) ?? "",
            category:       category,
            color:          doc.string(forKey: Field.color) ?? "",
            material:       doc.string(forKey: Field.material) ?? "",
            sizes:          sizes,
            price:          price,
            tags:           tags,
            stock:          doc.int(forKey: Field.stock),
            imageEmbedding: embedding,
            primaryImage:   primaryData,
            altImages:      altImages
        )
    }
}
