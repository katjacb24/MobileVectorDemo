import UIKit

// ── InventorySeeder ─────────────────────────────────────────────────────────────
// Populates the `store.inventory` collection from bundled data files rather than a
// hard-coded array:
//
//   • Catalog/catalog.json  — the product list (the fields below, minus the image)
//   • Catalog/Images/<file> — the product photos, referenced by `image` in the JSON
//
// Both live in the `Catalog` folder reference that ships inside the app bundle, so new
// products are added by editing catalog.json and dropping an image into Catalog/Images
// — no code changes. For each item the seeder loads the photo, computes the SigLIP
// vision embedding, and upserts a Product (image stored as a Blob, vector in
// `image_embedding`).
//
// @MainActor because SigLIPEmbedder is main-actor isolated (Core ML inference here runs
// on the main actor; fine for a handful of seed items in a test harness).
@MainActor
final class InventorySeeder {

    // One catalog entry as described in catalog.json. `image` is a filename (e.g.
    // "blue_shirt.jpg") resolved against Catalog/Images. Everything else maps 1:1 onto
    // Product; the embedding and Blob are produced at seed time.
    struct SeedItem: Decodable {
        let image:       String
        let sku:         String
        let name:        String
        let description: String
        let category:    Product.Category
        let color:       String
        let material:    String
        let sizes:       [String]
        let price:       Product.Price
        let tags:        [String]
        let stock:       Int
    }

    enum SeedError: LocalizedError {
        case catalogMissing
        var errorDescription: String? {
            switch self {
            case .catalogMissing:
                return "Catalog/catalog.json was not found in the app bundle."
            }
        }
    }

    // Folder-reference paths inside the bundle (the `Catalog` folder is copied as-is).
    private static let catalogSubdirectory = "Catalog"
    private static let imagesSubdirectory  = "Catalog/Images"
    private static let catalogResourceName = "catalog"

    // Reads and decodes Catalog/catalog.json from the app bundle.
    static func loadCatalog() throws -> [SeedItem] {
        guard let url = Bundle.main.url(forResource: catalogResourceName,
                                        withExtension: "json",
                                        subdirectory: catalogSubdirectory) else {
            throw SeedError.catalogMissing
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode([SeedItem].self, from: data)
    }

    private let store:    InventoryStore
    private let embedder: SigLIPEmbedder

    init(store: InventoryStore, embedder: SigLIPEmbedder) {
        self.store    = store
        self.embedder = embedder
    }

    // Seeds only when the collection is empty (safe to call on every launch).
    @discardableResult
    func seedIfEmpty() throws -> Int {
        guard store.isEmpty else { return 0 }
        return try seedAll()
    }

    // Embeds and upserts every catalog item. Returns the number of products written.
    @discardableResult
    func seedAll() throws -> Int {
        var seeded = 0
        for item in try Self.loadCatalog() {
            guard let image = Self.loadImage(named: item.image) else {
                print("InventorySeeder: missing image '\(item.image)' in \(Self.imagesSubdirectory), skipping")
                continue
            }
            guard let jpeg = image.jpegData(compressionQuality: 0.9) else {
                print("InventorySeeder: could not JPEG-encode '\(item.image)', skipping")
                continue
            }

            let embedding = try embedder.embed(image: image).embedding
            let product = Product(
                sku:            item.sku,
                name:           item.name,
                description:    item.description,
                category:       item.category,
                color:          item.color,
                material:       item.material,
                sizes:          item.sizes,
                price:          item.price,
                tags:           item.tags,
                stock:          item.stock,
                imageEmbedding: embedding,
                primaryImage:   jpeg,
                altImages:      []
            )
            try store.upsert(product)
            seeded += 1
        }
        return seeded
    }

    // Loads a product photo from Catalog/Images by its filename (with or without an
    // extension), reading the bundled file directly rather than via the asset catalog.
    private static func loadImage(named filename: String) -> UIImage? {
        let name = (filename as NSString).deletingPathExtension
        let ext  = (filename as NSString).pathExtension
        guard let url = Bundle.main.url(forResource: name,
                                        withExtension: ext.isEmpty ? nil : ext,
                                        subdirectory: imagesSubdirectory) else {
            return nil
        }
        return UIImage(contentsOfFile: url.path)
    }
}
