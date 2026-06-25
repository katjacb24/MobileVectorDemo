import SwiftUI
import CouchbaseLiteSwift

@main
struct MobileDemoApp: App {
    init() {
        // The Vector Search extension must be enabled once, before any Couchbase Lite
        // database is opened (Couchbase Lite requirement). Enabling it here in the
        // app's init guarantees it runs before any database access later in the app.
        do {
            try Extension.enableVectorSearch()
        } catch {
            // No database is opened yet, so failing here means the extension binary is
            // missing/mismatched — surface it loudly during development.
            assertionFailure("Failed to enable Couchbase Lite Vector Search: \(error)")
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
