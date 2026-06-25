import SwiftUI

struct ContentView: View {
    var body: some View {
        // The catalog screen renders its own top menu bar, so no NavigationStack chrome.
        StoreAssistantView()
    }
}
