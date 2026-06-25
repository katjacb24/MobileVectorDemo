import Foundation

// ── AppointmentSeeder ───────────────────────────────────────────────────────────
// Reads Customers/customers.json from the app bundle and upserts each entry into
// the `store.appointments` Couchbase Lite collection via AppointmentStore.
// Call seedAll() only when AppointmentStore.isEmpty is true (safe on every launch).
final class AppointmentSeeder {

    private struct CustomerJSON: Decodable {
        let name:              String
        let gender:            String
        let previousPurchases: [String]
        let size:              String
        let time:              String
        let date:              String
        let customerInsight:   String?
    }

    private let store: AppointmentStore

    init(store: AppointmentStore) { self.store = store }

    @discardableResult
    func seedAll() throws -> Int {
        guard let url = Bundle.main.url(forResource: "customers",
                                        withExtension: "json",
                                        subdirectory: "Customers") else {
            print("AppointmentSeeder: Customers/customers.json not found in bundle")
            return 0
        }
        let customers = try JSONDecoder().decode([CustomerJSON].self, from: Data(contentsOf: url))
        for (i, c) in customers.enumerated() {
            try store.upsert(Appointment(
                id:                String(i + 1),
                name:              c.name,
                gender:            c.gender,
                size:              c.size,
                previousPurchases: c.previousPurchases,
                customerInsight:   c.customerInsight,
                tasteSummary:      nil,
                time:              c.time,
                date:              c.date
            ))
        }
        return customers.count
    }
}
