import Foundation
import CouchbaseLiteSwift

// ── Appointment domain model ────────────────────────────────────────────────────
struct Appointment: Identifiable {
    let id:                String   // document id, e.g. "appointment::1"
    let name:              String
    let gender:            String
    let size:              String
    let previousPurchases: [String]
    let customerInsight:   String?
    let tasteSummary:      String?
    let time:              String   // "HH:mm"
    let date:              String   // "yyyy-MM-dd", or "" meaning unscheduled
}

// ── AppointmentStore ────────────────────────────────────────────────────────────
// Manages the `store.appointments` Couchbase Lite collection in the same `brand`
// database as InventoryStore. Appointments are loaded from Customers/customers.json
// by AppointmentSeeder on first launch.
final class AppointmentStore {

    static let collectionName = "appointments"

    enum Field {
        static let name              = "name"
        static let gender            = "gender"
        static let size              = "size"
        static let previousPurchases = "previousPurchases"
        static let customerInsight   = "customerInsight"
        static let tasteSummary      = "tasteSummary"
        static let time              = "time"
        static let date              = "date"
    }

    private let database:   Database
    private let collection: Collection

    init() throws {
        database   = try Database(name: InventoryStore.databaseName)
        collection = try database.createCollection(name: Self.collectionName,
                                                   scope: InventoryStore.scopeName)
    }

    var isEmpty: Bool { collection.count == 0 }

    // MARK: - Write

    func upsert(_ appointment: Appointment) throws {
        let doc = MutableDocument(id: "appointment::\(appointment.id)")
        doc.setString(appointment.name,   forKey: Field.name)
        doc.setString(appointment.gender, forKey: Field.gender)
        doc.setString(appointment.size,   forKey: Field.size)
        doc.setString(appointment.time,   forKey: Field.time)
        doc.setString(appointment.date,   forKey: Field.date)
        doc.setValue(appointment.previousPurchases, forKey: Field.previousPurchases)
        if let insight = appointment.customerInsight {
            doc.setString(insight, forKey: Field.customerInsight)
        }
        if let summary = appointment.tasteSummary {
            doc.setString(summary, forKey: Field.tasteSummary)
        }
        try collection.save(document: doc)
    }

    // MARK: - Delete

    func deleteAll() throws {
        let query = QueryBuilder
            .select(SelectResult.expression(Meta.id))
            .from(DataSource.collection(collection))
        for row in try query.execute() {
            guard let docID = row.string(forKey: "id"),
                  let doc   = try collection.document(id: docID) else { continue }
            try collection.delete(document: doc)
        }
    }

    // MARK: - Date reassignment

    // Redistributes appointment dates on each app launch, but only when today's date
    // is later than the earliest date already stored (or when no dates are set yet).
    //
    // Assignment rules:
    //   • 3–5 randomly chosen appointments  → today
    //   • all others                         → spread evenly across the next 7 days
    //                                          (tomorrow … today+7), round-robin
    func reassignDatesIfNeeded() throws {
        let cal   = Calendar.current
        let today = cal.startOfDay(for: Date())
        let fmt   = DateFormatter()
        fmt.dateFormat = "yyyy-MM-dd"
        let todayStr = fmt.string(from: today)

        // Collect every document id + its current date string in one pass.
        let allQuery = QueryBuilder
            .select(SelectResult.expression(Meta.id),
                    SelectResult.property(Field.date))
            .from(DataSource.collection(collection))

        var docDates: [(id: String, date: String)] = []
        for row in try allQuery.execute() {
            guard let docID = row.string(forKey: "id") else { continue }
            docDates.append((id: docID, date: row.string(forKey: Field.date) ?? ""))
        }
        guard !docDates.isEmpty else { return }

        // Minimum of all non-empty date strings ("" sorts below any yyyy-MM-dd value).
        let minDate = docDates.map { $0.date }.filter { !$0.isEmpty }.min() ?? ""

        // Only proceed when today is strictly later than the earliest stored date
        // (or no dates have been assigned yet, in which case minDate == "").
        guard todayStr > minDate else { return }

        // Build the target date pool: today + next 7 days.
        let nextSevenDays: [String] = (1...7).compactMap { offset in
            cal.date(byAdding: .day, value: offset, to: today).map { fmt.string(from: $0) }
        }

        // Shuffle all ids and split: first batch → today, rest → spread across next 7 days.
        var shuffled     = docDates.map { $0.id }.shuffled()
        let todayCount   = Int.random(in: 3...5)
        let todayIDs     = Array(shuffled.prefix(todayCount))
        let remainingIDs = Array(shuffled.dropFirst(todayCount))

        var assignments: [String: String] = [:]
        for id in todayIDs { assignments[id] = todayStr }
        for (i, id) in remainingIDs.enumerated() {
            assignments[id] = nextSevenDays[i % nextSevenDays.count]
        }

        // Write updated date field back to each document.
        for (docID, dateStr) in assignments {
            guard let doc = try collection.document(id: docID) else { continue }
            let mutable = doc.toMutable()
            mutable.setString(dateStr, forKey: Field.date)
            try collection.save(document: mutable)
        }
    }

    // MARK: - Read

    // Returns all appointments whose date matches `date` (formatted as yyyy-MM-dd),
    // sorted by time ascending. Unscheduled appointments (empty date) are excluded
    // after the initial reassignment has run.
    func fetchAppointments(for date: Date) throws -> [Appointment] {
        let query = QueryBuilder
            .select(SelectResult.expression(Meta.id))
            .from(DataSource.collection(collection))

        let fmt = DateFormatter()
        fmt.dateFormat = "yyyy-MM-dd"
        let dateStr = fmt.string(from: date)

        var result: [Appointment] = []
        for row in try query.execute() {
            guard let docID = row.string(forKey: "id"),
                  let doc   = try collection.document(id: docID) else { continue }

            let apptDate = doc.string(forKey: Field.date) ?? ""
            guard apptDate == dateStr else { continue }

            let purchases = (doc.array(forKey: Field.previousPurchases)?.toArray() as? [String]) ?? []
            result.append(Appointment(
                id:                docID,
                name:              doc.string(forKey: Field.name)   ?? "",
                gender:            doc.string(forKey: Field.gender) ?? "",
                size:              doc.string(forKey: Field.size)   ?? "",
                previousPurchases: purchases,
                customerInsight:   doc.string(forKey: Field.customerInsight),
                tasteSummary:      doc.string(forKey: Field.tasteSummary),
                time:              doc.string(forKey: Field.time)   ?? "",
                date:              apptDate
            ))
        }
        return result.sorted { $0.time < $1.time }
    }
}
