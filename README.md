# Couchbase Mobile Demo for iOS

An iOS demo app that combines on-device vector search and a conversational AI assistant
to help store associates serve customers. Everything runs locally — no network, no server.

**Two on-device AI models:**
- **SigLIP ViT-B/16** (Apache 2.0, Google) via **Core ML** — maps images and text into the
  same 768-dim vector space for cross-modal product search.
- **Llama 3.2 3B Instruct** (Q4_K_M GGUF) via **llama.cpp** — powers the conversational
  assistant and generates customer taste summaries.

**Storage and search:**
- **Couchbase Lite Enterprise** with the Vector Search extension — stores products with their
  embeddings and serves ANN queries via a cosine vector index (`approx_vector_distance()`).

---

## App overview

The app has three tabs, all accessible from the bottom navigation bar.

### Catalog

A browsable product grid that supports text and image search powered by SigLIP embeddings.

- **Text search:** type a query (e.g. "blue silk blouse", "wool coat") and the app embeds it
  with the SigLIP text encoder, queries the Couchbase Lite vector index, and returns the top 5
  most visually similar products ranked by cosine similarity.
- **Image search:** tap the camera icon to pick a photo from the photo library or take one
  with the camera. The image is embedded with the SigLIP vision encoder and matched against
  the catalog in the same way as a text query.
- The grid paginates at 50 items per page and shows a total product count.

### Schedule

A daily appointment calendar showing customers booked for the selected date.

- Each appointment card shows the customer's name, appointment time, size, and previous
  purchases.
- Tap a customer to expand their profile, including an AI-generated taste summary.
- The **Generate Recommendation** button runs the LLM to produce a personalised product
  suggestion: it searches the catalog for items that match the customer profile and injects
  both the customer insight and the matched products into the Llama prompt, then streams
  the response.
- Appointment dates are redistributed on every app launch (3–5 customers assigned to today,
  the rest spread across the next 7 days) so the Schedule tab always has live content.

### Assistant

A multi-turn chat interface backed by Llama 3.2 3B running fully on-device.

- Each message triggers a RAG (retrieval-augmented generation) pipeline: the query is
  embedded via SigLIP to retrieve relevant products from the catalog, and today's
  appointments are injected as context, giving the LLM grounding in both inventory and
  customers.
- The conversation history is maintained across turns within a session.
- The LLM model state is shown at the top of the tab (loading / ready / failed).

---

## Requirements

- **Xcode** with an iOS 17+ simulator or device.
- **Couchbase Lite Enterprise** — Vector Search is Enterprise-only. The packages resolve via
  SPM automatically, but a valid EE licence is required for non-trivial or production use.
- **Python 3.11–3.13** for the one-time SigLIP model conversion (**not 3.14** — PyTorch
  ≤2.7.0 ships no wheels for 3.14).
- **`huggingface-cli`** (or `curl`) for downloading the Llama model.

---

## Setting up for development

### 1. Clone the repo

```bash
git clone <repo-url>
cd MobileVectorDemo
```

### 2. Convert the SigLIP model (once)

This downloads `google/siglip-base-patch16-224` (~350 MB) and writes two `.mlpackage` files
plus a tokenizer vocabulary into `Models/` (~386 MB total). These are gitignored but required
by Xcode at build time.

**Use Python 3.11, 3.12, or 3.13 — not 3.14.** If your system `python3` is 3.14
(`python3 --version`), install 3.13 explicitly: `brew install python@3.13`.

```bash
# Create a venv with Python 3.13 explicitly:
"$(brew --prefix python@3.13)/bin/python3.13" -m venv siglip-convert
siglip-convert/bin/pip install --upgrade pip
siglip-convert/bin/pip install -r Scripts/requirements.txt
siglip-convert/bin/python3 Scripts/convert_siglip.py --output Models/SigLIP
```

A successful run ends with:
```
Generated vocab.json: 32000 pieces  (unk=2, eos=1, pad=0)
Vision  output shape: (1, 768)  |  L2 norm: 0.9996 (should be ≈1.0)
Text    output shape: (1, 768)  |  L2 norm: 0.9994 (should be ≈1.0)
Verification passed ✓
```

The `overflow encountered in cast` warnings during the text encoder are benign.

<details><summary>Troubleshooting conversion errors</summary>

- `No matching distribution found for torch` → your venv is Python 3.14. Delete and recreate
  with 3.13.
- `Failed building wheel for tokenizers` → you loosened the `transformers` pin to ≤4.44.x,
  which requires a `tokenizers` build without a 3.13 wheel. Keep the pinned versions in
  `Scripts/requirements.txt`.
- `converting 'int' op … only 0-dimensional arrays can be converted to Python scalars` →
  `convert_siglip.py` already patches this (replaces `nn.MultiheadAttention`); don't revert
  the script.
</details>

### 3. Download the Llama model (once)

Downloads Llama 3.2 3B Instruct Q4_K_M (~2.0 GB GGUF) from Hugging Face into `Models/Llama/`.
This is gitignored and must be downloaded locally.

```bash
bash Scripts/download_llama.sh
```

The script uses `huggingface-cli` if available, otherwise falls back to `curl`. It is
idempotent — safe to re-run if interrupted.

### 4. Add the product catalog

The repo includes a `Catalog_example/` folder with two sample products to show the expected
data format. To run the app with real data:

1. **Rename or copy** `Catalog_example/` to `Catalog/` (the app bundle looks for this name).
2. **Replace** `Catalog/catalog.json` with your product data following this schema:

```json
[
  {
    "image": "filename.jpg",
    "sku": "UNIQUE-SKU-001",
    "name": "Product Name",
    "description": "A short product description.",
    "category": { "department": "Women", "type": "Tops", "subtype": "Blouses" },
    "color": "blue",
    "material": "100% silk",
    "sizes": ["XS", "S", "M", "L"],
    "price": { "amount": 120.00, "currency": "EUR" },
    "tags": ["silk", "blouse", "formal", "blue"],
    "stock": 15
  }
]
```

3. **Place the corresponding product images** in `Catalog/Images/`. Each `"image"` value in
   `catalog.json` must match a filename in that folder (JPG or PNG).

The catalog is seeded into Couchbase Lite on first launch. If you later change the catalog
data, use the **Re-seed** button in the app to reset and re-import everything.

### 5. Add customer appointments data

The repo includes a `Customers/customers.json` file with sample customer appointments used to
populate the Schedule tab. Replace its contents with your own data following this schema:

```json
[
  {
    "name": "Customer Name",
    "gender": "female",
    "size": "M",
    "time": "10:30",
    "date": "",
    "previousPurchases": ["item one", "item two"],
    "customerInsight": "Optional free-text profile note used as context for the LLM."
  }
]
```

Leave `"date"` as an empty string — the app assigns dates dynamically on each launch.

### 6. Build and run

Open `MobileDemo.xcodeproj` in Xcode, select an iOS 17+ simulator, and run.

- Xcode compiles the `.mlpackage` files to `.mlmodelc` at build time (they are in the target's
  Sources build phase). `vocab.json` is a bundled resource. Both load from `Bundle.main` —
  the same path works on simulator and device.
- On first launch the app seeds the inventory (embeds all product images — this takes a moment
  depending on catalog size) then shows **"Ready — vector index active"**.
- **Physical iPhone:** set a Development Team under *Signing & Capabilities* (a free Apple ID
  works). First install is larger (~389 MB for the SigLIP models + ~2 GB for Llama) — plan
  for the download/build time accordingly.
- The Llama model loads when the Assistant tab is first opened. Model loading progress is
  shown at the top of the tab.

---

## Project layout

```
MobileDemo.xcodeproj              Open this in Xcode
MobileDemo/
  MobileDemoApp.swift             Entry point; enables Vector Search extension at startup
  ContentView.swift               Thin wrapper that renders StoreAssistantView
  StoreAssistantView.swift        All three tabs (Catalog, Schedule, Assistant) and UI logic
  InventoryStore.swift            Couchbase Lite: inventory collection, vector index, search
  InventorySeeder.swift           Seeds catalog.json + product images into Couchbase Lite
  AppointmentStore.swift          Couchbase Lite: appointments collection, date reassignment
  AppointmentSeeder.swift         Seeds customers.json into AppointmentStore
  SigLIPEmbedder.swift            Core ML inference + pure-Swift SentencePiece tokenizer
  LlamaEngine.swift               Llama 3.2 lifecycle management + multi-turn chat
Catalog_example/                  Sample catalog data showing the expected format
  catalog.json
  Images/
Customers/
  customers.json                  Customer profiles for the Schedule tab
Models/                           Gitignored — generated locally (see setup steps above)
  SigLIP/
    SigLIPVision.mlpackage        Compiled to .mlmodelc at build time
    SigLIPText.mlpackage          Compiled to .mlmodelc at build time
    tokenizer/vocab.json          Bundled resource read by SigLIPTokenizer
  Llama/
    llama-3.2-3b-instruct-q4_k_m.gguf
Scripts/
  convert_siglip.py               PyTorch → Core ML conversion
  requirements.txt                Pinned Python dependencies for conversion
  download_llama.sh               Downloads the Llama GGUF from Hugging Face
```

---

## Dependencies (Swift Package Manager)

Configured in the project; Xcode resolves them on first open:

| Package | Version | Product |
|---|---|---|
| Couchbase Lite Swift (Enterprise) | 4.0.x (from 4.0.3) | `CouchbaseLiteSwift` |
| Couchbase Lite Vector Search | 2.0.x | `CouchbaseLiteVectorSearch` |
| llama.cpp Swift | latest | `llama` |

---

## License

Couchbase Lite Enterprise requires a Couchbase licence. 
SigLIP ViT-B/16 is Apache 2.0 (Google). 
Llama 3.2 is licensed under the Llama 3.2 Community License Agreement, Copyright © Meta Platforms, Inc. All Rights Reserved. 