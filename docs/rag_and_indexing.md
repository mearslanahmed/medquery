# RAG Pipeline & Vector Indexing

## Overview

MedQuery searches through 23,167 passages extracted from 6 standard medical textbooks. Here is how we parsed the PDFs, chunked the text, generated embeddings, and stored them in Pinecone without crashing our local machine.

---

## 1. What Textbooks We Used

We indexed 6 reference books (~685 MB total, ~15,000 pages):
- **Clinical Practice & Treatment:** *CURRENT Medical Diagnosis & Treatment (2025)* (latest clinical diagnostic criteria and treatment guidelines)
- **Internal Medicine:** *Harrison’s Principles of Internal Medicine (21st Edition)* (clinical pathology, systemic diseases, differential diagnosis)
- **Medical Physiology:** *Guyton and Hall Textbook of Medical Physiology (14th Edition)* (cellular and organ-system physiological mechanisms)
- **Pharmacology:** *Katzung’s Basic and Clinical Pharmacology (16th Edition)* (pharmacokinetics, pharmacodynamics, drug contraindications)
- **Pathology:** *Robbins & Cotran Pathologic Basis of Disease* (cellular mechanisms of disease and structural tissue pathology)
- **General Medicine Reference:** *The Gale Encyclopedia of Medicine* (patient-facing medical encyclopedic reference)

---

## 2. Text Chunking (`src/helper.py`)

We used LangChain's `RecursiveCharacterTextSplitter` with these settings:
- **Chunk Size:** 1,000 characters
- **Chunk Overlap:** 100 characters

### Why these numbers?
- **Too small (<400 chars):** Medical context gets cut off. For example, a drug's name might be separated from its contraindications or warnings, making the retrieved chunk useless.
- **Too big (>2,000 chars):** The embedding vector becomes diluted with too many unrelated topics, which hurts similarity search accuracy.
- **1,000 chars (~150–200 words):** This is the sweet spot. It captures a complete medical thought (e.g., drug dosage + warning signs) while staying focused enough for cosine similarity.
- **100 char overlap:** Prevents sentences or drug numbers from being chopped in half at chunk borders.

---

## 3. The Embedding Model (`all-MiniLM-L6-v2`)

We use `sentence-transformers/all-MiniLM-L6-v2`.

### Why this model over OpenAI or larger transformers?
1. **Lightweight:** The model is only **~80 MB** on disk. When loaded into RAM, it uses around 120 MB. This is critical because Render's free tier only gives us 512 MB of RAM. A larger 1 GB embedding model would immediately trigger an out-of-memory crash.
2. **Speed:** Runs on CPU in ~20 milliseconds per query.
3. **Dimensions:** Outputs 384-dimensional dense vectors. Compared to 1,536-dimensional OpenAI embeddings, 384 dimensions take 4x less memory and search faster in Pinecone.

In `src/helper.py`, we enforce CPU usage and pin PyTorch to a single thread:
```python
import torch
torch.set_num_threads(1)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
```

---

## 4. How Ingestion Runs Without Crashing (`store_index.py`)

If you try to load 6 medical textbooks (15,000 pages) all at once in Python using `PyPDFLoader`, your machine will quickly run out of RAM and freeze.

To solve this, `store_index.py` processes each book one at a time:

```python
for pdf_file in pdf_files:
    print(f"Loading {pdf_file}...")
    docs = load_single_pdf(pdf_path)
    
    # Strip heavy metadata, keep only filename and page number
    minimal_docs = filter_to_minimal_docs(docs)
    chunks = text_split(minimal_docs)
    
    # Upsert in batches of 100 to Pinecone
    for i in tqdm(range(0, len(chunks), 100)):
        batch = chunks[i:i + 100]
        PineconeVectorStore.from_documents(batch, embeddings, index_name="medquery")
    
    # Clean up memory before starting the next book
    del docs, minimal_docs, chunks
    gc.collect()
```

By upserting in 100-chunk batches and running `gc.collect()` between books, memory stays flat at ~300 MB throughout the entire ingestion of all 23,167 vectors.
