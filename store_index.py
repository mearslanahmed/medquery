import os
import gc
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from tqdm import tqdm

from src.helper import (
    load_single_pdf,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings,
)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY or ""

print("=" * 60)
print("  MedQuery — Clinical Literature Ingestion Pipeline")
print("=" * 60)

print("\n[1/3] Loading embeddings model (all-MiniLM-L6-v2)...")
embeddings = download_hugging_face_embeddings()

print("\n[2/3] Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medquery"

if not pc.has_index(index_name):
    print(f"Creating Pinecone index: '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)
stats = index.describe_index_stats()
print(f"✓ Connected to '{index_name}'. Current vector count: {stats.total_vector_count}")

# Initialize vector store connection
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

data_dir = "data"
pdf_files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")])
print(f"\n[3/3] Found {len(pdf_files)} PDF books in '{data_dir}/':")
for i, f in enumerate(pdf_files, 1):
    size_mb = os.path.getsize(os.path.join(data_dir, f)) / (1024 * 1024)
    print(f"  {i}. {f} ({size_mb:.1f} MB)")

BATCH_SIZE = 100

print("\nStarting memory-safe, book-by-book ingestion...")
for i, filename in enumerate(pdf_files, 1):
    file_path = os.path.join(data_dir, filename)
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"\n{'='*50}")
    print(f"[{i}/{len(pdf_files)}] Processing: {filename} ({size_mb:.1f} MB)")
    print(f"{'='*50}")

    try:
        # Step 1: Load single book safely
        print("  → Parsing PDF pages...")
        raw_docs = load_single_pdf(file_path)
        print(f"  ✓ Parsed {len(raw_docs)} pages.")

        # Step 2: Preserve source filename and exact page number
        filtered_docs = filter_to_minimal_docs(raw_docs)

        # Step 3: Optimal medical chunking (1000 chars, 100 overlap)
        chunks = text_split(filtered_docs, chunk_size=1000, chunk_overlap=100)
        print(f"  ✓ Generated {len(chunks)} chunks.")

        # Step 4: Batch embed and upsert with live progress bar
        print(f"  → Embedding & upserting in batches of {BATCH_SIZE}...")
        short_title = filename[:22] + "..." if len(filename) > 25 else filename
        for b_idx in tqdm(range(0, len(chunks), BATCH_SIZE), desc=f"Upserting {short_title}"):
            batch = chunks[b_idx : b_idx + BATCH_SIZE]
            docsearch.add_documents(batch)

        print(f"  ✓ Successfully completed: {filename} ({len(chunks)} vectors indexed).")

        # Cleanup memory before next book
        del raw_docs, filtered_docs, chunks
        gc.collect()

    except Exception as e:
        print(f"  ✗ Error indexing {filename}: {e}")
        gc.collect()

final_stats = index.describe_index_stats()
print("\n" + "=" * 60)
print(f"Ingestion complete!")
print(f"Total vectors in Pinecone '{index_name}': {final_stats.total_vector_count}")
print("=" * 60)
