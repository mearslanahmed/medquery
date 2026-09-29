import langchain_community
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document


# Extract Data from a single PDF File safely
def load_single_pdf(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    return loader.load()


# Extract Data from PDF Directory
def load_pdf_file(data):
    loader = DirectoryLoader(data, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of document objects
    preserving 'source' and 'page' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        page = doc.metadata.get("page")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src, "page": page},
            )
        )
    return minimal_docs


# Split the doc into smaller chunks with optimal medical context size
def text_split(docs, chunk_size: int = 1000, chunk_overlap: int = 100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    texts_chunk = text_splitter.split_documents(docs)
    return texts_chunk


# Download the HuggingFace embeddings model


def download_hugging_face_embeddings():
    """
    Download and return the HuggingFace embeddings model with minimal memory footprint.
    """
    import torch
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

