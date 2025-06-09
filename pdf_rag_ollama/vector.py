import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from langchain_core.documents import Document

def get_vector_collection() -> chromadb.Collection:
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )

    chroma_client = chromadb.PersistentClient(path="./rag-chroma")

    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )

def add_to_vector_collection(all_splits: list[Document], file_name: str):
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    collection = get_vector_collection()
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

def query_collection(prompt: str, n_results: int = 10):
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results

def get_file_names_from_ids():
    collection = get_vector_collection()

    # Fetch all entries (default includes IDs)
    results = collection.get()
    # Extract IDs from result
    all_ids = results["ids"]

    # Extract and deduplicate "{file_name}_{idx}"
    file_names = set(id.rsplit("_", 1)[0] for id in all_ids)

    # Return the unique file_names
    return sorted(file_names)
