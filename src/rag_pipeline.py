import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Define directories
PERSIST_DIRECTORY = "./chroma_db"
DATA_DIRECTORY = "./data"


def initialize_vector_store(data_dir: str = DATA_DIRECTORY):
    """Loads documents from the data directory, splits them into chunks,

    embeds them, and persists them into a Chroma vector store.
    """
    print(f"Loading documents from {data_dir}...")

    # Ensure data directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(
            f"Created missing '{data_dir}' directory. Please add text files for your RAG system."
        )
        return None

    # Load text documents using LangChain
    loader = DirectoryLoader(
        data_dir, glob="./*.txt", loader_cls=TextLoader, show_progress=True
    )
    documents = loader.load()

    if not documents:
        print("No documents found to embed. Add some .txt files to the data/ folder.")
        return None

    # Split documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(docs)} chunks.")

    # Initialize open-source embeddings (HuggingFace)
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create and persist Chroma vector database
    print(f"Storing embeddings in ChromaDB at {PERSIST_DIRECTORY}...")
    vector_store = Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=PERSIST_DIRECTORY
    )

    print("Vector store initialized successfully!")
    return vector_store


def get_retriever(k_results: int = 3):
    """Loads the persisted Chroma vector store and returns a retriever instance."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if not os.path.exists(PERSIST_DIRECTORY):
        print(
            "Vector database not found. Please run initialize_vector_store() first."
        )
        return None

    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings
    )
    return vector_store.as_retriever(search_kwargs={"k": k_results})


if __name__ == "__main__":
    # Test initialization when run directly
    initialize_vector_store()
