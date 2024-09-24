from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage
)
import chromadb, os
from utils import ensure_directory_exists, load_data_from_files
from llama_index.vector_stores.chroma import ChromaVectorStore
from constants import CHROMA_DB_PATH, OPENAI_EMBEDDING_MODEL

class Index:
    def __init__(self, collection_name: str) -> None:
        """
        Initialize the Index class.

        Args:
            collection_name (str): The name of the collection to use for the ChromaDB.
        """
        
        collection_path = os.path.join(CHROMA_DB_PATH, collection_name)

        # Ensure the directory exists
        ensure_directory_exists(collection_path)

        db = chromadb.PersistentClient(path=collection_path)

        collection = db.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)

        self.collection_name = collection_name
        self.collection_path = collection_path
        self.vector_store = vector_store
        self.storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = None

    def load_or_create_index(self) -> VectorStoreIndex:
        """
        Load an existing index from storage or create a new one if it doesn't exist.

        Returns:
            VectorStoreIndex: The loaded or newly created index.
        """
        try:
            # Attempt to load the index from storage
            # self.index = load_index_from_storage(storage_context=self.storage_context)
            self.index = VectorStoreIndex.from_vector_store(self.vector_store)
            print("Index found")
        except Exception:
            # Index does not exist, create a new one
            print("No index found, creating...")
            self.index = self._create_index()
            self.index.storage_context.persist(persist_dir=self.collection_path)

        return self.index

    def _create_index(self) -> VectorStoreIndex:
        """
        Create a new index from documents in the specified collection.

        Returns:
            VectorStoreIndex: The newly created index.
        """
        # Load documents from files in the specified directory
        file_docs = load_data_from_files(directory=self.collection_name)

        # Create a new index from the documents
        index = VectorStoreIndex.from_documents(documents=file_docs, storage_context=self.storage_context)
        
        return index

    def get_index(self) -> VectorStoreIndex:
        """
        Get the currently loaded index.

        Returns:
            VectorStoreIndex: The loaded index.
        """
        if self.index is None:
            raise ValueError("Index is not loaded. Call 'load_or_create_index' first.")
        return self.index
    
    def get_storage_context(self) -> StorageContext:
        """
        Get the currently loaded storage_context.

        Returns:
            StorageContext: The loaded StorageContext.
        """
        return self.storage_context
    
    def get_collection_path(self) -> str:
        """
        Get the currently working collection path.

        Returns:
            StorageContext: The loaded StorageContext.
        """
        return self.collection_path
