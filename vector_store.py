
from langchain_community.vectorstores import Chroma 

class VectorStore:
    """Deals with the creation and storage of the Chroma vector database."""
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def create_store(self, chunks):
        """Creates the ChromaDB store from chunks in memory and returns it."""
        
        print("Creating new ChromaDB store in memory...")
        
        # Create the vector store from the documents (chunks)
        # The persist_directory parameter is now explicitly omitted, 
        # signaling a pure in-memory operation.
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
        )

        print("ChromaDB vector store successfully created in memory.")
        return vectordb