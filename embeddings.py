
from langchain_community.embeddings import HuggingFaceEmbeddings
class HuggingFaceEmbeddingsModel:
    """local running sentence-transformers embedding model."""
    def __init__(self, model_name = "sentence-transformers/all-miniLM-L6-v2"):
        # Initializes the Embedding model of HuggingFace
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs = {'device': 'cpu'}

        )
        # status for whether model loaded
        print(f"HuggingFace Embeddings model loaded: {model_name}")
    
    def get_embeddings(self):
        return self.embeddings
