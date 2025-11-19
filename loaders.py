
from langchain_community.document_loaders import TextLoader 
import os 

class DocumentLoader:
    def __init__(self, doc_path: str):
        # Will get the path from the main pipeline
        self.doc_path = doc_path
        
    def load_content(self):
        # Check if the path exists first 
        if not os.path.exists(self.doc_path):
            print(f"Error: File not found at path: {self.doc_path}, kindly Check the path")
            return []
            
        loader = TextLoader(self.doc_path)
        try:

            # Load the document content
            documents = loader.load()
            print(f"Successfully loaded content from file: {self.doc_path}, resulting in {len(documents)} document(s).")
            return documents
        except Exception as e:
            print(f"Error loading file content: {str(e)}")
            return []

