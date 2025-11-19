
from langchain_text_splitters import CharacterTextSplitter

class DocumentChunker:
    
    # Splits the document into small manageable chunks (For embedding)
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 50):
        self.splitter = CharacterTextSplitter(
            separator="\n\n", 
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    # Actual chunks generated
    def create_chunks(self, documents):
        chunks = self.splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks from {len(documents)} documents.")
        return chunks
