from langchain_classic.chains import RetrievalQA # CORRECT PATH for RetrievalQA
from langchain_community.llms import Ollama # Updated path for Ollama
from langchain_community.vectorstores import Chroma # Updated path for Chroma
from langchain_classic.prompts import PromptTemplate
from typing import Dict, Any

class Retriever:
    """Configures the vector store as a retriever component."""
    def __init__(self, vectorstore: Chroma, k: int = 3):
        # Using MMR (Maximum Marginal Relevance) help in finding relevant documents retrieval
        self.retriever = vectorstore.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": k}
        )
        print(f"Retriever initialized with k={k}.")

class ResponseGenerator:
    """Manages the Ollama LLM and creates the final RetrievalQA chain which takes to a Recursive Retriever chain."""
    def __init__(self, model_name: str = "mistral"):
        # Initialize the local LLM instance
        self.llm = Ollama(model=model_name) 
        print(f"Ollama LLM initialized with model: {model_name}")
        self.qa_chain = None
    
    def create_qa_chain(self, retriever):
        """Creates and returns the RetrievalQA chain."""
        # Define a custom prompt template to guide the LLM's response
        custom_prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Only use the source documents provided. Answer concisely.

        {context}

        Question: {question}
        Helpful Answer:"""

        QA_CHAIN_PROMPT = PromptTemplate.from_template(custom_prompt_template)

        # Tie the LLM and the retriever together
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff", # Stuffing all context into one prompt
            retriever=retriever,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}, # Apply the custom prompt
            return_source_documents=True 
        )
        print("RetrievalQA Chain created.")
        return self.qa_chain