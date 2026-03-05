from sentence_transformers import SentenceTransformer
from typing import List, Any
import numpy as np
from src.dataLoader import load_All_doc
from langchain_text_splitters import RecursiveCharacterTextSplitter

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size=400, chunk_overlap=80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")
        
    
    def chunk_doc(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function = len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks
    
    def embed_chunks(self, chunks:List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings
    
# test 
if __name__ == "__main__":
    
    docs = load_All_doc("../data")
    emb_pipe = EmbeddingPipeline()
    chunks = emb_pipe.chunk_doc(docs)
    embeddings = emb_pipe.embed_chunks(chunks)