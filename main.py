from src.dataLoader import load_All_doc
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch
import os
print("Current Working Directory:", os.getcwd())

def main():
    docs = load_All_doc("data")
    store = FaissVectorStore("faiss_store")
    store.build_from_doc(docs)
    store.load()
    
    rag_search = RAGSearch()
    query = input("Ask about Anmol: ")
    info = rag_search.search_and_summarize(query, top_k=3)
    print("info:", info)

    
    
    
if __name__ == "__main__":
    main()