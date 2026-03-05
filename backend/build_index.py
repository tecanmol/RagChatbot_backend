import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataLoader import load_All_doc
from src.vectorstore import FaissVectorStore

docs = load_All_doc("data")

store = FaissVectorStore("faiss_store")
store.build_from_doc(docs)

print("✅ FAISS index built successfully")