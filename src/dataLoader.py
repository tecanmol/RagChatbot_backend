from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def load_All_doc(data_dir: str) -> List[Any]:
    # .resolve() turns the path into an absolute path.
    data_path = Path(data_dir).resolve()
    print(f"[Debug] Data path: {data_path}")
    documents = []
    
    #PDF files
    
    pdf_files = list(data_path.glob('**/*.pdf'))
    print(f"\n[Debug] Found {len(pdf_files)} PDF Files: {[str(f) for f in pdf_files]}")
    
    for pdf in pdf_files:
        print(f'\n[Debug] loading: {pdf}')
        try:
            loader = PyPDFLoader(str(pdf))
            loaded = loader.load()
            print(f'\n[Debug] Loaded {len(loaded)} PDF from {pdf_files}')
            documents.extend(loaded)
        except Exception as e:
            print(f'\n[Error] Failed to load PDF {pdf}: {e}')
    
    
    #txt file
    txt_files = list(data_path.glob('**/*.txt'))
    print(f'\n[Debug] Found {len(txt_files)} txt Files : {[str(f) for f in txt_files]}')
    
    for txt in txt_files:
        print(f'\n[Debug] Loading {txt}')
        try:
            loader = TextLoader(txt)
            loaded = loader.load()
            print(f'\n[Debug] Loaded {len(loaded)} Txt from {txt_files}')
            documents.extend(loaded)
        except Exception as e:
            print(f'\n[Error] Failed to load Txt {txt}: {e}')
    
    # print(documents)
    return documents

if __name__ == "__main__":
    load_All_doc("../data")