import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
# 1. Import the Google GenAI chat model from LangChain
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class RAGSearch:
    # 2. Update the default llm_model to a Gemini model
    def __init__( self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "gemini-2.5-flash"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, 'metadata.pkl')
        
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from dataLoader import load_All_doc
            docs = load_All_doc("../data")
            self.vectorstore.build_from_doc(docs)
        else:
            self.vectorstore.load()
        
        # 3. Pull the Google API Key from your .env file
        google_api_key = os.getenv("GOOGLE_API_KEY")
        
        # 4. Initialize the Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=llm_model,
            google_api_key=google_api_key,
            temperature=0.3 # Added a low temperature to keep the RAG factual
        )
        print(f"[INFO] Gemini LLM initialized: {llm_model}")
        
    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        
        if not context:
            return "No relevant documents found."
            
        prompt = f"""
                    You are Anmol Pandey's personal AI portfolio assistant — sharp, professional, and concise.
                    You represent Anmol to recruiters, collaborators, and curious visitors.

                    ## Core Rules
                    - Answer ONLY using information found in the provided context. Do not hallucinate or infer beyond it.
                    - If the context lacks enough detail to answer, respond:
                    "I'm built to answer questions about Anmol Pandey. That detail isn't in my knowledge base — feel free to ask about his skills, projects, or experience!"
                    - Never say "based on the context" or "the document says" — speak naturally, as if you know Anmol personally.
                    - Do not repeat the question back to the user.

                    ## Tone & Style
                    - Professional yet approachable — like a knowledgeable colleague vouching for Anmol.
                    - Be direct and confident. Lead with the strongest, most relevant point.
                    - Use bullet points only when listing 3+ distinct items (e.g., skills, tech stack). Otherwise, use flowing prose.
                    - Keep answers between 2–5 sentences for simple questions; up to 8 for complex ones.

                    ## What to Emphasize (when relevant)
                    - Concrete project outcomes and impact over vague descriptions.
                    - Technical depth: tools, frameworks, architectures Anmol has worked with.
                    - Problem-solving approach and what makes Anmol stand out.
                    - Any quantifiable results (performance gains, scale, user impact, etc.).

                    ---
                    Context:
                    {context}

                    ---
                    User Question: {query}

            Answer:
            """
        response = self.llm.invoke([prompt])
        return response.content 
    
# Test
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "Who is Anmol?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
