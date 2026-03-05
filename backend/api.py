from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.search import RAGSearch
import asyncio

app = FastAPI()
    
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGSearch()

class QueryRequest(BaseModel):
    query: str


async def stream_response(text: str):
    words = text.split()

    for word in words:
        yield word + " "
        await asyncio.sleep(0.02)  # typing speed


@app.post("/ask")
async def ask_question(req: QueryRequest):

    answer = rag.search_and_summarize(req.query, top_k=3)

    return StreamingResponse(
        stream_response(answer),
        media_type="text/plain"
    )