from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
from embeddings_model import get_embeddings
from sentiment_model import run_sentiment_analysis
app = FastAPI()

class EmbeddingsRequest(BaseModel):
    snippets:  List[str] 

class SentimentsRequest(BaseModel):
    snippet:str
    aspect:str





@app.post('/extract_embeddings')
async def extract_snippets(request: EmbeddingsRequest):
    result=await get_embeddings(request.snippets)
    return {"embeddings":result.tolist()}


@app.post('/extract_sentiments')
async def extract_sentiments(request: SentimentsRequest):
    result=await run_sentiment_analysis(request.snippet,request.aspect)
    return {"sentiment":result}