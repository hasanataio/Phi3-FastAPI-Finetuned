from fastapi import FastAPI, Depends
import asyncio
import json, re
from typing import List
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch, os
from pydantic import BaseModel

from ast import literal_eval

from angle_emb import AnglE
from uvicorn.config import Config
import time



app = FastAPI()
semaphore = asyncio.Semaphore(10)  # Adjust this number based on your GPU's capacity

class ReviewRequest(BaseModel):
    review: str

class EmbeddingsRequest(BaseModel):
    aspects:  List[str] 


embeddings_model = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').to("cuda")

# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "reviewAnalyzerWeightsWithSentiments_phi_3.5_mini_4bit", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
stop_token = "<|end|>"
stop_token_id = tokenizer.encode(stop_token)[0]

async def get_result(review):
    # return {"review":review,"result":{
    #     "overall_sentiment":"Positive",
    #     "aspects":{
    #         "Staff Service and Behavior": [],
    #         "Ambience": [
    #             {
    #                 "Extracted Segment": "Dummy segment.",
    #                 "sentiment": "Positive"
    #             }
    #         ]}
    #     }
    #         }
    # review = review.replace('"', '').replace("'", '`')
    # review = review.replace("'", '"')
    try:
        review = review.replace("'", '')

        messages = [
        {"from": "human", "value": review},


        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True, # Must add for generation
            return_tensors = "pt",
        ).to("cuda")

        with torch.no_grad():
            model_response = model.generate(input_ids = inputs,eos_token_id=stop_token_id,max_new_tokens=1024)
        # print(len(response))
        # response.to_cpu()
        response=tokenizer.batch_decode(model_response, skip_special_tokens=True)[0]
        # print(response)
        torch.cuda.empty_cache()    
        # Generate the output
        # with torch.no_grad():
        #     outputs = model.generate(input_ids=inputs, max_new_tokens=512, use_cache=True)

        # response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        # print(response)

        start = response.find('{')
        end = response.rfind('}') + 1
        json_string = response[start:end]
        start=time.time()
        py_dict = literal_eval(json_string)

        # json_string = json_string.replace("'", '"').replace('`', "'")


        # json_data = json.loads(json_string)
        return {"review":review,"result":py_dict}
    except Exception as e:
        return {"error":f"Error occured: {e}"}

@app.post('/extract_snippets')
async def extract_snippets(request: ReviewRequest):
    async with semaphore:  # Control concurrency
        results = await get_result(request.review)
    return results


@app.post('/extract_embeddings')
async def extract_snippets(request: EmbeddingsRequest):
    result=embeddings_model.encode(request.aspects)
    return {"embeddings":result.tolist()}