from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel
from typing import List
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, pipeline
 
import torch
app = FastAPI()
 
temperature = 1.6
# Label smoothing parameter
label_smoothing_factor = 0.5
 
from transformers import pipeline
 
pipeline_overall_sentiment = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
 
tokenizer = AutoTokenizer.from_pretrained("laiyer/deberta-v3-base-zeroshot-v1-onnx")
model = ORTModelForSequenceClassification.from_pretrained("laiyer/deberta-v3-base-zeroshot-v1-onnx")
classifier = pipeline(
    task="zero-shot-classification",
    model=model,
    tokenizer=tokenizer,
)
 
async def apply_temperature_scaling(logits, temperature):
    logits = torch.tensor(logits) / temperature
    return torch.nn.functional.softmax(logits, dim=-1).numpy()
 
async def apply_label_smoothing(probs, label_smoothing_factor):
    num_classes = len(probs)
    smooth_label = (1.0 - label_smoothing_factor) + (label_smoothing_factor / num_classes)
    smooth_probs = probs * smooth_label + (label_smoothing_factor / num_classes)
    return smooth_probs
 
async def ensemble_classification(sequence, candidate_labels):
    output1 = classifier(sequence, candidate_labels, multi_label=False)
    # Apply temperature scaling to logits
    scaled_scores1 =await  apply_temperature_scaling(output1['scores'], temperature)
 
    # Apply label smoothing
    smoothed_scores1 = await apply_label_smoothing(scaled_scores1, label_smoothing_factor)
 
    # Convert scaled outputs to dictionaries
    scores1 = {label: score for label, score in zip(output1['labels'], smoothed_scores1)}
 
    scores = {label: scores1[label] for label in candidate_labels}
 
 
    return scores
 
async def custom_post_process(sequence_to_classify, candidate_labels):
    # Your custom logic here
    scores = await ensemble_classification(sequence_to_classify, candidate_labels)
    # print(scores)
    sentiment = max(scores, key=scores.get)
 
    return sentiment.split()[-1], scores
 
async def get_sentiment_from_stars(segment):
    result=pipeline_overall_sentiment(segment)[0]['label']
    if "3" in result:
        return "neutral"
    if "2" in result or "1" in result:
        return "negative"
    
    if "4" in result or "5" in result:
        return "positive"
    
 
 
async def run_sentiment_analysis(segment, aspect):
    if aspect is None or aspect=="":
        result=await get_sentiment_from_stars(segment)
        return result
    candidate_labels = [f"{aspect} positive", f"{aspect} negative"]
    sentiment,_ =await custom_post_process(segment,candidate_labels)
    return sentiment
 