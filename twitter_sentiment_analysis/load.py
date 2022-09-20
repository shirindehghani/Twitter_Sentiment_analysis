from transformers import AutoTokenizer, AutoModelForSequenceClassification
from log_handler.Setup import logger

def load_tokenizer_model():
    autotokenizer=AutoTokenizer.from_pretrained("./models", use_fast=True, return_tensors="pt")
    logger.info("Tokenier loaded!")
    automodel=AutoModelForSequenceClassification.from_pretrained("./models", ignore_mismatched_sizes=True)
    logger.info("Model loaded!")
    return autotokenizer, automodel
