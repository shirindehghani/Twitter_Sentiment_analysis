import pandas as pd
from log_handler.Setup import logger
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification
from tqdm import tqdm

import json
import warnings

from twitter_sentiment_analysis.create_dataset import MyDataset
from twitter_sentiment_analysis.prediction import prediction
from twitter_sentiment_analysis.preprocess import preprocess
from twitter_sentiment_analysis.train import fit_model
from twitter_sentiment_analysis.load import load_tokenizer_model


warnings.filterwarnings('ignore')
tqdm.pandas()

tokenizer, model= load_tokenizer_model()

class SentimentUsage:
    def __init__(self):
        self.EPOCHS=None
        self.LEARNING_RATE=None
        self.TRAIN_BATCH_SIZE=None
        self.TEST_BATCH_SIZE=None
        self.WARMUP_STEPS=None
        self.WEIGHT_DECAY=None
        self.LOGGING_STEPS=None
        self.predictions_model_path=None
        self.tokenizer_path=None
        self.training_model_path=None
        self.read_config(config_path='./configs/my_config.json')

    def read_config(self, config_path):
        try:
            config_file=open(config_path,'r')
        except IOError:
            logger.error('Specify the correct config path.')
        else:
            my_config=json.load(config_file)
            config_file.close()
            self.EPOCHS=my_config["EPOCHS"]
            self.LEARNING_RATE=my_config["LEARNING_RATE"]
            self.TRAIN_BATCH_SIZE=my_config["TRAIN_BATCH_SIZE"]
            self.TEST_BATCH_SIZE=my_config["TEST_BATCH_SIZE"]
            self.WARMUP_STEPS=my_config["WARMUP_STEPS"]
            self.WEIGHT_DECAY=my_config["WEIGHT_DECAY"]
            self.LOGGING_STEPS=my_config["LOGGING_STEPS"]
            self.predictions_model_path=my_config['predictions_model']
            self.tokenizer_path=my_config['tokenizer']
            self.training_model_path=my_config['training_model']

    def train_model(self, data):
        data['preprocessed_text']=preprocess(data['full_text'])
        data=data[['preprocessed_text','label']]
        df_train, df_validation=train_test_split(data,test_size=0.1,random_state=42)
        train_encodings=tokenizer(df_train["preprocessed_text"].values.tolist(),truncation=True,padding=True)
        val_encodings=tokenizer(df_validation["preprocessed_text"].values.tolist(),truncation=True,padding=True)
        train_dataset=MyDataset(train_encodings, df_train['label'].values.tolist(),use_labels=True)
        val_dataset=MyDataset(val_encodings, df_validation['label'].values.tolist(),use_labels=True)
        model1=AutoModelForSequenceClassification.from_pretrained(self.training_model_path,num_labels=3,ignore_mismatched_sizes=True)
        fit_model(model1,tokenizer,train_dataset,val_dataset,self.EPOCHS,self.TRAIN_BATCH_SIZE,self.TRAIN_BATCH_SIZEself.WARMUP_STEPS,self.WEIGHT_DECAY,self.LOGGING_STEPS,self.LEARNING_RATE,self.predictions_model)
    def predict_labels(self,data):
        data['preprocessed_text']=preprocess(data['full_text'])
        data=data['preprocessed_text']
        df_test=data
        test_encodings=tokenizer(df_test.values.tolist(),truncation=True,padding=True)
        test_dataset=MyDataset(test_encodings)
        test_results=prediction(model,tokenizer,test_dataset,self.TEST_BATCH_SIZE)
        return test_results

    def predict_text(self,text):
        input_data={'full_text':[text]}
        input_df=pd.DataFrame(input_data)
        test_result=self.predict_labels(input_df)
        return test_result
