from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
import numpy as np
from log_handler.Setup import logger

import warnings

warnings.filterwarnings('ignore')

def prediction(model, tokenier, test_dataset, per_device_eval_batch_size):
    test_args=TrainingArguments(output_dir = './results', do_train = False, do_eval=False, do_predict = True, per_device_eval_batch_size = per_device_eval_batch_size, dataloader_drop_last = False)
    trainer=Trainer(model=model, args=test_args, tokenizer=tokenier)
    test_results, _, _ = trainer.predict(test_dataset)
    test_results = np.argmax(test_results, axis=-1)
    logger.info("Prediction finished!")
    return test_results
