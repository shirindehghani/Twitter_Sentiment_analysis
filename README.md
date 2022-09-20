Twitter Sentiment Analysis

Metodata

#########################################################################################################################

The model that used in this task is "XLM-t-roberta-sentiment" from huggingface hub!

#########################################################################################################################

needed python version>=3.7.14

#########################################################################################################################

labels are encoding as:

  0->Negative(sad)
  1->Neutral(meh)
  2->Positve(happy)

Note that if you want to train new model on "XLM-t-roberta-sentiment" you should use same encoding for sentiment labels!

#########################################################################################################################

Note that if you want train new model, you should download "cardiffnlp/twitter-xlm-roberta-base-sentiment" from huggingface with VPN!

#########################################################################################################################

If you want to train model on a new dataset, you should create dataframe with one column named as "full_text" and one columns named as "label"

#########################################################################################################################

Before doing any task(predictions and training) you should load model and tokenizer!

#########################################################################################################################

Note that pytorch_model.bin is in.gitignore file;because it is heavy file...
