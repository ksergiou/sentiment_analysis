import matplotlib.pyplot as plt
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import torch
import tensorflow as tf

tickers = ['LYG', 'GSK', 'TSCO']

exec(open('news_df.py').read())


def vader_sentiment(df):
    """

    :rtype: plot a barplot of daily compound sentiment scores 
    """
    vader = SentimentIntensityAnalyzer()

    f = lambda x: vader.polarity_scores(x)['compound']           # lies in [-1,1]
    df['compound'] = df['title'].apply(f)
    df['date'] = pd.to_datetime(df.date).dt.date

    # plt.figure(figsize=(10,8))
    mean_df = df.groupby(['ticker', 'date']).mean().unstack()
    mean_df = mean_df.xs('compound', axis="columns")
    mean_df.plot(kind='bar')
    plt.show()

    print(mean_df.shape)
    print(mean_df.to_string())


# vader_sentiment(df)

# Getting the tokenizer and the model
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

#That's where the headlines get tokenized to be inputted into model

inputs = tokenizer(list(df['title']), padding = True, truncation = True, return_tensors='pt')

outputs = model(**inputs)
#Postprocessing with softmax


predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)

#print(tf.nn.softmax(outputs.logits, axis=None, name=None  ))  Change the output logitsto dim=-1, i.e detach them