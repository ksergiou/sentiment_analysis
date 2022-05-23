import matplotlib.pyplot as plt
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import torch
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Select the stocks to be analyzed
tickers = [
           # 'LYG'
             'GSK',
             'TSCO'
            ]
# Execute the news_df.py to create a csv and a df with news headlines by date
exec(open('news_df.py').read())

#####################################################################################
# Define sentiment analysis  functions

def vader_sentiment(df):
    """Sentiment analysis of financial news using Vader.
    Args:
    df (DataFrame): Financial news (daily)
    Returns:
    DataFrame: daily value of sentiment
    Raises:
    ValueError: -
    Notes:
    See https://predictivehacks.com/how-to-run-sentiment-analysis-in-python-using-vader/  """

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
    return mean_df


def finbert_sentiment(df):
    """Sentiment analysis of financial news using FinBERT.
    Args:
    df (DataFrame): Financial news (daily)
    Returns:
    DataFrame: daily value of sentiment
    Raises:
    ValueError: -
    Notes:
    See https://huggingface.co/models """
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    #That's where the headlines get tokenized to be inputted into model
    inputs = tokenizer(list(df['title']), padding = True, truncation = True, return_tensors='pt')
    outputs = model(**inputs)
    #Postprocessing with softmax
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    #print(predictions)
    #print(tf.nn.softmax(outputs.logits, axis=None, name=None  ))  Change the output logitsto dim=-1, i.e detach them
    #Model classes
    model.config.id2label

    df['positive']= predictions[:, 0].tolist()
    df['negative'] = predictions[:, 1].tolist()
    df['neutral']= predictions[:, 2].tolist()
    df['compound']=df['positive']+df['neutral']-df['negative']
    df['date'] = pd.to_datetime(df.date).dt.date

    mean_df = df.groupby(['ticker', 'date']).mean().unstack()
    mean_df = mean_df.xs('compound', axis="columns")
    #mean_df.plot(kind='bar')
    mean_df_s=mean_df.iloc[:, -8:].copy()
    mean_df_s.plot(kind='bar')
    plt.show()

    print(mean_df_s.shape)
    print(mean_df_s.to_string())
    return mean_df_s

############################################################################

finbert_sentiment(df)
