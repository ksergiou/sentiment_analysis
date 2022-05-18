from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

tickers = ['AMZN', 'GOOG', 'FB']

exec(open('news_df.py').read())

vader = SentimentIntensityAnalyzer()

f = lambda x: vader.polarity_scores(x)['compound']
df['compound'] = df['title'].apply(f)
df['date'] = pd.to_datetime(df.date).dt.date

#plt.figure(figsize=(10,8))
mean_df = df.groupby(['ticker', 'date']).mean().unstack()
mean_df = mean_df.xs('compound', axis="columns")
mean_df.plot(kind='bar')
plt.show()

print(mean_df.shape)
print(mean_df.to_string())

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   None

