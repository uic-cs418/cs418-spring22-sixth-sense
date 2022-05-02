import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import dates
from matplotlib.pyplot import figure
import json
import nltk
import sklearn

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# import textblob
# from textblob import TextBlob
# import wordcloud
# from wordcloud import WordCloud
# plt.style.use('fivethirtyeight')

"""
This function will add Month and Year Columns in the dataframe
Input Param: df = Dataframe for which we have to add the required columns
Returns: Dataframe with the added columns
"""
def addYearAndMonthColumns(df):
    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month
    return df

"""
This function will add Month,Year and Day Columns in the dataframe
Input Param: df = Dataframe for which we have to add the required columns
Returns: Dataframe with the added columns
"""
def addYearMonthAndDayColumns(df):
    df = addYearAndMonthColumns(df)
    df['Day'] = df['date'].dt.day
    return df

"""
This method plots the cryptocurrency price distribution during the year.
Input Param: df = The price dataframe used to create the plot
             year = The year for which we have to generate the plot
Returns: The price dataframe of the corressponding year
"""
def price_of_popular_cryptos_during_a_particular_year(df,year):
    df = df[(df['Year']==year)]
    ax = sns.lineplot(data=df, x='Month', y='price', hue='currency',marker='o')
    if year != 2018:
        ax.set(xticks=df.Month.values)
    ax.set_title('Price of Coin by Month ' + str(year))
    plt.show()
    return df
def sentiment_analysis(df): 
    sid = SentimentIntensityAnalyzer()
    df['scores'] = df['text'].apply(lambda text: sid.polarity_scores(text))
    df['compound']  = df['scores'].apply(lambda score_dict: score_dict['compound'])
    df['comp_score'] = df['compound'].apply(lambda c: 1 if c >=0 else -1)
    df = df.drop(['scores', 'compound'], axis = 1)
    df["Date_extracted"] = df["date"].dt.date
    def text_extracting(data):
        data = word_tokenize(data)
        for i in range(len(data)):
            if data[i] == 'bitcoin':
                return 'bitcoin'
            elif data[i] == 'ethereum':
                return 'ethereum'
            elif data[i] == 'tether':
                return 'tether'
            elif data[i] == 'binance-coin':
                return 'binance-coin'
            elif data[i] == 'cardano':
                return 'cardano'
        return 'rest'
    df['crypto'] = df['text'].apply(text_extracting)
    df = df.sort_values('Date_extracted')
    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month
    df['Day'] = df['date'].dt.day
    return df
# This method takes year and returns plots of news per every year
def sentiment_of_popular_cryptos_during_a_particular_year(year):
    news1 = news_of_popular_cryptos2[(news_of_popular_cryptos2['Year']==year)]
    ax = sns.lineplot(data=news1, x='Month', y='comp_score', hue='crypto',marker='o',ci= None)
    ax.set(xticks=news_of_popular_cryptos2.Month.values)
    plt.title('news of Coin by Month ' + str(year))
    plt.show()
    return news1
