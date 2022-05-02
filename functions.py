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
