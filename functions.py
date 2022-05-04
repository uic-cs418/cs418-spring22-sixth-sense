import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import dates
from matplotlib.pyplot import figure
import json
import nltk
import sklearn
import warnings


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
"""
This method does sentiment analysis.
Input Param: df = The news/reddit/bitcoinTalk dataframe used to create the plot
Returns: The news/reddit/bitcoinTalk dataframe
"""
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
"""
This method plots the sentiment  distribution during the year.
Input Param: df = The sentiment dataframe used to create the plot
             year = The year for which we have to generate the plot
Returns: The sentiment dataframe of the corressponding year
"""
# This method takes year and returns plots of news per every year
def sentiment_of_popular_cryptos_during_a_particular_year(year, i):
    news1 = i[(i['Year']==year)]
    ax = sns.lineplot(data=news1, x='Month', y='comp_score', hue='crypto',marker='o',ci= None)
    ax.set(xticks=i.Month.values)
    plt.title('news of Coin by Month ' + str(year))
    plt.show()
    return news1
"""
This method plots the sentiment and price of a particular coin distribution during the year.
Input Param: coin = The coin
             year = The year for which we have to generate the plot
"""
#plotting price and sentiment in a single plot for bitcoin
def sentiment_price_of_popular_cryptos_during_a_particular_year(year,coin,sentiment, price):
    popular_cryptos2 = [coin.lower()]
    news_of_popular_cryptos2 = sentiment[sentiment['crypto'] == 'bitcoin']
    news_year = news_of_popular_cryptos2[(news_of_popular_cryptos2['Year']==year)]
    prices_of_popular_cryptos = price[price['currency'].isin(popular_cryptos2)]
    price_year = prices_of_popular_cryptos[(prices_of_popular_cryptos['Year']==year)]
    fig, (ax1,ax2) = plt.subplots(nrows=2, sharex=True, subplot_kw=dict(frameon=False)) # frameon=False removes frames
    
    plt.subplots_adjust(hspace=.2)
    plt.title('Fluctuations in price and sentiment of '+coin+ ' for year ' + str(year) , y= 2.2)
    ax1.grid()
    ax2.grid()
    ax1.plot(news_year['Month'], news_year['comp_score'], color='r',label ='sentiment')
    ax2.plot(price_year['Month'], price_year['price'], color='b', linestyle='--',label ='price')
    ax1.legend(bbox_to_anchor =(1.1, 0.8))
    ax2.legend(bbox_to_anchor =(1.32, 1.6))
    plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
    plt.xlabel('Months')
    ax1.set_ylabel('Sentiment')
    ax2.set_ylabel('Price')
    plt.show()
"""
This method prepares price data for ML
Input Param: df = price
Returns: The price dataframe
"""    
def price_extraction(df):
    df["Date_extracted"] = df["date"].dt.date
    df = df.sort_values('Date_extracted')
    df['price_variation'] = df['price'].diff()
    df['price_raise/drop'] = np.where(df['price_variation']>0, 1, -1)
    df = df.groupby(['Date_extracted'])['price_raise/drop'].sum().reset_index()
    df['price_raise/drop'] = np.where(df['price_raise/drop']>0,1,-1)
    return df
    plt.show()

"""
This method prepares sentiment data for ML
Input Param: df = price
Returns: The price dataframe
"""
#combining price and news for all coins
def comp_score(df):
    df = df.groupby(['Date_extracted'])['comp_score'].sum().reset_index()
    df['comp_score'] = np.where(df['comp_score']>0,1,-1)
    return df
#baseline ML model 

def fit(X, y, z):
    index = z.max()
    length = X.size
    y_predict = np.empty(length,dtype=int)
    for i in range(length):
        y_predict[i] = index
    return y_predict

#converting data in a format compatible with svm
def data_svmReliable_conversion(z):
    z= z.values
    return z.reshape(-1, 1)

# learning the classifier
def learn_classifier(X_train, y_train, kernel):
    classifier = sklearn.svm.SVC(kernel = kernel)
    return classifier.fit(X_train,y_train)

#evaluate the classifier
def evaluate_classifier(classifier, X_validation, y_validation):
    return sklearn.metrics.accuracy_score(y_validation , classifier.predict(X_validation))

#finding the best kernel for svm
def best_model_selection(kf, X, y):
    scores = []
    dict = {}
    for kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
        for train_index, test_index in kf.split(X):
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
            classifier = learn_classifier(X_train,y_train,kernel)
            scores.append(evaluate_classifier(classifier, X_test, y_test))
        dict[kernel] = np.mean(scores)
        scores = []
    return max(dict , key = dict.get)

#prediction on test data
def classify(classifier, x_test):
    return classifier.predict(x_test)

def bar_chart(reddit, news, bitcoinTalk):
    warnings.filterwarnings('ignore')
    news_per_year = news.groupby(['Year']).size().to_frame('size').reset_index()
    reddit_per_year = reddit.groupby(['Year']).size().to_frame('size').reset_index()
    bitcoinTalk_per_year = bitcoinTalk.groupby(['Year']).size().to_frame('size').reset_index()
    news_per_year.loc[-1] = [2015,0]
    news_per_year.index = news_per_year.index + 1  # shifting index
    news_per_year.sort_index(inplace=True) 
    fig,ax = plt.subplots()
    index = np.arange(max(len(reddit_per_year['Year']), len(news_per_year['Year']), len(bitcoinTalk_per_year['Year'])))
    bar_width = 0.2
    opacity = 0.8

    ax.bar(index-bar_width, news_per_year['size'], width=bar_width, label='News')
    ax.bar(index, reddit_per_year['size'], width=bar_width, label='Reddit')
    ax.bar(index+bar_width, bitcoinTalk_per_year['size'], width=bar_width, label='Bitcoin Talk')
    ax.set_yscale('log')
    ax.tick_params(axis=u'both', which=u'both',length=0)
    labels = ['','2015','','2016','','2017','','2018','']
    #ax.set_yticklabels['','2015','','2016','','2017','','2018','']
    ax.set_xticklabels(labels)

    plt.xlabel('Year')
    plt.ylabel('Amount of content')
    plt.title('Amount of content regarding cryptocurrency from 2015-2018')
    plt.legend(loc='best')

    plt.show()

def machine_learning(crypto_prices_df, news):
    #price_sentiment = pd.merge(crypto_prices_df,news_reddit_bitcoin,on='Date_extracted')
    price_sentiment = pd.merge(crypto_prices_df,news,on='Date_extracted')
    x_train, x_test, y_train, y_test= train_test_split(price_sentiment['comp_score'], price_sentiment['price_raise/drop'])
    y_predict = fit(x_train,y_train,crypto_prices_df['price_raise/drop'])
    print("Baseline ML Model Accuracy: ")
    print(sklearn.metrics.accuracy_score(y_train,y_predict))
    #calling classifier
    classifier = learn_classifier(data_svmReliable_conversion(x_train), data_svmReliable_conversion(y_train), 'linear') 
    #calculating accuracy of learned classifier with training data
    accuracy = evaluate_classifier(classifier, data_svmReliable_conversion(x_train), data_svmReliable_conversion(y_train))
    print("Training accuracy: ", accuracy)
    #calculating accuracy of learned classifier with testing data
    test_accuracy = evaluate_classifier(classifier, data_svmReliable_conversion(x_test), data_svmReliable_conversion(y_test))
    print("Test accuracy: ", test_accuracy)
    #predicting 
    y_pred = classify(classifier, data_svmReliable_conversion(x_test))
    print("Prediction on test data")
    print(y_pred)

def KNN(crypto_prices_df, news):
    price_sentiment = pd.merge(crypto_prices_df,news,on='Date_extracted')
    x_train, x_test, y_train, y_test= train_test_split(price_sentiment['comp_score'], price_sentiment['price_raise/drop'])
    #prediction using KNN
    knn = KNeighborsClassifier()
    #Train the model using the training sets
    classifier = knn.fit(data_svmReliable_conversion(x_train), data_svmReliable_conversion(y_train))
    #Predict the response for test dataset
    accuracy = evaluate_classifier(classifier, data_svmReliable_conversion(x_test), data_svmReliable_conversion(y_test))
    print('KNN Test Accuracy: ' , accuracy)

def GNB(crypto_prices_df, news):
    price_sentiment = pd.merge(crypto_prices_df,news,on='Date_extracted')
    x_train, x_test, y_train, y_test= train_test_split(price_sentiment['comp_score'], price_sentiment['price_raise/drop'])
    # Prediction using Gaussian Naive Bayes
    gnb = GaussianNB()
    #Train the model using the training sets
    classifier = gnb.fit(data_svmReliable_conversion(x_train), data_svmReliable_conversion(y_train))
    #Predict the response for test dataset
    accuracy = evaluate_classifier(classifier, data_svmReliable_conversion(x_test), data_svmReliable_conversion(y_test))
    print('Gaussian Naive Bayes Test Accuracy:' , accuracy)

def data_return():
    #Source: https://edloginova.github.io/cryptodata/
    news = pd.read_json('cryptodata/news.json')
    reddit = pd.read_json('cryptodata/reddit.json')
    bitcoinTalk = pd.read_json('cryptodata/bitcointalk.json')
    news = news[(news['date'].dt.year >= 2015) & (news['date'].dt.year <= 2018)]
    reddit = reddit[(reddit['date'].dt.year >= 2015) & (reddit['date'].dt.year <= 2018)]
    bitcoinTalk = bitcoinTalk[(bitcoinTalk['date'].dt.year >= 2015) & (bitcoinTalk['date'].dt.year <= 2018)]
    news = addYearAndMonthColumns(news)
    reddit = addYearAndMonthColumns(reddit)
    bitcoinTalk = addYearAndMonthColumns(bitcoinTalk)
    return news, reddit, bitcoinTalk

def price_of_coin_2017_scatter(crypto_prices_df):
    popular_cryptos = ['bitcoin','ethereum', 'tether', 'binance-coin', 'cardano']
    prices_of_popular_cryptos = crypto_prices_df[crypto_prices_df['currency'].isin(popular_cryptos)]
    #plots of price per year
    price_2016 = price_of_popular_cryptos_during_a_particular_year(prices_of_popular_cryptos,2016)
    price_2017 = price_of_popular_cryptos_during_a_particular_year(prices_of_popular_cryptos,2017)
    price_2018 = price_of_popular_cryptos_during_a_particular_year(prices_of_popular_cryptos,2018)

def sentiment_vs_price(news, crypto_prices_df):
    popular_cryptos2 = ['bitcoin','ethereum', 'tether']
    news_of_popular_cryptos2 = news[news['crypto'].isin(popular_cryptos2)]
    bit = sentiment_price_of_popular_cryptos_during_a_particular_year(2017,'bitcoin',news, crypto_prices_df)
    eth = sentiment_price_of_popular_cryptos_during_a_particular_year(2017,'ethereum',news, crypto_prices_df)
    teth = sentiment_price_of_popular_cryptos_during_a_particular_year(2017,'tether',news, crypto_prices_df)