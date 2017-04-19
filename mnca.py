# -*- coding: utf-8 -*-

import requests
import os
import json
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
#from urlparse import urlparse
#import schedule
import datetime
import newspaper
import time
import string
import re
from flask import Flask, jsonify, request
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#from vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from oauth2client.client import GoogleCredentials
from google.cloud import storage

app = Flask(__name__)
app.config['PORT'] = os.getenv("PORT", "8085")
app.config['DEBUG'] = os.getenv("DEBUG", "true") == "true"

def remove_cap_punc(in_string):
    """Function removes capitalization and punctuation from string."""
    out_string = in_string
    translator = str.maketrans('', '', string.punctuation)
    out_string = out_string.translate(translator)
    out_words = out_string.split()
    out_words = [word.lower() for word in out_words]
    out_string = ' '.join(word for word in out_words)
    return(out_string)

def remove_overfit_words(in_string, wordlist, sourcelist, phraselist):
    """Function removes words and phrases indicative of source from string.
    This function requires a wordlist of individual words to exclude, a
    sourcelist of sources to exclude, and a phraselist of phrases to exclude.
    The purpose of this function is to prevent our learner from making undesired
    associations between words and phrases that aren't themselves indicative of
    a classification beyond their existence in a large proportion of the articles
    in our training corpus."""
    out_string = in_string
    for phrase in phraselist:
        out_string = out_string.replace(phrase, '')
    out_string = out_string.lower()
    for source in sourcelist:
        out_string = out_string.replace(source, '')
    non_url_words = [word for word in out_string.split() if ("www" not in word) and ("https" not in word)] #remove URLs
    out_words = [word for word in non_url_words if word not in wordlist]
    out_string = ' '.join(word for word in out_words)
    return(out_string)

def remove_shortwords(in_string):
    """Function removes words with length <=2 from string."""
    out_string = in_string
    out_words = out_string.split()
    out_words = [word for word in out_words if len(word) > 2]
    out_string = ' '.join(word for word in out_words)
    return(out_string)

def split_into_sentences(text):
    """Function accepts a string and returns a list of the sentences inside it."""
    caps = "([A-Z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov)"
    
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
    #if """ in text: text = text.replace("."","".")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return(sentences)

def sent_analysis(text, uoa="sentences"):
    """If uoa="sentences", function accepts a string with multiple sentences and
    returns the average sentiment score for all sentences. If uoa="string",
    function returns the overall sentiment score for the string."""
    if uoa == "sentences":
        sid = SentimentIntensityAnalyzer()
        counter=0
        total_compound=0
        for sentence in text:
            ss = sid.polarity_scores(sentence)
            total_compound = total_compound + ss['compound']
            counter+=1

        if counter==0:
            avg_compound=0
        else:
            avg_compound = total_compound/counter

        return(avg_compound)
    
    elif uoa == "string":
        filtered_text = remove_cap_punc(text)
        sid = SentimentIntensityAnalyzer()
        ss = sid.polarity_scores(filtered_text)
        compound = ss['compound']
        return(compound)
    else:
        print("uoa (unit of analysis) not recognized")
        
def pct_char_quesexcl(title):
    """Function accepts a string and returns the % of characters that are question
    marks or exclamation points."""
    try:
        ques_excl = [char for char in title if char=='?' or char=='!']
        return(len(ques_excl)/len(title))
    except:
        return(0)        

def pct_punct_quesexcl(in_string):
    """Function accepts a string and returns the % of punctuation in it that are 
    question marks or exclamation points."""
    try:
        punct = [char for char in in_string if char in string.punctuation]
        ques_excl = [p for p in punct if p=='?' or p=='!']
        return(len(ques_excl)/len(punct))
    except:
        return(0)
    
def pct_allcaps(title):
    """Function accepts a string and returns the % of words in it that are ALL
    CAPS."""
    try:
        translator = str.maketrans('', '', string.punctuation)
        title = title.translate(translator)
        words = title.split()
        all_caps = [word for word in words if word.isupper()]
        return(len(all_caps)/len(words))
    except:
        return(0)

@app.route('/mnca-train')    
def train_classifiers():
    """Function trains a MNB and AdaBoost ensemble classifier on an even split of credible/
    non-credible news articles. This function takes care of the sampling and
    preprocessing and pickles the two classification models and outputs a
    JSON file containing important statistics for article classification in the
    mnca module."""
    cred_fp = '/ebs_volume/data/Credible/'
    ncred_fp = '/ebs_volume/data/notCredible/'

    articles = pd.DataFrame(columns=('label',
                                     'text',
                                     'title',
                                     'date',
                                     'source'))
    i = 0    
    for root, dirs, files in os.walk(cred_fp):
        for file in files:
            if file.endswith(".txt") and 'api' not in file:
                 curr_file = os.path.join(root, file)
                 #print(curr_file)
                 with open(curr_file) as json_file:
                    try:
                        data = json.load(json_file)
                        if data["source"] == "new-york-times":
                            articles.loc[i] = [0,data["text"],data["title"],data["date"],"the-new-york-times"]
                        else:                        
                            articles.loc[i] = [0,data["text"],data["title"],data["date"],data["source"]]
                        i+=1
                    except ValueError:
                        continue

    for root, dirs, files in os.walk(ncred_fp):
        for file in files:
            if file.endswith(".txt") and 'api' not in file:
                 curr_file = os.path.join(root, file)
                 #print(curr_file)
                 with open(curr_file) as json_file:
                    try:
                        data = json.load(json_file)
                        articles.loc[i] = [1,data["text"],data["title"],data["date"],data["source"]]
                        i+=1
                    except ValueError:
                        continue
                        
    unique_articles = articles.drop_duplicates(subset = 'text') #remove duplicates
    unique_articles = unique_articles[unique_articles["text"].str.len()>200] #remove really short articles

    #cred_articles = unique_articles[unique_articles["label"]==0.0]
    cred_articles = unique_articles[unique_articles["source"].isin(["new-york-times","the-new-york-times","reuters","the-wall-street-journal","the-washington-post","usa-today"])]
    #noncred_articles = unique_articles[unique_articles["label"]==1.0]
    noncred_articles = unique_articles[unique_articles["source"].isin(["activistpost","dcclothesline","gopthedailydose","infostormer","rickwells","success-street","usanewsflash","usapoliticsnow","usasupreme"])]

    cred_articles = cred_articles[~cred_articles["date"].isin(list(set(cred_articles["date"]) - set(noncred_articles["date"])))]
    date_cnts = Counter(cred_articles["date"])
    noncred_even = pd.DataFrame(columns=('label','text','title','date','source'))

    #Ensure an even distribution of publish dates in training set
    for date in date_cnts:
        noncred_even = pd.concat([noncred_even, noncred_articles[noncred_articles["date"]==date].sample(n=date_cnts[date])])

    even_articles = pd.concat([cred_articles, noncred_even]) #train set should contain even number of cred/noncred articles
    
    #Load in list of overfit words & phrases from training sources
    wordlist = ["advertisement", "skip", "main", "photo", "embed", "www", "com",
                "https", "http", "photo", "getty", "continue", "sunday", "monday",
                "tuesday", "wednesday", "thursday", "friday", "saturday", "stopthetakeover"
                "sidebar", "usatwentyfour"]

    #Generate sources list
    sources = list(set(even_articles['source']))
    sourcelist = [source.replace('-', ' ') for source in sources]
    sourcelist.extend(['rickrwells', 'rickwells', 'rick wells', 'wall street journal', 'gop the daily dose', 'new york times', 'washington post', 'activist post', 'wsj'])


    #Generate indicative phrase list from training sources
    phraselist = ["Share this:",
                  "by usapoliticsnow admin",
                  "Our Standards: The Thomson Reuters Trust Principles",
                  "Don't forget to follow the D.C. Clothesline on Facebook and Twitter. PLEASE help spread the word by sharing our articles on your favorite social networks.",
                  "Share With Your Friends On Facebook, Twitter, Everywhere",
                  "Thank you for reading and sharing my work –  Please look for me, Rick Wells, at http://www.facebook.com/RickRWells/ , http://www.gab.ai/RickRWells , https://plus.google.com/u/0/+RickwellsUs and on my website http://RickWells.US  – Please SUBSCRIBE in the right sidebar at RickWells.US – not dot com.  I'm also at Stop The Takeover, https://www.facebook.com/StopTheTakeover/ and please follow me on Twitter @RickRWells. Subscribe also on my YouTube Channel.",
                  "Like this Article? Share it!",
                  "Do you have information the public should know? Here are some ways you can securely send information and documents to Post journalists.",
                  "Share news tips with us confidentially",
                  "Share on Facebook",
                  "Tweet on Twitter",
                  "We encourage you to share and republish our reports, analyses, breaking news and videos (Click for details).",
                  "Next post",
                  "Previous post",
                  "Thank you for reading and sharing my work – Facebook is trying to starve us out of existence, having cut literally 98% of our traffic over the last year. Your shares are crucial for our survival, and we thank you. We’ve also created a presence on Gab.ai and MeWe.com, although their reach is presently much smaller, the continued abuse by Facebook of conservative voices leaves us no option. We’re remaining on Facebook for the time being, as we make the transition. Please take a look when you have a chance or if we “suddenly disappear” from Facebook as has happened to many other truth-tellers. They’ll either starve us out or take us down, one way or another, sooner or later. Now and in the future, please look for me, Rick Wells, at http://www.facebook.com/RickRWells/ , http://www.gab.ai/RickRWells , https://mewe.com/profile/rick.wells.1 and on my website http://RickWells.US – Please SUBSCRIBE in the right sidebar at RickWells.US – not dot com. I’m also at Stop The Takeover, https://www.facebook.com/StopTheTakeover/ and please follow me on Twitter @RickRWells."]

    even_articles['filtered_text'] = even_articles.apply(lambda x: remove_overfit_words(x['text'], wordlist=wordlist, sourcelist=sourcelist, phraselist=phraselist), axis=1)
    even_articles['filtered_text'] = even_articles['filtered_text'].apply(remove_shortwords)

    even_articles['sentences'] = even_articles['text'].apply(split_into_sentences)
    even_articles['text_sentiment'] = even_articles['sentences'].apply(sent_analysis)
    even_articles['title_sentiment'] = even_articles.apply(lambda x: sent_analysis(x['title'], uoa="string"), axis=1)
    even_articles['pct_char_quesexcl_title'] = even_articles['title'].apply(pct_char_quesexcl)
    even_articles['pct_punc_quesexcl_text'] = even_articles['text'].apply(pct_punct_quesexcl)
    even_articles['pct_allcaps_title'] = even_articles['title'].apply(pct_allcaps)
    
    mnb_clf = Pipeline(steps = [('tfidf', TfidfVectorizer(analyzer='word', ngram_range=(1,1), min_df=0, lowercase=True, stop_words='english')),
                            ('clf', MultinomialNB())])

    mnb_scores=[]
    ada_scores=[]
    k_fold = KFold(n_splits=5, shuffle=True)

    for train_index, test_index in k_fold.split(even_articles):
        train_text = even_articles.iloc[train_index]['filtered_text'].values
        train_x = np.array([even_articles.iloc[train_index]['pct_allcaps_title'].values,
                            even_articles.iloc[train_index]['pct_punc_quesexcl_text'].values,
                            even_articles.iloc[train_index]['pct_char_quesexcl_title'].values,
                            even_articles.iloc[train_index]['text_sentiment'].values,
                            even_articles.iloc[train_index]['title_sentiment'].values])
        train_y = even_articles.iloc[train_index]['label'].values
        
        
        test_text = even_articles.iloc[test_index]['filtered_text'].values
        test_x = np.array([even_articles.iloc[test_index]['pct_allcaps_title'].values,
                           even_articles.iloc[test_index]['pct_punc_quesexcl_text'].values,
                           even_articles.iloc[test_index]['pct_char_quesexcl_title'].values,
                           even_articles.iloc[test_index]['text_sentiment'].values,
                           even_articles.iloc[test_index]['title_sentiment'].values])
        test_y = even_articles.iloc[test_index]['label'].values

        #MNB CLASSIFIER
        mnb_clf.fit(train_text, train_y)
        mnb_predictions = mnb_clf.predict(test_text)
        mnb_score = accuracy_score(test_y, mnb_predictions)
        mnb_scores.append(mnb_score)

        #ADABOOST
        ada_clf = AdaBoostClassifier(n_estimators=100).fit(train_x.T, train_y)
        ada_predictions = ada_clf.predict(test_x.T)
        ada_score = accuracy_score(test_y, ada_predictions)
        ada_scores.append(ada_score)

    perform_statistics = {}

    #Calculate classification accuracy for ensemble weighting    
    perform_statistics['mnb_accuracy'] = sum(mnb_scores)/len(mnb_scores)
    perform_statistics['ada_accuracy'] = sum(ada_scores)/len(ada_scores)

    #Calculate mean scores for cred vs. noncred "tonal" features to assist with interpreting the model's decisions
    perform_statistics['mean_sent_score_text_cred'] = np.mean(even_articles['text_sentiment'][even_articles['label']==0])
    perform_statistics['mean_sent_score_text_noncred'] = np.mean(even_articles['text_sentiment'][even_articles['label']==1])
    perform_statistics['mean_sent_score_title_cred'] = np.mean(even_articles['title_sentiment'][even_articles['label']==0])
    perform_statistics['mean_sent_score_title_noncred'] = np.mean(even_articles['title_sentiment'][even_articles['label']==1])
    perform_statistics['mean_pct_punc_text_cred'] = np.mean(even_articles['pct_punc_quesexcl_text'][even_articles['label']==0])
    perform_statistics['mean_pct_punc_text_noncred'] = np.mean(even_articles['pct_punc_quesexcl_text'][even_articles['label']==1])
    perform_statistics['mean_pct_punc_title_cred'] = np.mean(even_articles['pct_char_quesexcl_title'][even_articles['label']==0])
    perform_statistics['mean_pct_punc_title_noncred'] = np.mean(even_articles['pct_char_quesexcl_title'][even_articles['label']==1])
    perform_statistics['mean_pct_allcaps_title_cred'] = np.mean(even_articles['pct_allcaps_title'][even_articles['label']==0])
    perform_statistics['mean_pct_allcaps_title_noncred'] = np.mean(even_articles['pct_allcaps_title'][even_articles['label']==1])
    
    train_x = np.array([even_articles['pct_allcaps_title'].values,
                    even_articles['pct_punc_quesexcl_text'].values,
                    even_articles['pct_char_quesexcl_title'].values,
                    even_articles['text_sentiment'].values,
                    even_articles['title_sentiment'].values])
    train_y = even_articles['label'].values

    mnb_clf = mnb_clf.fit(even_articles['filtered_text'], train_y)
    ada_clf = AdaBoostClassifier(n_estimators=100).fit(train_x.T, train_y)

    #Change this to upload to google cloud

    joblib.dump(mnb_clf, 'mnb_clf.pkl')
    joblib.dump(ada_clf, 'ada_clf.pkl')

    with open('DONOTDELETE.json', 'w') as outfile:
        json.dump(perform_statistics, outfile)


def most_predictive_feats(filtered_text, clf, label, n=10):
    """Function returns most predictive words (by differential log_prob) in 
    filtered_text in support of a "Credible" or "Non-Credible" labeling. n 
    can be set to adjust the number of words returned."""
    features = pd.DataFrame()
    features["word"] = clf.named_steps['tfidf'].get_feature_names()
    features["tfidf"] = list(clf.named_steps['tfidf'].transform([filtered_text]).toarray()[0])
    features["log_prob_0"] = clf.named_steps['clf'].feature_log_prob_[0]
    features["log_prob_1"] = clf.named_steps['clf'].feature_log_prob_[1]
    labels = []
    for lp0, lp1 in zip(features["log_prob_0"], features["log_prob_1"]):
        if lp1 > lp0:
            labels.append(1)
        else:
            labels.append(0)
    features["label"] = labels
    features["log_prob_diff"] = abs(features["log_prob_1"] - features["log_prob_0"])
    features["weighted_log_prob_diff"] = features["tfidf"] * features["log_prob_diff"]
    features_c_sort = features[features["label"]==0].sort_values(by=["weighted_log_prob_diff"], ascending=False)
    features_nc_sort = features[features["label"]==1].sort_values(by=["weighted_log_prob_diff"], ascending=False)
    if label == "Non-Credible":
        word_features = list(features_nc_sort["word"].head(n))
    elif label == "Credible":
        word_features = list(features_c_sort["word"].head(n))
        
    return(word_features)


def classify_article(article):
    """Function accepts articles in the form of Python dictionaries
    containing the raw text (article['text']) and title (article['title']).
    It returns a Python dictionary containing the classification label
    (label), probability (label_prob), and a dictionary containing the
    word and tonal features that had the greatest impact on classification
    (interpretation). Model weights stored on google cloud."""


    article['filtered_text'] = remove_shortwords(article['text'])
    article['sentences'] = split_into_sentences(article['text'])
    article['text_sentiment'] = sent_analysis(article['sentences'])
    article['title_sentiment'] = sent_analysis(article['title'], uoa="string")
    article['pct_char_quesexcl_title'] = pct_char_quesexcl(article['title'])
    article['pct_punc_quesexcl_text'] = pct_punct_quesexcl(article['text'])
    article['pct_allcaps_title'] = pct_allcaps(article['title'])
    
    project_id = 'genuine-charger-124604'
    credentials = GoogleCredentials.get_application_default()

    client = storage.Client(project=project_id)
    bucket = client.get_bucket('genuine-charger-124604.appspot.com')
    model_weights_file = bucket.blob('DONOTDELETE.json')#.read_from()
    txt = model_weights_file.download_as_string()
    print(txt)
    weights = txt.decode('utf-8').strip('\ufeff')
    #model_weights_file.download_to_filename('weights.json')
    #load the weights
    perform_statistics = json.loads(weights)

    #with open('DONOTDELETE.json') as json_data:
        #perform_statistics = json.load(json_data)

    ada_feats = [article['pct_allcaps_title'],
                 article['pct_punc_quesexcl_text'],
                 article['pct_char_quesexcl_title'],
                 article['text_sentiment'],
                 article['title_sentiment']]

    #read model pickel file from gcs
   
    mnb_clf_pkl_file = bucket.blob('mnb_clf.pkl')#.read_from()
    mnb_clf_pkl_file.download_to_filename('model1.pkl')
    #load the model and predict
    mnb_clf = joblib.load('model1.pkl')

    ada_clf_pkl_file = bucket.blob('ada_clf.pkl')#.read_from()
    ada_clf_pkl_file.download_to_filename('model2.pkl')
    #load the model and predict
    ada_clf = joblib.load('model2.pkl')
    
    mnb_prob = mnb_clf.predict_proba([article['filtered_text']])
    ada_prob = ada_clf.predict_proba([np.asarray(ada_feats)])
    
    classification = {}
    classification["label_prob"] = [(((perform_statistics['mnb_accuracy']*mnb_prob[0][0])+(perform_statistics['ada_accuracy']*ada_prob[0][0]))/(perform_statistics['mnb_accuracy']+perform_statistics['ada_accuracy'])), (((perform_statistics['mnb_accuracy']*mnb_prob[0][1])+(perform_statistics['ada_accuracy']*ada_prob[0][1]))/(perform_statistics['mnb_accuracy']+perform_statistics['ada_accuracy']))]
    if classification["label_prob"][1] > classification["label_prob"][0]:
        classification["label"] = "Non-Credible"
    else:
        classification["label"] = "Credible"
            
    interpretation = {}
    interpretation["word_feats"] = most_predictive_feats(article["filtered_text"], mnb_clf, classification["label"], n=10)
    tone_feats = {}
    
    tone_feat_classifications = []
    if abs(article['pct_allcaps_title'] - perform_statistics['mean_pct_allcaps_title_noncred']) < abs(article['pct_allcaps_title'] - perform_statistics['mean_pct_allcaps_title_cred']):
        tone_feat_classifications.append(1)
    else:
        tone_feat_classifications.append(0)
    
    if abs(article['pct_punc_quesexcl_text'] - perform_statistics['mean_pct_punc_text_noncred']) < abs(article['pct_punc_quesexcl_text'] - perform_statistics['mean_pct_punc_text_cred']):
        tone_feat_classifications.append(1)
    else:
        tone_feat_classifications.append(0)
    
    if abs(article['pct_char_quesexcl_title'] - perform_statistics['mean_pct_punc_title_noncred']) < abs(article['pct_char_quesexcl_title'] - perform_statistics['mean_pct_punc_title_cred']):
        tone_feat_classifications.append(1)
    else:
        tone_feat_classifications.append(0)
        
    if abs(article['text_sentiment'] - perform_statistics['mean_sent_score_text_noncred']) < abs(article['text_sentiment'] - perform_statistics['mean_sent_score_text_cred']):
        tone_feat_classifications.append(1)
    else:
        tone_feat_classifications.append(0)
        
    if abs(article['title_sentiment'] - perform_statistics['mean_sent_score_title_noncred']) < abs(article['title_sentiment'] - perform_statistics['mean_sent_score_title_cred']):
        tone_feat_classifications.append(1)
    else:
        tone_feat_classifications.append(0)
        
    if (tone_feat_classifications[0]*ada_clf.feature_importances_[0] + tone_feat_classifications[1]*ada_clf.feature_importances_[1] + tone_feat_classifications[2]*ada_clf.feature_importances_[2])/( ada_clf.feature_importances_[0] + ada_clf.feature_importances_[1] + ada_clf.feature_importances_[2]) > 0.5:
        convention_pred = "Non-Credible"
    else:
        convention_pred = "Credible"
        
    if (tone_feat_classifications[3]*ada_clf.feature_importances_[3] + tone_feat_classifications[4]*ada_clf.feature_importances_[4])/( ada_clf.feature_importances_[3] + ada_clf.feature_importances_[4]) > 0.5:
        sentiment_pred = "Non-Credible"
    else:
        sentiment_pred = "Credible"
    
    if convention_pred == classification["label"]:
        tone_feats["convention"] = convention_pred
        
    if sentiment_pred == classification["label"]:
        tone_feats["sentiment"] = sentiment_pred
    
    interpretation["tone_feats"] = tone_feats
   
    classification["interpretation"] = interpretation
        
    return(classification)


@app.route('/mnca-classify-url',methods=['POST'])
def classify_url():
    """Function that takes the url string of a news article and returns the
    classifciation """
    if request.method == 'POST':

        #url = request.form.query
        url = request.form.get('url')      

        link = newspaper.Article(url)
        link.download()
        link.parse()

        article = {}
        article["title"] = link.title
        article["text"] = link.text

    return jsonify(classify_article(article))

@app.route('/mnca-classify-article-content',methods=['POST'])
def classify_article_content():
    """Function that takes the article as a json input with "title" and "text" as input keys and returns the classification"""
    
    #article = extract_article(url)
    article = request.get_json(force=True)
    #article = {}
    #article["title"] = title
    #article["text"] = text
    #result = classify_article(article)
    
    return jsonify(classify_article(article))
    
    
if __name__ == '__main__':
  if app.config['DEBUG']:
      app.run(host='127.0.0.1', port=int(app.config['PORT']), debug=True)
  else:
    app.run(host='0.0.0.0', port=int(app.config['PORT']), debug=False)