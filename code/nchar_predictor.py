#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ______                __   _____ __                 __
# /_  __/________ ______/ /__/ ___// /_________  ___  / /_
#  / / / ___/ __ `/ ___/ //_/\__ \/ __/ ___/ _ \/ _ \/ __/
# / / / /  / /_/ / /__/ ,<  ___/ / /_/ /  /  __/  __/ /_
# /_/ /_/   \__,_/\___/_/|_|/____/\__/_/   \___/\___/\__/
#
"""
Created on Wed April 17 13:18:11 2019
Predicting Product or Brand ID from Product Name

@author: Adrian Santuario (adrian@trackstreet.com)
"""

########################################################################
# IMPORT MODULES
########################################################################

import re

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


########################################################################
# TEXT PREDICTOR CLASS
########################################################################


class TextPredictorEngine:

    ####################################################################
    # Life Cycle Methods
    ####################################################################

    def __init__(self):
        if not stopwords:
            nltk.download('stopwords')

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        self.model = None
        self.model_outlier = LocalOutlierFactor()

        self.stop_words = set(stopwords.words('english'))
        self.model_file_name = ""
        self.dict_trick = {}

        self.features = None
        self.labels = None
        self.tfidf = None
        self.is_tfidf_loaded = False

    def __repr__(self):
        return self.description()

    def do_training(self, raw_X, raw_y):
        self.set_X_Y(raw_X, raw_y)
        self.train()

    def description(self):
        description_text = "TextPredictor: Using LinearSVC to get Brand or UPC based on Product Name"
        return description_text

    def set_X_Y(self, X, Y: [int]):
        if not isinstance(X, pd.Series):
            self.features = pd.DataFrame(X)[0]
        else:
            self.features = X

        self.labels = Y

    ####################################################################
    # Process Methods
    ####################################################################

    def transform_input(self, X):
        # lowering and removing punctuation
        X = X.apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))
        # numerical feature engineering
        # total length of sentence
        X.apply(lambda x: len(x))
        # get number of words
        X.apply(lambda x: len(x.split(' ')))
        X.apply(
                lambda x: len([t for t in x.split(' ') if t not in self.stop_words]))
        # get the average word length
        X.apply(lambda x: np.mean([len(t) for t in x.split(' ') if t not in self.stop_words]) if len(
                    [len(t) for t in x.split(' ') if t not in self.stop_words]) > 0 else 0)
        # get the average word length
        X.apply(lambda x: x.count(','))
        return X

    def generate_dict_trick(self, X_phrases):
        frame = {'X': pd.Series(X_phrases), 'y': pd.Series(self.labels)}
        trackstreet_df = pd.DataFrame(frame)
        all_brand_id = trackstreet_df.y.value_counts().index.tolist()
        all_brand_value = trackstreet_df.y.value_counts().tolist()

        for brand, count in zip(all_brand_id, all_brand_value):
            product_text = trackstreet_df[trackstreet_df.y == brand].X.str.cat(sep=' ').lower()
            # function to split text into word
            tokens = word_tokenize(product_text)
            frequency_dist = nltk.FreqDist(tokens)
            # This gives the top word used in the text
            word_brand = sorted(frequency_dist, key=frequency_dist.__getitem__, reverse=True)[0]
            self.dict_trick[str(word_brand)] = int(brand)

    def vectorize(self, X):

        min_df, max_df = self.select_tfidf_params(X)

        try:
            self.tfidf = TfidfVectorizer(sublinear_tf=True, min_df=min_df, max_df=max_df, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                                     stop_words='english')
            X = self.tfidf.fit_transform(X).toarray()
        except ValueError:
            self.tfidf = TfidfVectorizer(sublinear_tf=True, min_df=0.1, max_df=0.5, norm='l2', encoding='latin-1',
                                         ngram_range=(1, 2),
                                         stop_words='english')
            X = self.tfidf.fit_transform(X).toarray()
        return X


    def select_tfidf_params(self, X):
        count_data = len(X)
        min_df = 0.1 - np.log(count_data) / 500
        max_df = 1.0 - np.log(count_data) / 100
        return min_df, max_df

    ####################################################################
    # Train Methods
    ####################################################################

    def train(self):
        # If we set X as a list we transform in a DataFrame
        # The process previous to train
        features_phrases = self.features
        self.features = self.transform_input(self.features)
        self.features = self.vectorize(self.features)
        self.features, self.labels = self.remove_outliers(self.features, self.labels)
        self.generate_dict_trick(features_phrases)
        del features_phrases
        self.fit(self.features, self.labels)
        del self.features, self.labels

    def fit(self, X, y):
        # Find the best SVC
        self.model = self.find_bestEstimator_SVC()
        self.model.fit(X, y)

    ####################################################################
    # Outlier Detection Methods
    ####################################################################
    def find_outliers(self, X):
        params = {
            'n_neighbors': 20,
            'contamination': 0.01,
            'novelty': False
        }

        self.model_outlier = LocalOutlierFactor(**params)

        y_pred_outliers = self.model_outlier.fit_predict(X)
        # self.save_model(self.model_outlier, '{}{}'.format(self.model_file_name, '_outlier'))
        return y_pred_outliers

    def remove_outliers(self, X, y):
        outliers = self.find_outliers(X)
        # Select elements that not are outliers, i.e., if the element is bigger than 0
        return X[outliers > 0], np.array(y)[outliers > 0]

    ####################################################################
    # Model Methods
    ####################################################################

    def find_bestEstimator_SVC(self):
        SVCpipe = Pipeline([('scale', StandardScaler()), ('SVC', LinearSVC())])
        # Gridsearch to determine the value of the best hyperparameters
        param_grid = {'SVC__C': np.arange(0.01, 100, 10)}
        linearSVC_grid = GridSearchCV(SVCpipe, param_grid, cv=2, return_train_score=True)
        try:
            linearSVC_trained = linearSVC_grid.fit(self.features, self.labels)
        except ValueError:
            return LinearSVC().fit(self.features, self.labels)
        return linearSVC_trained.best_estimator_._final_estimator


    def predict(self, x):
        if isinstance(x, list):
            x = pd.DataFrame(x)[0]
        elif isinstance(x, str):
            x = pd.Series(x)
        x = self.transform_input(x)
        predictions = []

        # if self.model is None:
        #     self.model = self.load_model(self.model_file_name)

        for _x in x:
            _x_Raw = _x.split(' ')

            dict_trick_prediction = self.general_dict_trick_predict(_x_Raw)

            if dict_trick_prediction > -1:
                predictions.append(dict_trick_prediction)
            else:
                _x_iterable = [_x]
                _x_tfidf = self.tfidf.transform(_x_iterable)
                predictions.append(self.model.predict(_x_tfidf)[0])

        return predictions

    def general_dict_trick_predict(self, _x_Raw):
        for _x_Word in _x_Raw:
            _x_Word =_x_Word.lower()
            if _x_Word in self.dict_trick:
                return self.dict_trick[_x_Word]
        return -1


########################################################################
# BRAND PREDICTOR CLASS
########################################################################


class BrandPredictorEngine(TextPredictorEngine):

    ####################################################################
    # Life Cycle Methods
    ####################################################################

    def __init__(self):
        super().__init__()
        self.model_file_name = 'brand_model'

    def __repr__(self):
        return self.description()

    def description(self):
        description_text = "Brand Predictor: Using LinearSVC to get Brand based on Product Description"
        return description_text

    ####################################################################
    # Process Methods
    ####################################################################

    def transform_input(self, X):
        # lowering and removing punctuation
        X = X.apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))
        X = X.apply(lambda x: re.sub(r'\d +', '', x))

        sparse_stopword_X = X.apply(lambda x: len([t for t in x.split(' ') if t not in self.stop_words]))
        X = X.apply(lambda x: " ".join(x for x in x.split() if x not in sparse_stopword_X))
        return X

    ####################################################################
    # Train Methods
    ####################################################################

    ####################################################################
    # Predict Methods
    ####################################################################


    ####################################################################


# Model Methods
####################################################################


########################################################################
# PRODUCT PREDICTOR CLASS
########################################################################


class ProductPredictorEngine(TextPredictorEngine):

    ####################################################################
    # Life Cycle Methods
    ####################################################################

    def __init__(self):
        super().__init__()
        self.model_file_name = 'product_model'

    def __repr__(self):
        return self.description()

    def description(self):
        description_text = "TextPredictor: Using LinearSVC to get UPC based on Product Description"
        return description_text

    ####################################################################
    # Process Methods
    ####################################################################

    def transform_input(self, X):
        # lowering and removing punctuation
        X = X.apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))
        X = X.apply(lambda x: re.sub(r'\d +', '', x))

        sparse_stopword_X = X.apply(lambda x: len([t for t in x.split(' ') if t not in self.stop_words]))
        X = X.apply(lambda x: " ".join(x for x in x.split() if x not in sparse_stopword_X))
        return X



    ####################################################################
    # Train Methods
    ####################################################################

    ####################################################################
    # Outlier Detection Methods
    ####################################################################
    def find_outliers(self, X):

        params = {
            'n_neighbors': 4,
            'contamination': 0.001,
            'novelty': False
        }
        self.model_outlier = LocalOutlierFactor(**params)
        y_pred_outliers = self.model_outlier.fit_predict(X)
        # self.save_model(self.model_outlier, '{}{}'.format(self.model_file_name, '_outlier'))
        return y_pred_outliers


    ####################################################################
    # Predict Methods
    ####################################################################

    def predict(self, x):
        if isinstance(x, list):
            x = pd.DataFrame(x)[0]
        elif isinstance(x, str):
            x = pd.Series(x)
        x = self.transform_input(x)

        predictions = []
        #predictions = set()


        for _x in x:
            _x_Raw = _x.split(' ')

            dict_trick_prediction = self.general_dict_trick_predict(_x_Raw)

            if dict_trick_prediction > -1:
                predictions.append(int(dict_trick_prediction))
                #predictions.add(dict_trick_prediction)
            else:
                _x_iterable = [_x]
                _x_tfidf = self.tfidf.transform(_x_iterable)
                predictions.append(int(self.model.predict(_x_tfidf)[0]))
                #predictions = predictions.union({self.model.predict(_x_tfidf)[0]})

        return predictions





####################################################################
# Model Methods
####################################################################
