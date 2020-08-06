#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ______                __   _____ __                 __
# /_  __/________ ______/ /__/ ___// /_________  ___  / /_
#  / / / ___/ __ `/ ___/ //_/\__ \/ __/ ___/ _ \/ _ \/ __/
# / / / /  / /_/ / /__/ ,<  ___/ / /_/ /  /  __/  __/ /_
# /_/ /_/   \__,_/\___/_/|_|/____/\__/_/   \___/\___/\__/
#
"""
Created on Fri March 8 11:04:39 2019
Rating Engine
Predicting Product ID from Product Name

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
from sklearn.feature_extraction.text import TfidfVectorizer


# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC

import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels

# Model evaluation procedure: K-Fold
from sklearn.model_selection import cross_validate


# Model evaluation metrics: F1 Score
from sklearn.metrics import f1_score


########################################################################
# RATING ENGINE CLASS
########################################################################


class RatingEngine:

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

        # Define models
        self.models = [
            LogisticRegression(),
            RandomForestClassifier(),
            LinearSVC(),
            KNeighborsClassifier(),
            MultinomialNB(),
            GaussianNB(),
            BernoulliNB()
        ]

        self.resultados = {}
        self.folds= 5

        self.model = None
        self.model_outlier = LocalOutlierFactor()

        self.stop_words = set(stopwords.words('english'))


        self.features = None
        self.labels = None
        self.tfidf = None
        self.is_tfidf_loaded = False

        self.is_using_hash = False

    def __repr__(self):
        return self.description()


    def description(self):
        description_text = "RatingEngine: Evaluate classifiers to predict Product ID from Product Name"
        return description_text

    def set_X_Y(self, X, Y: [int]):
        if not isinstance(X, pd.Series):
            self.features = pd.DataFrame(X)[0]
        else:
            self.features = X

        self.labels = Y


    ####################################################################
    # Processing Methods
    ####################################################################


    def pre_processing_data(self, X,common_freq_count=10,rare_freq_count=10):
        # Lower Case
        X = X.apply(lambda x: " ".join(x.lower() for x in x.split()))
        print(type(X))
        # Removing Punctuation
        X = X.str.replace('[^\w\s]','')
        # Removing Stop Words
        X = X.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
        # Common Word Removal
        common_freq = pd.Series(' '.join(X).split()).value_counts()[:common_freq_count]
        common_freq = list(common_freq.index)
        X = X.apply(lambda x: " ".join(x for x in x.split() if x not in common_freq))
        # Rare words Removal
        rare_freq = pd.Series(' '.join(X).split()).value_counts()[-rare_freq_count:]
        rare_freq = list(rare_freq.index)
        X = X.apply(lambda x: " ".join(x for x in x.split() if x not in rare_freq))
        
        return X


    def prepare_X_y(self, X,y):
        self.is_using_hash = False

        if is_using_hash is True:
            hashing_tfidf = Pipeline([("hashing", hashing), ("tidf", tfidf_T)])
            features  = hashing_tfidf.fit_transform(X).toarray()
            labels = y
        else:
            features  = tfidf_V.fit_transform(X).toarray()
            labels = y

        #print('Features shape: {}'.format(features.shape))
        return features, labels




    ####################################################################
    # Rating Methods
    ####################################################################

    def f1_multilabel(self, estimador, X, y):
        preds = estimador.predict(X)
        return f1_score(y, preds, average="micro")

    def evaluate_model(self, estimador, X, y):
        resultados_estimador = cross_validate(estimador, X, y,
                     scoring=self.f1_multilabel, n_jobs=-1, cv=self.folds, return_train_score=True, return_estimator=True)
        return resultados_estimador


    def get_results(self):
        resultados_df  = pd.DataFrame(self.resultados).T
        resultados_cols = resultados_df.columns
        for col in resultados_df:
            if col != "estimator":
                resultados_df[col] = resultados_df[col].apply(np.mean)
                resultados_df[col+"_idx"] = resultados_df[col] / resultados_df[col].max()
            else:
                resultados_df[col] = sys.getsizeof(pickle.dumps(resultados_df[col]))/(1000000*folds)
        return resultados_df


    def make_plot_confusion_matrix(self,conf_mat,
                          hide_spines=False,
                          hide_ticks=False,
                          figsize=None,
                          cmap=None,
                          colorbar=False,
                          show_absolute=True,
                          show_normed=False,
                              show_text=True,
                              normed_type="precision"):

        if not (show_absolute or show_normed):
            raise AssertionError('Both show_absolute and show_normed are False')
        
        
        if normed_type == "precision":
            total_samples = conf_mat.sum(axis=1)[:, np.newaxis]
        elif normed_type == "recall":
            total_samples = conf_mat.sum(axis=0)[:, np.newaxis]
        
        
        normed_conf_mat = conf_mat.astype('float') / total_samples

        fig, ax = plt.subplots(figsize=figsize)
        ax.grid(False)
        if cmap is None:
            cmap = plt.cm.Blues

        if figsize is None:
            figsize = (len(conf_mat)*1.25, len(conf_mat)*1.25)

        if show_absolute:
            matshow = ax.matshow(conf_mat, cmap=cmap)
        else:
            matshow = ax.matshow(normed_conf_mat, cmap=cmap)

        if colorbar:
            fig.colorbar(matshow)
            
        if show_text:
            for i in range(conf_mat.shape[0]):
                for j in range(conf_mat.shape[1]):
                    cell_text = ""
                    if show_absolute:
                        cell_text += format(conf_mat[i, j], 'd')
                        if show_normed:
                            cell_text += "\n" + '('
                            cell_text += format(normed_conf_mat[i, j], '.2f') + ')'
                    else:
                        cell_text += format(normed_conf_mat[i, j], '.2f')
                        ax.text(x=j,
                                y=i,
                                s=cell_text,
                                va='center',
                                ha='center',
                                color="white" if normed_conf_mat[i, j] > 0.5 else "black")

        if hide_spines:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        if hide_ticks:
            ax.axes.get_yaxis().set_ticks([])
            ax.axes.get_xaxis().set_ticks([])

        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        return fig, ax






