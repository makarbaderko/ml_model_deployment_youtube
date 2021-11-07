#Imports
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def create_model():
    data = pd.read_csv("spam.csv")
    data["Category"] = data["Category"].str.replace("ham", "0")
    data["Category"] = data["Category"].str.replace("spam", "1")

    model=Pipeline([
        ('vectorizer',CountVectorizer()),
        ('classifirer',MultinomialNB())
    ])
    model.fit(data.Message, data.Category)
    return model