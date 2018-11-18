#
# text processing example: movie review classification as "good" or "bad"
# 11/1/18 THJ
# 

"""
# read in the model using pickle
# accept the text of a movie review
# output if the moview review sttes that the moview is good or bad.

to run in cmd, for example,
python movie_review_deploy.py <text>

example:  python movie_review_deploy2.py "This is a slow, boring movie with no plot."

output: "Good Movie Review"  or "Bad Movie Review"

"""

## import numpy as np
## from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.linear_model import LogisticRegression
import pickle 
import sys

def main(review_text):
    mr_text =[]
    mr_text.append(review_text)
    
    # read the vectorizer and the model
    filenamevect = 'movie_review_vect.sav'
    vect = pickle.load(open(filenamevect,'rb'))
    filenamemodel = 'movie_review_model.sav'
    clf = pickle.load(open(filenamemodel,'rb'))
    
    x = vect.transform(mr_text)
    output_prob = clf.predict_proba(x)
    
    for p in output_prob:
        if p[1] >= p[0]:
            print('Good Movie Review')
        else:
            print('Bad Movie Review')


if __name__ == "__main__":
    main(sys.argv[1])


