import requests
import pandas as pd
import json
from numpy import fromfile

"""Classifier Part"""
# /api/classifiers post
"""
category_names = ['Politics', '2nd Amendment rights', 'Gun control', 
    'Public opinion', 'Mental health', 'School or public space safety', 
    'Society', 'Race', 'Economic consequences']

data = {
    "name": "samesex classifier", "notify_at_email": "vibs97@bu.edu"
    }
res = requests.post('http://www.openframing.org:5000/api/classifiers/', json=data)
print(res.text)
"""

# /api/classifiers/1 get
"""
res = requests.get('http://0.0.0.0:5000/api/classifiers/4')
print(res.text)
"""

# /api/classifiers/1/training/file post
"""
fil = open('testing_files/train_classifier.csv', 'r')
data = {"file": fil}
res = requests.post('http://0.0.0.0:5000/api/classifiers/1/training/file', files=data)
print(res.text)
"""

# /api/classifiers/1/test_sets post
"""
data = {
    "test_set_name": "sample classifier_prediction_set1", "notify_at_email": "vibs97@bu.edu"
}
res = requests.post('http://0.0.0.0:5000/api/classifiers/4/test_sets/', json=data)
print(res.text)
"""

# /api/classifiers/1/set_status_to_be_completed post
"""
category_names = ['Economicconsequences', 'Gun2ndAmendmentrights', 'Guncontrolregulation', 'Mentalhealth', 'Politics', 'Publicopinion', 'Raceethnicity', 'Schoolorpublicspacesafety', 'Societyculture']
metrics = ['0.9261538461538462', '0.879877077719961', '0.8812814391247712', '0.8831354650027471']
data = {
    "category_names": category_names, "metrics": metrics
}
res = requests.post('http://www.openframing.org:5000/api/classifiers/3/set_status_to_be_completed', json=data)
print(res.text)
"""

# /api/classifiers/1/test_sets get
"""
res = requests.get('http://0.0.0.0:5000/api/classifiers/7/test_sets')
print(res.text)
"""

# /api/classifiers/1/test_sets/1/file/ get
"""
fil = open('testing_files/test_classifier.csv', 'r')
data = {"file": fil}
res = requests.post('http://0.0.0.0:5000/api/classifiers/4/test_sets/3/file',files=data)
print(res.text)
"""

# /api/classifiers/1/test_sets/1/predictions get
"""
res = requests.get('http://0.0.0.0:5000/api/classifiers/7/test_sets/1/predictions')
"""


"""Topic Modelling Part"""
# /api/topic_models post
"""
data = {
    "topic_model_name": "all things must pass", "num_topics": 10, 
    "notify_at_email": "vibs97@bu.edu", "language": "english",
    "remove_stopwords": True, "remove_punctuation": True, 
    "do_stemming": True, "do_lemmatizing": True, "min_word_length": 2
    }
res = requests.post('http://0.0.0.0:5000/api/topic_models/', json=data)
print(res.text)
"""

# api/topic_models/1 get
"""
res = requests.get('http://0.0.0.0:5000/api/topic_models/1')
print(res.text)
"""

# api/topic_models/1/training/file/
"""
fil = open('testing_files/step1.csv', 'r')
data = {"file": fil} 
# print(pd.read_csv(fil))
res = requests.post('http://0.0.0.0:5000/api/topic_models/17/training/file', files=data)
print(res.text)
"""

# api/topic_models/1/topics/preview get
"""
res = requests.get('http://0.0.0.0:5000/api/topic_models/4/topics/preview')
print(res.text)
"""

# api/topic_models/1/topics/keywords get
# api/topic_models/1/topics_by_doc get
"""
# This is an excel file so won't be used here but in browser, you can download this.
res = requests.get('http://0.0.0.0:5000/api/topic_models/1/keywords?file_type=xlsx')
print(pd.read_excel(res.raw))
"""

# /topic_models/1/topics/names
"""
data = {"topic_names": ['my', 'name', 'is', 'vubh', '5', '6', '7', '8', '9', '10']}
res = requests.post('http://0.0.0.0:5000/api/topic_models/1/topics/names', json=data)
print(res.text)
"""
