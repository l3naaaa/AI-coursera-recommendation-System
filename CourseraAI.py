# -*- coding: utf-8 -*-
"""Recommendation System.ipynb
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
print('Dependencies Imported')

"""## First Model:"""

data = pd.read_csv("/content/Dataset.csv")
descriptions = data['gender'] +' '+ data['subject'] + ' ' + data['stream'] +' '+ data['marks'].apply(str)

print(f'The total number of recommendations is {descriptions.count()}')

# Splitting the descriptions into test and train sets
from sklearn.model_selection import train_test_split

descriptions_train, test_descriptions = train_test_split(descriptions, test_size = 0.1, random_state=42)

# import TfidfVector from sklearn.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
def create_similarity_matrix(new_description, overall_descriptions):
        #Append the new description to the overall set.
        pd.concat([overall_descriptions, new_description])
        # Define a tfidf vectorizer and remove all stopwords.
        tfidf = TfidfVectorizer(stop_words="english")
        #Convert tfidf matrix by fitting and transforming the data.
        tfidf_matrix = tfidf.fit_transform(overall_descriptions)
        # output the shape of the matrix.
        tfidf_matrix.shape
        # print(tfidf_matrix)
        # calculating the cosine similarity matrix.
        cosine_sim = linear_kernel(tfidf_matrix,tfidf_matrix)
        return cosine_sim

def get_recommendations(new_description,overall_descriptions):
        # create the similarity matrix
        cosine_sim = create_similarity_matrix(new_description,overall_descriptions)
        # Get pairwise similarity scores of all the students with new student.
        sim_scores = list(enumerate(cosine_sim[-1]))
        # Sort the descriptions based on similarity score.
        sim_scores = sorted(sim_scores,key =lambda x:x[1],reverse= True )
        # Get the scores of top 10 descriptions.
        sim_scores = sim_scores[1:10]
        # Get the student indices.
        indices = [i[0]for i in sim_scores]
        return data.iloc[indices]

"""### Test Cases:"""

# Perform testing and evaluate relevance
for description in test_descriptions:
    recommendations = get_recommendations(pd.Series(description), descriptions_train)
    print("Test Description:", description)
    print("Recommended Courses:")
    print(recommendations)
    print()

# Evaluate diversity
all_recommendations = []
for description in test_descriptions:
    recommendations = get_recommendations(pd.Series(description), descriptions_train)
    all_recommendations.extend(recommendations)

unique_recommendations = set(all_recommendations)
diversity_score = len(unique_recommendations) / len(all_recommendations)

print("Diversity Score:", diversity_score * 100)

"""### Second Model:"""

!unzip "/content/Coursera.csv.zip" -d "/content"

data = pd.read_csv("../content/Coursera.csv")
data.head(5)

data.shape

data.info()

data.isnull().sum()

data['Difficulty Level'].value_counts()

data['Course Rating'].value_counts()

data['University'].value_counts()

data['Course Name']

data = data[['Course Name','Difficulty Level','Course Description','Skills']]
data.head(5)

"""Pre-processing:"""

# Removing spaces between the words (Lambda funtions can be used as well)

data['Course Name'] = data['Course Name'].str.replace(' ',',')
data['Course Name'] = data['Course Name'].str.replace(',,',',')
data['Course Name'] = data['Course Name'].str.replace(':','')
data['Course Description'] = data['Course Description'].str.replace(' ',',')
data['Course Description'] = data['Course Description'].str.replace(',,',',')
data['Course Description'] = data['Course Description'].str.replace('_','')
data['Course Description'] = data['Course Description'].str.replace(':','')
data['Course Description'] = data['Course Description'].str.replace('(','')
data['Course Description'] = data['Course Description'].str.replace(')','')

#removing paranthesis from skills columns
data['Skills'] = data['Skills'].str.replace('(','')
data['Skills'] = data['Skills'].str.replace(')','')
data.head(5)

data['tags'] = data['Course Name'] + data['Difficulty Level'] + data['Course Description'] + data['Skills']
data.head(5)

new_df = data[['Course Name','tags']]
new_df.head(5)

new_df['tags'] = data['tags'].str.replace(',',' ')

new_df['Course Name'] = data['Course Name'].str.replace(',',' ')
new_df.rename(columns = {'Course Name':'course_name'}, inplace = True)
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower()) #lower casing the tags column
new_df.head(5)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

#defining the stemming function
def stem(text):
    y=[]

    for i in text.split():
        y.append(ps.stem(i))

    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)

# Similarity Measure
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)

def recommend(course):
    course_index = new_df[new_df['course_name'] == course].index[0]
    distances = similarity[course_index]
    course_list = sorted(list(enumerate(distances)),reverse=True, key=lambda x:x[1])[1:7]

    for i in course_list:
        print(new_df.iloc[i[0]].course_name)

recommend('Business Strategy Business Model Canvas Analysis with Miro')

test_set = ['Silicon Thin Film Solar Cells','Parallel programming','Business Strategy Business Model Canvas Analysis with Miro', 'Python Programming','The Changing Arctic']
for i in test_set:
  recommend(i)
  print("\n")

# Evaluate diversity
all_recommendations = []
for i in test_set:
    recommendations = recommend(i)
    all_recommendations.extend(recommendations)

unique_recommendations = set(all_recommendations)
diversity_score = len(unique_recommendations) / len(all_recommendations)

print("Diversity Score:", diversity_score * 100)