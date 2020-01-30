# -*- coding: utf-8 -*-
"""

@author: Samip
"""

#Import the libraries
import pandas as pd
from math import sqrt
"""Pre-processing"""

"""Data downloaded from https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/moviedataset.zip""" 

#Get the movies data
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

#Remove the title from the movie and put in separate year column using (extract)
movies_df['year'] = movies_df['title'].str.extract('(\(\d\d\d\d\))', expand = False)
#Remove the parenthesis
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)', expand = False)
#Remove the years from title column 
movies_df['title'] = movies_df['title'].str.replace('(\(\d\d\d\d\))', '')
#Applying strip to remove the white space at the end
movies_df['title'] = movies_df.title.apply(lambda x : x.strip())

#Dropping genre as we won't need it
movies_df = movies_df.drop('genres', 1)


#Looking at rating column

#Drop timestamp
ratings_df = ratings_df.drop('timestamp', 1)

"""Collaborative Filtering (User-User filtering)"""

#Get the sample user input
userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
            ]

#Store dictionary to a dataframe
inputMovies = pd.DataFrame(userInput) 

#Add movieId to input user

#Filter the movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
#Merge with inputMovies
inputMovies = pd.merge(inputId, inputMovies)
#Drop the year column as is unimportant
inputMovies = inputMovies.drop('year', 1)


#Get the subset of users who have seen the movies and given the ratings
userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
#Group the rows by userId
userSubsetGroup = userSubset.groupby(['userId'])

#Sort the groups so users with more common movies will be in higher priority
userSubsetGroup = sorted(userSubsetGroup, key = lambda x : len(x[1]), reverse = True)


#Similarity of users to input user (using Pearson Coefficient)
"""We use Pearson Co-efficient because it is invarient to scaling. i.e. multiplying
 all elements by a nonzero constant or adding any constant to all elements. For example,
 if you have two vectors X and Y,then, pearson(X, Y) == pearson(X, 2 * Y + 3)"""
 
#Calculating the Pearson Correlation in dictionary, where key is userId and value is co-efficient
pearsonCorrelationDict = {}

for name, group in userSubsetGroup:
    
    #Sort the input and current user group 
    group = group.sort_values(by = 'movieId')
    inputMovies = inputMovies.sort_values(by = 'movieId')
    
    #Get N for formula
    nRatings = len(group)
    #Get the review score that is common in both
    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
    #And then store them in a temporary buffer variable in a list format to facilitate future calculations
    tempRatingList = temp_df['rating'].tolist()
    #Let's also put the current user group reviews in a list format
    tempGroupList = group['rating'].tolist()
    #Now let's calculate the pearson correlation between two users, so called, x and y
    Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)
    Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)
    Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)
    
    #If the denominator is different than zero, then divide, else, 0 correlation.
    if Sxx != 0 and Syy != 0:
        pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
    else:
        pearsonCorrelationDict[name] = 0
pearsonCorrelationDict.items()

#Convert dictionary to dataframe
pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))
pearsonDF.head()

#The top x similar users to input user
topUsers=pearsonDF.sort_values(by = 'similarityIndex', ascending=False)

#Rating of selected users to all movies
topUsersRating=topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')

#Multiplies the similarity by the user's ratings
topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']

#Applies a sum to the topUsers after grouping it up by userId
tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
tempTopUsersRating.head()

#Creates an empty dataframe
recommendation_df = pd.DataFrame()
#Now we take the weighted average
recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
recommendation_df['movieId'] = tempTopUsersRating.index
recommendation_df.head()

recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
recommendation_df.head(10)

movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())]

