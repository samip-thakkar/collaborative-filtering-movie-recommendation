# collaborative-filtering-movie-recommendation
Using User-user similarity model, recommend the movies using collaborative filtering
Data is collected from: https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/moviedataset.zip

Steps:
1. Select a user with the movies the user has watched
2. Based on his rating to movies, find the top X neighbours
3. Get the watched movie record of the user for each neighbour.
4. Calculate a similarity score using some formula
5. Recommend the items with the highest score
