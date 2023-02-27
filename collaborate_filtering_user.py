import os
import pickle

import numpy as np
import pandas as pd
from sortedcontainers import SortedList
from multiprocessing import Pool

# Load preprocessed densed matrix
with open('data/user_to_movie.json', 'rb') as f:
    user_to_movie = pickle.load(f)

with open('data/movie_to_user.json', 'rb') as f:
    movie_to_user = pickle.load(f)

with open('data/userMovie_to_rating.json', 'rb') as f:
    userMovie_to_rating = pickle.load(f)

with open('data/userMovie_to_rating_test.json', 'rb') as f:
    userMovie_to_rating_test = pickle.load(f)
    
# retrieve the dimension of the sparse matrix
N = int(np.max(list(user_to_movie.keys()))) + 1
m1 = np.max(list(movie_to_user.keys()))
m2 = np.max([m for (u, m), r in userMovie_to_rating_test.items()])
M = int(max(m1, m2)) + 1


K = 25
limit = 5

def compute_stats(i):
    # Retrieve movies for user i
    movies_i = user_to_movie[i]
    movies_i_set = set(movies_i)

    # Compute averages and deviations for users
    ratings_i = {movie:userMovie_to_rating[(i, movie)] for movie in movies_i}
    avg_i = np.mean(list(ratings_i.values()))
    dev_i = {movie:(rating - avg_i) for movie, rating in ratings_i.items()}
    dev_i_values = np.array(list(dev_i.values()))
    sigma_i = np.sqrt(np.sum(np.square(dev_i_values)))
    
    neighbors_i = SortedList()
    for j in range(N):
        
        if j != i:
            
            # Retrieve movies for user j
            movies_j = user_to_movie[j]
            movies_j_set = set(movies_j)
            
            # Compute w_ij if user j is a qualified neighbor
            common_movies = (movies_i_set & movies_j_set)
            if len(common_movies) > limit:
                # calculate avg and deviation
                ratings_j = {movie:userMovie_to_rating[(j, movie)] for movie in movies_j}
                avg_j = np.mean(list(ratings_j.values()))
                dev_j = {movie:(rating - avg_j) for movie, rating in ratings_j.items()}
                dev_j_values = np.array(list(dev_j.values()))
                sigma_j = np.sqrt(np.sum(np.square(dev_j_values)))

                # calculate correlation coefficient
                numerator = sum(dev_i[m]*dev_j[m] for m in common_movies)
                w_ij = numerator / (sigma_i * sigma_j)
                
                # negate weight, because list is sorted ascending
                # maximum value (1) is "closest"
                neighbors_i.add((-w_ij, j))
                if len(neighbors_i) > K:
                    del neighbors_i[-1]
    if i % 100 == 0:
        print(f"Processing {i}")
    
    return avg_i, dev_i, neighbors_i







