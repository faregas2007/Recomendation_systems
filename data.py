import pandas as pd
import numpy as np

# reading user file
u_cols = ['users_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols, encoding='latin=1')

n_users = users.shape[0]
print('Number of users:', n_users)
# users.head() # uncomment this to see some few examples

# reading rating file
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')

rate_train = ratings_base.to_numpy()
rate_test = ratings_test.to_numpy()

print("Number of training rates: %s"% rate_train.shape[0])
print("Nubmer of test rates: %s"% rate_test.shape[0])

# item profiles
# reading item file
i_cols = ['movie_id', 'movie title', 'release date', 'video release date', 'IMDB URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')

n_items = items.shape[0]
print('Number of items:', n_items)

# focus only on 19 binary values at the end of each rows
X0 = items.to_numpy()
X_train_counts = X0[:,-19]

# Construct the feature vector on each item based on matrix of
# geners and feature TF-IDF.

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=True, norm='l2')
tfidf = transformer.fit_transform(X_train_counts.tolist()).toarray()

# tfidf corresponds to feature vector of each movie

def get_items_rated_by_user(rate_matrix, user_id):
    """
    in each line of rate_matrix, we have infor: user_id, item_id, rating(scores),
    time_stamp, we care about the first three values
    return (item_ids, scores) rated by user user_id
    """

    y = rate_matrix[:, 0] # all users
    # item indices rated by user_id
    # we need to +1 to user_id since in the rate_matrix, id start from 1
    # while index in python starts from 0.
    ids = np.where(y == user_id+1)[0]
    item_ids = rate_matrix[ids, 1] - 1 # index starts from 0
    scores = rate_matrix[ids, 2]
    return (item_ids, scores)
    
