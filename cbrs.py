import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfTransformer

# Reading user file
u_cols = ['user_id', 'age ', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')

n_users = users.shape[0]

# Reading rating file
r_cols = ['users_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')

rate_train = ratings_base.to_numpy()
rate_test = ratings_test.to_numpy()

# Reading items from file:
i_cols = ['movie_id', 'movie_title', 'release_date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children\s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Flim-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')
n_items = items.shape[0]

X0 = items.to_numpy()
X_train_counts = X0[:, -19:]
print(X_train_counts)

transformer = TfidfTransformer(smooth_idf=True, norm='l2')
tfidf = transformer.fit_transform(X_train_counts.tolist()).toarray()

def get_items_rated_by_user(rate_matrix, user_id):
    y = rate_matrix[:,0]
    ids = np.where(y==user_id +1)[0]
    item_ids = rate_matrix[ids, 1] -1 
    scores = rate_matrix[ids,2]
    return item_ids, scores

# finding model for each user
d = tfidf.shape[1] # data dimension
W = np.zeros((d, n_users))
b = np.zeros((1, n_users))

# Apply ridge regression in sklearn lib for each user
for n in range(n_users):
    ids, scores = get_items_rated_by_user(rate_train, n)
    clf = Ridge(alpha=0.01, fit_intercept=True)
    Xhat = tfidf[ids, :]

    clf.fit(Xhat, scores)
    W[:, n] = clf.coef_
    b[0, n] = clf.intercept_

# Predicted scores
Yhat = tfidf.dot(W) + b

# test with user id = 10
n = 10
np.set_printoptions(precision=2)
ids, scores = get_items_rated_by_user(rate_test, n)
Yhat[n, ids]
print("Rated movies ids: ", ids)
print("True ratings: ", scores)
print("Predicted ratings: ", Yhat[ids, n])

def evaluate(Yhat, rates, W, b):
    se = 0
    cnt = 0
    for n in range(n_users):
        ids, scores_truth = get_items_rated_by_user(rates, n)
        scores_pred = Yhat[ids, n]
        e = scores_truth - scores_pred
        se += (e*e).sum(axis=0)
        cnt += e.size
    return np.sqrt(se/cnt)

print("RMSE for training:", evaluate(Yhat, rate_train, W, b))
print("RMSE for test:", evaluate(Yhat, rate_test, W, b))

