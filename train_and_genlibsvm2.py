import math
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, Multiply
from keras.layers import Embedding, Reshape, Dot, Concatenate, Dropout, Dense, merge, dot
from keras.models import Model
import numpy as np

def get_model(n_users, n_items, dim):
    user_id = Input(shape=(1,))
    item_id = Input(shape=(1,))
    user_emb = Embedding(n_users, dim, input_length=1)(user_id)
    item_emb = Embedding(n_items, dim, input_length=1)(item_id)
    pred = Dot(axes=-1)([user_emb, item_emb])
    dot_emb = Multiply()([user_emb, item_emb])
    emb = Concatenate(axis=-1)([user_emb, item_emb, dot_emb])
    return Model([user_id,item_id],pred), Model([user_id,item_id],emb)

RATINGS_CSV_FILE = 'ml1m_ratings.csv'
MODEL_WEIGHTS_FILE = 'ml1m_weights.h5'
LIBSVM_FILE= 'ml1m'
K_FACTORS = 60
RNG_SEED = 1446557
ratings = pd.read_csv(RATINGS_CSV_FILE, 
                      sep='\t', 
                      encoding='latin-1', 
                      usecols=['userid', 'movieid', 'user_emb_id', 'movie_emb_id', 'rating', 'timestamp'])
#ratings = ratings.sort_values(by = ['userid', 'timestamp'], ascending=True)
user_count = ratings.groupby(['userid']).count()
max_userid = ratings['userid'].drop_duplicates().max()
max_movieid = ratings['movieid'].drop_duplicates().max()

train_df = []
vali_df = []
test_df = []
for i in range(max_userid):
    uid = i+1
    df_slice = ratings.loc[ratings['userid'] == uid].sort_values(by=['timestamp'], ascending=True)
    cnt = df_slice.count()['userid']
    if cnt < 40:
        train_df.append(df_slice)
    else:
        slice1, slice2, slice3, slice4 = np.array_split(df_slice, 4)
        train_df.extend([slice1, slice2])
        vali_df.append(slice3)
        test_df.append(slice4)

train_part = pd.concat(train_df)
vali_part = pd.concat(vali_df)
test_part = pd.concat(test_df)

print (len(ratings), 'ratings loaded.')
shuffled_ratings = train_part.sample(frac=1., random_state=RNG_SEED)
Users = shuffled_ratings['user_emb_id'].values
print ('Train Users:', Users, ', shape =', Users.shape)
Movies = shuffled_ratings['movie_emb_id'].values
print ('Train Movies:', Movies, ', shape =', Movies.shape)
Ratings = shuffled_ratings['rating'].values
print ('Train Ratings:', Ratings, ', shape =', Ratings.shape)

ValiUsers = vali_part['user_emb_id'].values
print ('vali Users:', Users, ', shape =', ValiUsers.shape)
ValiMovies = vali_part['movie_emb_id'].values
print ('vali Movies:', Movies, ', shape =', ValiMovies.shape)
ValiRatings = vali_part['rating'].values
print ('vali Ratings:', Ratings, ', shape =', ValiRatings.shape)

TestUsers = test_part['user_emb_id'].values
print ('test Users:', Users, ', shape =', TestUsers.shape)
TestMovies = test_part['movie_emb_id'].values
print ('test Movies:', Movies, ', shape =', TestMovies.shape)
TestRatings = test_part['rating'].values
print ('test Ratings:', Ratings, ', shape =', TestRatings.shape)

model, emb_model = get_model(max_userid, max_movieid, K_FACTORS)

model.compile(loss='mse', optimizer='adamax')
callbacks = [EarlyStopping('val_loss', patience=2), 
             ModelCheckpoint(MODEL_WEIGHTS_FILE, save_best_only=True)]
history = model.fit([Users, Movies], Ratings, epochs=30, 
                    validation_data=([ValiUsers,ValiMovies], ValiRatings),
                    validation_split=0, verbose=2, callbacks=callbacks)

Users = train_part['user_emb_id'].values
print ('Users:', Users, ', shape =', Users.shape)
Movies = train_part['movie_emb_id'].values
print ('Movies:', Movies, ', shape =', Movies.shape)
Ratings = train_part['rating'].values
print ('Ratings:', Ratings, ', shape =', Ratings.shape)

def writeRecord(f, Users, Ratings, emb, i):
    uid = str(Users[i])
    rating = str(Ratings[i] - 1)
    feature = emb[i][0]
    line = str(rating) + " " + "qid:" + uid
    for j in range(len(feature)):
        elem = str(j+1) + ":" + str(feature[j])
        line = line + " " + elem
    f.write(line)
    f.write('\n')

def writeFile(name, Users, Ratings, emb):
  with open(name, "w") as f:
    for i in range(len(Users)):
        uid = Users[i]
        writeRecord(f, Users, Ratings, emb, i)

emb = emb_model.predict([Users, Movies])
print('Train Embeddings',emb, ', shape =', emb.shape)
writeFile(LIBSVM_FILE+'_train.libsvm', Users, Ratings, emb)

vali_emb = emb_model.predict([ValiUsers, ValiMovies])
print('Vali Embeddings',vali_emb, ', shape =', vali_emb.shape)
writeFile(LIBSVM_FILE+'_vali.libsvm', ValiUsers, ValiRatings, vali_emb)

test_emb = emb_model.predict([TestUsers, TestMovies])
print('Test Embeddings',test_emb, ', shape =', test_emb.shape)
writeFile(LIBSVM_FILE+'_test.libsvm', TestUsers, TestRatings, test_emb)