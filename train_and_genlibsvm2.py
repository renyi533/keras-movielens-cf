import math
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, Multiply
from keras.layers import Embedding, Reshape, Dot, Concatenate, Dropout, Dense, merge, dot, Flatten
from keras.models import Model
import numpy as np
from keras import backend as K
from keras.optimizers import adam_v2

RATINGS_CSV_FILE = 'ml1m_ratings.csv'
MODEL_WEIGHTS_FILE = 'ml1m_weights.h5'
LIBSVM_FILE= 'ml1m'
K_FACTORS = 60
RNG_SEED = 1446557
use_bpr = False

def get_mf_model(n_users, n_items, dim):
    user_id = Input(shape=(1,))
    item_id = Input(shape=(1,))
    user_emb = Embedding(n_users, dim, input_length=1)(user_id)
    item_emb = Embedding(n_items, dim, input_length=1)(item_id)
    pred = Dot(axes=-1)([user_emb, item_emb])
    dot_emb = Multiply()([user_emb, item_emb])
    emb = Concatenate(axis=-1)([user_emb, item_emb, dot_emb])
    train_model = Model([user_id,item_id],pred)
    train_model.compile(loss='mse', optimizer='adamax')
    return train_model, Model([user_id,item_id],emb)

def identity_loss(y_true, y_pred):
    return K.mean(y_pred)

def bpr_triplet_loss(X):
    positive_item_latent, negative_item_latent, user_latent = X

    # BPR loss
    loss = 1.0 - K.sigmoid(
        K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True) -
        K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True))

    return loss

def get_bpr_model(n_users, n_items, dim):
    user_id = Input(shape=(1,))
    item_id = Input(shape=(1,))
    neg_item_id = Input(shape=(1,))
    user_emb = Embedding(n_users, dim, input_length=1)(user_id)
    item_emb_layer = Embedding(n_items, dim, input_length=1)
    item_emb = item_emb_layer(item_id)
    neg_item_emb = item_emb_layer(neg_item_id)    
    loss = bpr_triplet_loss((Flatten()(item_emb), Flatten()(neg_item_emb), Flatten()(user_emb)))

    dot_emb = Multiply()([user_emb, item_emb])
    emb = Concatenate(axis=-1)([user_emb, item_emb, dot_emb])
    train_model = Model([user_id,item_id,neg_item_id],loss)
    train_model.compile(loss=identity_loss, optimizer='adamax')
    return train_model, Model([user_id,item_id],emb)

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
#shuffled_ratings = train_part.sample(frac=1., random_state=RNG_SEED)
Users = train_part['user_emb_id'].values
print ('Train Users:', Users, ', shape =', Users.shape)
Movies = train_part['movie_emb_id'].values
print ('Train Movies:', Movies, ', shape =', Movies.shape)
Ratings = train_part['rating'].values
print ('Train Ratings:', Ratings, ', shape =', Ratings.shape)

train_df = train_part
train_df = train_df.loc[train_df['rating'] > 2]
PosUsers = train_df['user_emb_id'].values
print ('Train PosUsers:', PosUsers, ', shape =', PosUsers.shape)
PosMovies = train_df['movie_emb_id'].values
print ('Train PosMovies:', PosMovies, ', shape =', PosMovies.shape)
NegMovies = np.random.choice(max_movieid, len(PosMovies))
print ('Train NegMovies:', NegMovies, ', shape =', NegMovies.shape)
PosRatings = train_df['rating'].values
print ('Train PosRatings:', PosRatings, ', shape =', PosRatings.shape)

ValiUsers = vali_part['user_emb_id'].values
print ('vali Users:', Users, ', shape =', ValiUsers.shape)
ValiMovies = vali_part['movie_emb_id'].values
print ('vali Movies:', Movies, ', shape =', ValiMovies.shape)
ValiRatings = vali_part['rating'].values
print ('vali Ratings:', Ratings, ', shape =', ValiRatings.shape)

vali_df = vali_part
vali_df = vali_df.loc[vali_df['rating'] > 2]
ValiPosUsers = vali_df['user_emb_id'].values
print ('Vali PosUsers:', ValiPosUsers, ', shape =', ValiPosUsers.shape)
ValiPosMovies = vali_df['movie_emb_id'].values
print ('Vali PosMovies:', ValiPosMovies, ', shape =', ValiPosMovies.shape)
ValiNegMovies = np.random.choice(max_movieid, len(ValiPosMovies))
print ('Vali NegMovies:', ValiNegMovies, ', shape =', ValiNegMovies.shape)
ValiPosRatings = vali_df['rating'].values
print ('Vali PosRatings:', ValiPosRatings, ', shape =', ValiPosRatings.shape)

TestUsers = test_part['user_emb_id'].values
print ('test Users:', Users, ', shape =', TestUsers.shape)
TestMovies = test_part['movie_emb_id'].values
print ('test Movies:', Movies, ', shape =', TestMovies.shape)
TestRatings = test_part['rating'].values
print ('test Ratings:', Ratings, ', shape =', TestRatings.shape)

callbacks = [EarlyStopping('val_loss', patience=2), 
             ModelCheckpoint(MODEL_WEIGHTS_FILE, save_best_only=True)]

if use_bpr:
    model, emb_model = get_bpr_model(max_userid, max_movieid, K_FACTORS)
    history = model.fit([PosUsers, PosMovies, NegMovies], PosRatings, epochs=30, 
                    validation_data=([ValiPosUsers,ValiPosMovies,ValiNegMovies], ValiPosRatings),
                    validation_split=0, verbose=2, callbacks=callbacks, shuffle=True)
else:
    model, emb_model = get_mf_model(max_userid, max_movieid, K_FACTORS)
    history = model.fit([Users, Movies], Ratings, epochs=30, 
                    validation_data=([ValiUsers,ValiMovies], ValiRatings),
                    validation_split=0, verbose=2, callbacks=callbacks, shuffle=True)

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