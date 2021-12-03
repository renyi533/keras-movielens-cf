import math
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, Multiply
from keras.layers import Embedding, Reshape, Dot, Concatenate, Dropout, Dense, merge, dot
from keras.models import Model


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
TRAIN_RATIO = 0.66
ratings = pd.read_csv(RATINGS_CSV_FILE, 
                      sep='\t', 
                      encoding='latin-1', 
                      usecols=['userid', 'movieid', 'user_emb_id', 'movie_emb_id', 'rating'])
max_userid = ratings['userid'].drop_duplicates().max()
max_movieid = ratings['movieid'].drop_duplicates().max()
print (len(ratings), 'ratings loaded.')
shuffled_ratings = ratings.sample(frac=1., random_state=RNG_SEED)
Users = shuffled_ratings['user_emb_id'].values
print ('Users:', Users, ', shape =', Users.shape)
Movies = shuffled_ratings['movie_emb_id'].values
print ('Movies:', Movies, ', shape =', Movies.shape)
Ratings = shuffled_ratings['rating'].values
print ('Ratings:', Ratings, ', shape =', Ratings.shape)


model, emb_model = get_model(max_userid, max_movieid, K_FACTORS)

model.compile(loss='mse', optimizer='adamax')
callbacks = [EarlyStopping('val_loss', patience=2), 
             ModelCheckpoint(MODEL_WEIGHTS_FILE, save_best_only=True)]
history = model.fit([Users, Movies], Ratings, epochs=30, validation_split=.1, verbose=2, callbacks=callbacks)

Users = ratings['user_emb_id'].values
print ('Users:', Users, ', shape =', Users.shape)
Movies = ratings['movie_emb_id'].values
print ('Movies:', Movies, ', shape =', Movies.shape)
Ratings = ratings['rating'].values
print ('Ratings:', Ratings, ', shape =', Ratings.shape)

emb = emb_model.predict([Users, Movies])

print('Embeddings',emb, ', shape =', emb.shape)

curr = 0
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

with open(LIBSVM_FILE+'_train.libsvm', "w") as f:
    for i in range(len(Users)):
        uid = Users[i]
        if uid > max_userid * TRAIN_RATIO:
            break
        writeRecord(f, Users, Ratings, emb, i)
        curr = i

with open(LIBSVM_FILE+'_vali.libsvm', "w") as f:
    for i in range(curr+1, len(Users)):
        uid = Users[i]
        if uid > max_userid * (TRAIN_RATIO + (1.0 - TRAIN_RATIO)/2):
            break
        writeRecord(f, Users, Ratings, emb, i)
        curr = i

with open(LIBSVM_FILE+'_test.libsvm', "w") as f:
    for i in range(curr+1, len(Users)):
        writeRecord(f, Users, Ratings, emb, i)
        curr = i
