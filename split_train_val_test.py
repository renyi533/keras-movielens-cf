import math
import pandas as pd
import numpy as np

RATINGS_CSV_FILE = 'ml1m_ratings.csv'
MODEL_WEIGHTS_FILE = 'ml1m_weights.h5'
LIBSVM_FILE= 'ml1m'
K_FACTORS = 60
RNG_SEED = 1446557

ratings = pd.read_csv(RATINGS_CSV_FILE, 
                      sep='\t', 
                      encoding='latin-1', 
                      usecols=['userid', 'movieid', 'rating', 'timestamp'])
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

train_part.to_csv('u1.base', sep='\t', header=False, index=False)
vali_part.to_csv('u1.val', sep='\t', header=False, index=False)
test_part.to_csv('u1.test', sep='\t', header=False, index=False)