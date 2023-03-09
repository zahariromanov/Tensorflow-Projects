import pandas as pd
from surprise import Reader

book_ratings = pd.read_csv(r'Tensorflow-Projects\Recommender Systems\book_recommender\ratedata.csv')
print(book_ratings.head())

#1. Print dataset size and examine column data types
print(len(book_ratings))
print(book_ratings.info())

#2. Distribution of ratings
print(book_ratings['rating'].value_counts())

#3. Filter ratings that are out of range
book_ratings = book_ratings[book_ratings['rating']!=0]

#4. Prepare data for surprise: build a Suprise reader object
from surprise import Reader
reader = Reader(rating_scale=(1, 5))

#5. Load `book_ratings` into a Surprise Dataset
from surprise import Dataset
rec_data = Dataset.load_from_df(book_ratings[['user_id',
                                              'book_id',
                                              'rating']],
                                reader)

#6. Create a 80:20 train-test split and set the random state to 7
from surprise.model_selection import train_test_split
trainset, testset = train_test_split(rec_data, test_size=.2, random_state=7)

#7. Use KNNBasice from Surprise to train a collaborative filter
from surprise import KNNBasic
nn_algo = KNNBasic()
nn_algo.fit(trainset)

#8. Evaluate the recommender system
from surprise import accuracy
predictions = nn_algo.test(testset)
accuracy.rmse(predictions)

#9. Prediction on a user who gave the "The Three-Body Problem" a rating of 5
print(nn_algo.predict('8842281e1d1347389f2ab93d60773d4d', '18007564').est)


# ↓↓↓ !!! Results Avalable Below !!! ↓↓↓

'''

PS D:\Repos> & d:/Repos/.venv/Scripts/Activate.ps1
(.venv) PS D:\Repos> & d:/Repos/.venv/Scripts/python.exe "d:/Repos/Tensorflow-Projects/Reccomender Engines/book_reccomender/reccomender.py"
                            user_id   book_id                         review_id  rating  ...                         read_at                      started_at n_votes n_comments
0  d089c9b670c0b0b339353aebbace46a1   7686667  3337e0e75701f7f682de11638ccdc60c       3  ...  Sat Jun 18 00:00:00 -0700 2011  Thu May 19 00:00:00 -0700 2011       0          0
1  6dcb2c16e12a41ae0c6c38e9d46f3292  18073066  7201aa3c1161f2bad81258b6d4686c16       5  ...  Mon Aug 19 00:00:00 -0700 2013  Mon Aug 12 00:00:00 -0700 2013      16         14
2  244e0ce681148a7586d7746676093ce9  13610986  07a203f87bfe1b65ff58774667f6f80d       5  ...  Fri Dec 19 00:00:00 -0800 2014  Sun Nov 23 00:00:00 -0800 2014       0          0
3  73fcc25ff29f8b73b3a7578aec846394  27274343  8be2d87b07098c16f9742020ec459383       1  ...  Wed Apr 26 16:06:28 -0700 2017  Sun Apr 23 00:00:00 -0700 2017       0          1
4  f8880e158a163388a990b64fec7df300  11614718  a29c4ba03e33ad073a414ac775266c5f       4  ...  Sun Apr 20 09:26:41 -0700 2014  Fri Apr 18 00:00:00 -0700 2014       0          0

[5 rows x 11 columns]
3500
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3500 entries, 0 to 3499
Data columns (total 11 columns):
 #   Column        Non-Null Count  Dtype 
---  ------        --------------  ----- 
 0   user_id       3500 non-null   object
 1   book_id       3500 non-null   int64 
 2   review_id     3500 non-null   object
 3   rating        3500 non-null   int64 
 4   review_text   3500 non-null   object
 5   date_added    3500 non-null   object
 6   date_updated  3500 non-null   object
 7   read_at       3167 non-null   object
 8   started_at    2395 non-null   object
 9   n_votes       3500 non-null   int64 
 10  n_comments    3500 non-null   int64 
dtypes: int64(4), object(7)
memory usage: 300.9+ KB
None
4    1278
5    1001
3     707
2     269
1     125
0     120
Name: rating, dtype: int64
Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 1.1105
3.8250739644970415
(.venv) PS D:\Repos> 

'''