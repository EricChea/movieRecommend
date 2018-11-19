Introduction
------------
Two algorithms are implemented, user-user collaborative filtering and a regression model. The former provides a ranking of movies for a user based on the ratings that similar users have given movies.  The latter imputes a users preference for a genre, and the genres a movie falls under in order to predict what a user would rate a movie.

### Installation
#### Clone the repository
``git clone git@github.com:EricChea/movieRecommend.git``

#### Optional(recommended)
Create a virtual environment before installing the python packages
```
conda create -n recommend
conda activate recommend
```

#### Install dependencies
``pip install -r requirements.txt``


Approach
--------
### Data
The MovieLens [ml-latest.zip](http://files.grouplens.org/datasets/movielens/ml-latest-small.zip) data was used for this exercise.

### Models
Two models are available:
1. ``recommend.models.memory.Memory``: This is a user-user collaboartaive model.
2. ``recommend.models.trees``: implements the lightgbm model to predict what a user
would rank a movie.

Usage
-----
#### Get user-movie rankings using user-user Collaborative Filtering
From the top level directory run (from the top level directory):
``python -m recommend.models.memory --data=<PATH TO MOVIELENS DATA>``
After this is complete, a file named *memory.movieranking.pkl* will output.
This file can be reloaded using

```
import pandas as pd  # Can be pickles as well.
user_movie_rankings = pd.read_pickle('memory.movieranking.pkl')
```

#### Get a users predicted rating for a movie.
To train a regression model run (from the top level directory):
``python -m recommend.models.trees --data=<PATH TO MOVIELENS DATA>``

After the process is complete a model will be saved called *trees.predictrating.lgb*.
This model can be reloaded

```
import lightgbm as lgb
lgb.Booster(model_file='trees.predictrating.lgb')

# X needs to have the dimensions (, 38).
predicted_scores = lgb.predict(X)
```

**Note**: Feature extraction can be found under ``recommend.utils.data.get_features_dependent``
