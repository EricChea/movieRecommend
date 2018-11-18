import os

import pandas as pd
import numpy as np

from utils import data
from models.memory import Memory

DATASOURCE = os.getenv('DATASOURCE', None)


def main():
    """
    Main runner for creating user - movie rankings.
    """

    data_dict = data.read_zipcsv(DATASOURCE)
    ratings = data_dict['ratings']

    # Create a user x movie matrix of ratings; fill with average user ratings 
    # followed by 0 for users who have never rated.
    table = pd.pivot_table(
        ratings,
        values='rating',
        index=['userId'],
        columns=['movieId'],
        aggfunc=np.sum
    ).fillna(ratings.mean(axis=0, skipna=True)).fillna(0)

    # Create movie rankings for each user.
    collab = Memory()
    collab.fit(table, entities=table.index)
    collab.similarity = (collab.similarity * 10000).astype('int16')
    movie_scores = pd.DataFrame(collab.score_items()).T
    movie_scores = movie_scores.apply(lambda x: x/x.sum(), axis=1)
    pd.to_pickle(movie_scores, 'memory.movieranking.pkl')


if __name__ == '__main__':
    import sys
    sys.exit(main())
