"""
This module uses memory-based collaborative filter.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Memory(object):
    """
    Creates recommendations based on the type of entity.

    Examples
    --------
    >>> from recommend.models.memory import Memory
    >>> entities = Memory()
    >>> entities.fit(
    >>>     [[1, 0, 1, 0],
    >>>     [0, 1, 0, 1],
    >>>     [1, 1, 1, 1],
    >>>     [1, 0, 0, 1],
    >>>     [1, 0, 0, 1]],
    >>>     entities=['Abe', 'Leo', 'Lil', 'Bob', 'Tom']
    >>> )
    >>> similar_entities, scores = entities.score()
    >>> similar_entities
    [['Leo', 'Bob', 'Tom', 'Lil'],
     ['Abe', 'Bob', 'Tom', 'Lil'],
     ['Abe', 'Leo', 'Bob', 'Tom'],
     ['Abe', 'Leo', 'Lil', 'Bob'],
     ['Abe', 'Leo', 'Lil', 'Bob']]
    >>> scores
    array([[0.        , 0.5       , 0.5       , 0.70710678],
           [0.        , 0.5       , 0.5       , 0.70710678],
           [0.70710678, 0.70710678, 0.70710678, 0.70710678],
           [0.5       , 0.5       , 0.70710678, 1.        ],
           [0.5       , 0.5       , 0.70710678, 1.        ]])
    >>> entities.score_items()
    {'Abe': array([0.54142136, 0.14142136, 0.34142136, 0.34142136]),
     'Bob': array([0.64142136, 0.24142136, 0.24142136, 0.64142136]),
     'Leo': array([0.34142136, 0.34142136, 0.14142136, 0.54142136]),
     'Lil': array([0.62426407, 0.34142136, 0.34142136, 0.62426407]),
     'Tom': array([0.64142136, 0.24142136, 0.24142136, 0.64142136])}
    """

    __similarity = None
    entities = None
    data = None
    metric = None

    @property
    def similarity(self):
        """Public access to __similarity, store of similarity scores.
        """
        return self.__similarity

    @similarity.setter
    def similarity(self, value):
        expected_dims = (len(self.entities), len(self.entities))
        value_dims = np.array(value).shape
        if expected_dims != value_dims:
            raise ValueError(f"Expected {expected_dims} dimensions",
                             f"Received {value_dims}")
        else:
            self.__similarity = value

    def fit(self, data, entities, metric='cosine'):
        """Creates entity groups using unsupervised learning.

        Parameters
        ----------
        data: array-like, features that describe a entity.  For example, a list
            of all items that a entity purchased in the past 2 months.
        entities: list or tuple, entity labels.
        metric: string, defines metric to score similarities. Default='cosine'
            'cosine' = cosine similarity
            'pearson' = pearson correlation

        Return
        ------
        array, n_data x n_data matrix of similarities.
        """

        self.data = data
        self.entities = entities
        self.metric = metric
        if metric == 'cosine':
            self.__similarity = cosine_similarity(data, data)
        elif metric == 'pearson':
            self.__similarity = np.corrcoef(data)
        else:
            print(f"Found metric = {metric}, expected cosine or pearson")

    def score(self, entities=None, n_entities=10):
        """Predicts the most similar entities.

        Parameters
        ----------
        entities: list or tuple or None, list of entities that the data was
        fit to.
        n_entities: int, the top highest scoring entities to return.

        Return
        ------
        entities that are most similar, and entity-entity similarity scores.
        """

        entities = self.entities if not entities else entities
        entity_indx = [list(self.entities).index(ent) for ent in entities]
        n_entities = len(entities) if n_entities >= len(entities) \
            else n_entities

        simentity_indx = np.argsort(
            self.similarity[entity_indx]
        )[:, -n_entities:-1]

        simentitiescore_indx = []
        simentities = []
        for i, row in enumerate(simentity_indx):
            simentitiescore_indx.append((i * n_entities) + row)
            simentities.append([self.entities[ent_ind] for ent_ind in row])

        top_scores = np.take(
            self.__similarity[entity_indx],
            simentitiescore_indx
        )

        return simentities, top_scores

    def score_items(self):
        """Scores each item for an entity
        """
        return {e: (self.similarity[i].reshape(-1, 1) * self.data).mean(axis=0)
                for i, e in enumerate(self.entities)}


def main():
    """
    Main runner for creating user - movie rankings.  Exports a list of movie
    scores for each user as a pickles file.

    Example
    -------
    python -m recommend.models.memory --data D:\data\ml-latest-small.zip
    """

    import argparse

    import pandas as pd

    from ..utils import data

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help="Path to the zip file containing data")
    args = parser.parse_args()

    data_dict = data.read_zipcsv(args.data)
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
