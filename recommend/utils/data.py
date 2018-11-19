"""
Convenience functions for loading data.
"""

import os
import zipfile

import numpy as np
import pandas as pd


def read_zipcsv(zippath):
    """
    Loads all csvs in a .zip file into a dictionary as a Pandas DataFrame.

    Parameters
    ----------
    zippath: str, path to zipfile with csv files.
    """

    _zip = zipfile.ZipFile(zippath)

    return {
        os.path.splitext(os.path.basename(_file))[0]: pd.read_csv(_zip.open(_file))
        for _file in _zip.namelist() if is_csv(_file)
    }


def get_features_dependent(data_dict, y='rating'):
    """
    Creates features for users and movies.

    Parameters
    ----------
    data_dict: dict, hash store of data name to it's dataframe.
    y: str, name of the dependent variable.

    Returns
    -------
    tuple, tuple of dataframe that contain (features,
        corresponding dependent variable, user_genre)
    """

    ratings = data_dict['ratings']
    movies = data_dict['movies']
    movies['genres'] = movies.genres.str.split('|')
    movies = explode_field(movies, 'genres')

    movie_genres = get_product_genres(movies, 'movieId', 'genres', 'val')
    movie_genres = movie_genres.fillna(0).drop(columns=['(no genres listed)'])

    user_genre = pd.merge(
        ratings[['userId', 'movieId']],
        movie_genres,
        right_index=True,
        left_on='movieId'
    ).groupby('userId').sum().drop(columns=['movieId'])

    features = pd.merge(
        pd.merge(
            ratings, user_genre, left_on='userId', right_index=True,
            how='left'),
        movie_genres, left_on='movieId', right_index=True, how='left',
        suffixes=('User', 'Movie')
    ).drop(columns=['timestamp'])

    return (
        features[[col + 'User' for col in user_genre.columns if col + 'User' != y]],
        features[y],
        user_genre,
    )


def get_product_genres(data, product_col, cat_col, val_col):
    """
    """

    products = data.copy(deep=True)
    products[val_col] = 1  # Tracks which categories a product belongs to

    return pd.pivot_table(
        products,
        index=[product_col],
        columns=[cat_col],
        values=val_col
    )


def explode_field(data, field):
    """Stacks a cell that represents multiple values (list)

    Parameters
    ----------
    data: pd.DataFrame, data containing the field of interest
    field: str, the column name containing a list.
    """

    kwargs = {col: np.repeat(data[col].values, data[field].str.len())
              for col in data.columns if col != field}
    kwargs.update({field: np.concatenate(data[field].values)})
    return pd.DataFrame(kwargs)


def is_csv(filepath):
    return os.path.splitext(filepath)[-1] == '.csv'
