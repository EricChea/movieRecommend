"""
Stock implementation of a neural net.
Thanks to:
https://nipunbatra.github.io/blog/2017/neural-collaborative-filtering.html
"""

import keras
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import merge
from keras.layers import BatchNormalization
from keras.layers import Dense

from ..utils import data

N_LATENT_FACTORS_USER = 8
N_LATENT_FACTORS_MOVIE = 10
N_LATENT_FACTORS_MF = 3


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help="Path to the zip file containing data")
    args = parser.parse_args()

    data_dict = data.read_zipcsv(args.data)
    ratings = data_dict['ratings']

    n_users = len(ratings.userId.unique())
    n_movies = len(ratings.movieId.unique())

    movie_input = Input(shape=[1], name='Item')
    movie_embedding_mlp = Embedding(
        n_movies + 1, N_LATENT_FACTORS_MOVIE, name='Movie-Embedding-MLP'
    )(movie_input)
    movie_vec_mlp = Flatten(name='FlattenMovies-MLP')(movie_embedding_mlp)
    movie_vec_mlp = Dropout(0.2)(movie_vec_mlp)

    movie_embedding_mf = Embedding(
        n_movies + 1, N_LATENT_FACTORS_MF, name='Movie-Embedding-MF'
    )(movie_input)
    movie_vec_mf = Flatten(name='FlattenMovies-MF')(movie_embedding_mf)
    movie_vec_mf = Dropout(0.2)(movie_vec_mf)

    user_input = Input(shape=[1], name='User')
    user_vec_mlp = Flatten(name='FlattenUsers-MLP')(Embedding(
        n_users + 1, N_LATENT_FACTORS_USER, name='User-Embedding-MLP'
    )(user_input))
    user_vec_mlp = Dropout(0.2)(user_vec_mlp)

    user_vec_mf = Flatten(name='FlattenUsers-MF')(Embedding(
        n_users + 1, N_LATENT_FACTORS_MF, name='User-Embedding-MF'
    )(user_input))
    user_vec_mf = Dropout(0.2)(user_vec_mf)

    concat = merge.concatenate(
        [movie_vec_mlp, user_vec_mlp], name='Concat', axis=1)
    concat_dropout = Dropout(0.2)(concat)
    dense = Dense(200, name='FullyConnected')(concat_dropout)
    dense_batch = BatchNormalization(name='Batch')(dense)
    dropout_1 = Dropout(0.2, name='Dropout-1')(dense_batch)
    dense_2 = Dense(100, name='FullyConnected-1')(dropout_1)
    dense_batch_2 = BatchNormalization(name='Batch-2')(dense_2)

    dropout_2 = Dropout(0.2, name='Dropout-2')(dense_batch_2)
    dense_3 = Dense(50, name='FullyConnected-2')(dropout_2)
    dense_4 = Dense(20, name='FullyConnected-3', activation='relu')(dense_3)

    pred_mf = merge.dot(
        [movie_vec_mf, user_vec_mf], name='Dot', axes=1)
    pred_mlp = Dense(1, activation='relu', name='Activation')(dense_4)

    combine_mlp_mf = merge.concatenate(
        [pred_mf, pred_mlp], name='Concat-MF-MLP')
    result_combine = Dense(100, name='Combine-MF-MLP')(combine_mlp_mf)
    deep_combine = Dense(100, name='FullyConnected-4')(result_combine)

    result = keras.layers.Dense(1, name='Prediction')(deep_combine)

    model = keras.Model([user_input, movie_input], result)
    model.compile(optimizer='adam', loss='mean_absolute_error')
    model.save('nn.movieranking.h5')


if __name__ == '__main__':
    import sys
    sys.exit(main())
