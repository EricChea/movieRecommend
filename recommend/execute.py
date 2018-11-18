from utils import data


def main():
    data_dict = data.read_zipcsv(
        'D:\\data\\ml-latest-small.zip', compression='zip'
    )
    # X_train = collab.CollabDataBunch.from_df(data_dict['ratings'])

    # learn = collab.collab_learner(X_train, n_factors=50, y_range=(0., 5.))
    # learn.fit_one_cycle(5, 5e-3, wd=0.1)
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    import sys
    sys.exit(main())
