"""
Convenience functions for loading data.
"""

import os
import zipfile
import pandas as pd


def is_csv(filepath):
    return os.path.splitext(filepath)[-1] == '.csv'


def read_zipcsv(zippath, compression):
    """
    Loads all csvs in a .zip file into a dictionary as a Pandas DataFrame.
    """

    _zip = zipfile.ZipFile(zippath)

    return {
        os.path.splitext(os.path.basename(_file))[0]: pd.read_csv(_zip.open(_file))
        for _file in _zip.namelist() if is_csv(_file)
    }
