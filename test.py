import numpy as np
import pandas as pd

from baseline import cat2num
import analysis
import data_exploration

"""
use with command : nosetests --with-coverage --cover-package=. test.py
"""


def function_cat2num(input, expected):
    output = cat2num(input)
    assert output == expected, "Expected %r, but got %r" % (expected, cer)


def test_cat2num_1():
    function_cat2num("bike", 0)


def test_cat2num_2():
    function_cat2num("other", 3)


def test_analysis_create_data_sets_1():

    # create a dataFrame and check the data split
    data = {
        "data1": [20, 21, 19, 18],
        "data2": [20, 21, 19, 18],
        "imageSet": ["train", "train", "test", "valid"],
        "category": ["bike", "car", "van", "truck"],
        "imageFileName": ["file1", "file2", "file3", "file4"],
        "category_num": [0, 1, 2, 3],
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    X_tr, y_tr, X_te, y_te, X_va, y_va = analysis.create_data_sets(df)

    assert len(y_tr) == 2, "Expected %r, but got %r" % (2, len(y_tr))
    assert len(y_te) == 1, "Expected %r, but got %r" % (1, len(y_te))
    assert len(y_va) == 1, "Expected %r, but got %r" % (1, len(y_va))
    assert X_tr.shape == (2, 3), "Expected %r, but got %r" % ((2, 3), X_tr.shape)


def test_get_batches_1():
    X_tr = np.random.rand(100, 2)
    y_tr = np.random.rand(100)
    i = 0
    batch_size_ok = False
    for X_batch, y_batch in analysis.get_batches(X_tr, y_tr, 25):
        i += 1
        if len(X_batch) == len(y_batch) and len(y_batch) == 25:
            batch_size_ok = True
        else:
            batch_size_ok = False

    # test if the batch amount is ok given the inputs and batch_size
    assert i == 4, "Expected %r, but got %r" % (4, i)
    assert batch_size_ok == True, "Expected %r, but got %r" % (True, False)


def test_get_confusion_matrix_1():
    y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
    y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
    labels = ["ant", "bird", "cat"]
    df = analysis.get_confusion_matrix(y_true, y_pred, labels, labels)
    expected_res = np.array([[2, 0, 0], [0, 0, 1], [1, 0, 2]])

    assert (df.values == expected_res).all(), "Expected %r, but got %r" % (
        res,
        df.values,
    )

