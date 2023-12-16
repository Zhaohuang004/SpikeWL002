import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
# from dataclasses import dataclass
from sklearn.model_selection import train_test_split

# @dataclass
# class Params:
#     x: float
#     y: float
#     z: float


def train_test_val_split(x_win_all, y_win_all, d_win_all, split_ratio=0.8):
    # split all data into train and test
    x_win_train, x_win_test, y_win_train, y_win_test, d_win_train, d_win_test = \
        train_test_split(x_win_all, y_win_all, d_win_all, test_size=split_ratio, random_state=0)

    # split train into train and validation with the same ratio
    x_win_train, x_win_val, y_win_train, y_win_val, d_win_train, d_win_val = \
        train_test_split(x_win_train, y_win_train, d_win_train, test_size=split_ratio, random_state=0)

    return x_win_train, x_win_val, x_win_test, \
           y_win_train, y_win_val, y_win_test, \
           d_win_train, d_win_val, d_win_test


def onehot_to_label(y_onehot):
    a = np.argwhere(y_onehot == 1)
    return a[:, -1]

def get_sample_weights(y, weights):
    '''
    to assign weights to each sample
    '''
    label_unique = np.unique(y)
    sample_weights = []
    for val in y:
        idx = np.where(label_unique == val)
        sample_weights.append(weights[idx])
    return sample_weights


def normalize(x):
    """Normalizes all sensor channels by mean substraction,
    dividing by the standard deviation and by 2.

    :param x: numpy integer matrix
        Sensor data
    :return:
        Normalized sensor data
    """
    x = np.array(x, dtype=np.float32)
    m = np.mean(x, axis=0)
    x -= m
    std = np.std(x, axis=0)
    std += 0.000001

    x /= std
    return x

def opp_sliding_window_w_d(data_x, data_y, d, ws, ss):  # window size, step size
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    data_d = np.asarray([[i[-1]] for i in sliding_window(d, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8), data_d.reshape(len(data_d)).astype(np.uint8)


def sliding_window(a, ws, ss=None, flatten=True):
    '''
    Return a sliding window over a in any number of dimensions

    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an
                  extra dimension for each dimension of the input.

    Returns
        an array containing each n-dimensional window from a
    '''

    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)

    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape), len(ws), len(ss)]
    if 1 != len(set(ls)):
        raise ValueError( \
            'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError( \
            'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape), str(ws)))

    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a, shape=newshape, strides=newstrides)
    if not flatten:
        return strided

    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    # commented by hangwei
    # dim = filter(lambda i: i != 1, dim)
    return strided.reshape(dim)

def norm_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.

    Parameters
        shape - an int, or a tuple of ints

    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass

    raise TypeError('shape must be an int, or a tuple of ints')

def opp_sliding_window(data_x, data_y, ws, ss):  # window size, step size
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)


# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 10:37:16 2019

@authors: Ryan Clark, Matt Hong, So Sosaki

File Description:
The utility file used to load and preprocess data, format results, and 
save results.
"""

# Libraries
from time import time
from os.path import exists, join
from os import mkdir
from numpy import mean, std, sum, min, delete
from pandas import read_csv, concat

# Hyper-Pareameters / CONSTANTS
SRC_NULL = 100  # Original Null Value
DST_NULL = -98  # Changed Null Value
MIN_WAPS = 9  # Minimum number of WAPS per sample.


def load_data(train_fname, val_fname, N, drop_columns=None, dst_null=DST_NULL,
              drop_val=False):
    '''
    Loads both the training and validation data (if drop_val is False),
    concatenates the datasets into one dataset. Splits the dataset into data
    and labels (X and Y). Replaces Null values and sets all lower null values
    to the replaced value. Normalizes data between 0 and 1 where 0 is weak
    intensity and 1 is strong intensity.

    Parameters: train_fname  : (str) file name of training data - *.csv
                val_fname    : (str) file name of validation data - *.csv
                N            : (int) number of features
                drop_columns : (list) column names to be removed from data
                dst_null     : (int) the value to change all null values to
                drop_val     : (boolean) if true then drops validation data

    Returns   : x_train      : (Dataframe) training data
                y_train      : (Dataframe) training labels
                x_test       : (Dataframe) test data
                y_test       : (Dataframe) test labels
    '''
    tic = time()  # Start function performance timer

    if drop_val:
        data = read_csv("data/" + train_fname)
    else:
        training_data = read_csv("data/" + train_fname)
        validation_data = read_csv("data/" + val_fname)
        data = concat((training_data, validation_data), ignore_index=True)

    if drop_columns:  # Drop useless columns if there are any specified.
        data.drop(columns=drop_columns, inplace=True)

    data = data[data.PHONEID != 17]  # Phone 17s data is clearly corrupted.
    # Split data from labels
    X = data.iloc[:, :N]
    Y = data.iloc[:, N:]

    # Change null value to new value and set all lower values to it.
    X.replace(SRC_NULL, dst_null, inplace=True)
    X[X < dst_null] = dst_null

    # Remove samples that have less than MIN_WAPS active WAPs
    # Normalize data between 0 and 1 where 1 is strong signal and 0 is null
    X /= min(X)
    X = 1 - X

    toc = time()  # Report function performance timer
    print("Data Load Timer: %.2f seconds" % (toc - tic))

    return X, Y


def filter_out_low_WAPS3D(data, labels1,labels2, num_samples=MIN_WAPS):
    '''
    Removes samples from the data that do not contain at least MIN_WAPS of
    non-null intensities.

    Parameters: data        : (ndarray) 2D array for WAP intensities
                labels      : (ndarray) 2D array for labels
                num_samples : (int) the mim required number of non-null values

    Returns:    new_data    : (ndarray) 2D array for WAP intensities
                new_labels  : (ndarray) 2D array for labels
    '''
    drop_rows = list()
    for i, x in enumerate(data):
        count = sum(x != DST_NULL)
        if count < num_samples:
            drop_rows.append(i)

    new_data = delete(data, drop_rows, axis=0)
    new_labels1 = delete(labels1, drop_rows, axis=0)
    new_labels2 = delete(labels2, drop_rows, axis=0)


    return new_data, new_labels1, new_labels2

def filter_out_low_WAPS2D(data, labels, num_samples=MIN_WAPS):
    '''
    Removes samples from the data that do not contain at least MIN_WAPS of
    non-null intensities.

    Parameters: data        : (ndarray) 2D array for WAP intensities
                labels      : (ndarray) 2D array for labels
                num_samples : (int) the mim required number of non-null values

    Returns:    new_data    : (ndarray) 2D array for WAP intensities
                new_labels  : (ndarray) 2D array for labels
    '''
    drop_rows = list()
    for i, x in enumerate(data):
        count = sum(x != DST_NULL)
        if count < num_samples:
            drop_rows.append(i)

    new_data = delete(data, drop_rows, axis=0)
    new_labels = delete(labels, drop_rows, axis=0)


    return new_data, new_labels

def save_fig(fig, model_name, phone_id, plot_type):
    '''
    This function saves each plot generated by the "phone_id" in the
    output/<model_name>/<plot_type> directory. If the directory doesn't exist,
    then it creates it. WARNING: This function will overwrite previous plots
    that are of the same model name and phone id.

    Parameters: fig           : (Figure)
                model_name    : (str)
                phone_id      : (int)
                plot_type     : (str)

    Returns:    None
    '''
    dir_path = "output"
    if not exists(dir_path):
        mkdir(dir_path)

    dir_path = join(dir_path, model_name)
    if not exists(dir_path):
        mkdir(dir_path)

    dir_path = join(dir_path, plot_type)
    if not exists(dir_path):
        mkdir(dir_path)

    file_name = "PhoneID_%d.png" % phone_id
    file_path = join(dir_path, file_name)
    fig.savefig(file_path, bbox_inches='tight')


def create_subreport(errors, M, phone_id=None):
    '''
    This function takes the set of errors and formats their output into a
    string so that it can be reported to the console, saved to a text file, or
    both.

    Parameters: errors     : (tuple) contains the four types of errors
                M          : (int) number of row elements in set
                phone_id   : (int or None) None implies that its a total report

    Returns:    subreport  : (str)
    '''
    build_missclass, floor_missclass, coords_err, std_err, coor_pr_err = errors

    mean_c = mean(coords_err)
    std_c = std(coords_err)

    build_error = build_missclass / M * 100  # Percent Error
    floor_error = floor_missclass / M * 100  # Percent Error

    if phone_id is not None:
        str1 = "Phone ID: %d" % phone_id
    else:
        str1 = "Totals Output:"
    str2 = "Mean Coordinate Error: %.2f +/- %.2f meters" % (mean_c, std_c)
    str3 = "Standard Error: %.2f meters" % std_err
    str4 = "Building Percent Error: %.2f%%" % build_error
    str5 = "Floor Percent Error: %.2f%%" % floor_error

    if coor_pr_err != "N/A":
        str6 = "Prob that Coordinate Error Less than 10m: %.2f%%" % coor_pr_err
    else:
        str6 = ""

    subreport = '\n'.join([str1, str2, str3, str4, str5, str6])

    return subreport


def save_report(model_name, report, report_type):
    '''
    This function saves the final report for the model in the
    output/<model_name>/ directory. If the directory doesn't exist, then it
    creates it. WARNING: This function will overwrite previous reports that are
    of the same model name.

    Parameters: model_name  : (str)
                report      : (str)
                report_type : (str) Totals or phone_id

    Returns:    None
    '''
    dir_path = "output"
    if not exists(dir_path):
        mkdir(dir_path)

    dir_path = join(dir_path, model_name)
    if not exists(dir_path):
        mkdir(dir_path)

    file_name = "%s_%s.txt" % (model_name, report_type)
    file_path = join(dir_path, file_name)
    with open(file_path, 'w') as text_file:
        text_file.write(report)
    text_file.close()