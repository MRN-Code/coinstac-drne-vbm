#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 19:25:26 2018

@author: Harshvardhan
"""
import os
import warnings

import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statsmodels.api as sm


def parse_for_y(args, y_files, y_labels):
    """Read contents of fsl files into a dataframe"""
    y = pd.DataFrame(index=y_labels)

    for file in y_files:
        if file:
            try:
                y_ = pd.read_csv(os.path.join(args["state"]["baseDirectory"],
                                              file),
                                 sep='\t',
                                 header=None,
                                 names=['Measure:volume', file],
                                 index_col=0)
                y_ = y_[~y_.index.str.contains("Measure:volume")]
                y_ = y_.apply(pd.to_numeric, errors='ignore')
                y = pd.merge(y,
                             y_,
                             how='left',
                             left_index=True,
                             right_index=True)
            except pd.errors.EmptyDataError:
                continue
            except FileNotFoundError:
                continue

    y = y.T

    return y


def fsl_parser(args):
    """Parse the freesurfer (fsl) specific inputspec.json and return the
    covariate matrix (X) as well the dependent matrix (y) as dataframes
    """
    input_list = args["input"]
    X_info = input_list["covariates"]
    y_info = input_list["data"]

    X_data = X_info[0][0]
    X_labels = X_info[1]

    X_df = pd.DataFrame.from_records(X_data)

    X_df.columns = X_df.iloc[0]
    X_df = X_df.reindex(X_df.index.drop(0))
    X_df.set_index(X_df.columns[0], inplace=True)

    X = X_df[X_labels]
    X = X.apply(pd.to_numeric, errors='ignore')
    X = pd.get_dummies(X, drop_first=True)
    X = X * 1

    y_files = y_info[0]
    y_labels = y_info[2]

    y = parse_for_y(args, y_files, y_labels)

    X = X.reindex(sorted(X.columns), axis=1)

    ixs = X.index.intersection(y.index)

    if ixs.empty:
        raise Exception('No common X and y files at ' +
                        args["state"]["clientId"])
    X = X.loc[ixs]
    y = y.loc[ixs]

    return (X, y)


def parse_covar_info(args):
    """Read covariate information from the UI
    """
    input_ = args["input"]
    state_ = args["state"]
    covar_info = input_["covariates"]

    # Reading in the inpuspec.json
    covar_data = covar_info[0][0]
    covar_labels = covar_info[1]
    covar_types = covar_info[2]

    # Converting the contents to a dataframe
    covar_df = pd.DataFrame(covar_data[1:], columns=covar_data[0])
    covar_df.set_index(covar_df.columns[1], inplace=True)

    # Selecting only the columns sepcified in the UI
    # TODO: This could be redundant (check with Ross)
    covar_info = covar_df[covar_labels]

    # convert bool to categorical as soon as possible
    for column in covar_info.select_dtypes(bool):
        covar_info[column] = covar_info[column].astype('object')

    # Checks for existence of files and if they don't delete row
    for file in covar_info.index:
        if not os.path.isfile(os.path.join(state_["baseDirectory"], file)):
            covar_info.drop(file, inplace=True)

    # Raise Exception if none of the files are found
    if covar_info.index.empty:
        raise Exception(
            'Could not find .nii files specified in the covariates csv')

    return covar_info, covar_types


def parse_for_categorical(args):
    """Return unique subsites as a dictionary
    """
    X, _ = parse_covar_info(args)

    site_dict1 = {
        col: list(X[col].unique())
        for col in X.select_dtypes(include=object)
    }

    return site_dict1


def create_dummies(data_f, cols, drop_flag=True):
    """ Create dummy columns
    """
    return pd.get_dummies(data_f, columns=cols, drop_first=drop_flag)


def perform_encoding(args, data_f, exclude_cols=(' ')):
    """Perform encoding of various categorical variables
    """
    cols_categorical = [col for col in data_f if data_f[col].dtype == object]
    cols_mono = [col for col in data_f if data_f[col].nunique() == 1]

    # Dropping columsn with unique values
    data_f = data_f.drop(columns=cols_mono)

    # Creating dummies on non-unique categorical variables
    cols_nodrop = set(cols_categorical) - set(cols_mono)
    data_f = create_dummies(data_f, cols_nodrop, True)

    data_f = data_f.dropna(axis=0, how='any')
    data_f = sm.add_constant(data_f, has_constant='add')

    return data_f
