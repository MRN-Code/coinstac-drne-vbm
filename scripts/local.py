#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the local computations for decentralized regression
(normal equation) including decentralized statistic calculation
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import ujson as json

from local_ancillary import (add_site_covariates, local_stats_to_dict_numba,
                             mean_and_len_y, multiply, stats_calculation,
                             vbm_parser)
from nipype_utils import average_nifti
from parsers import parse_for_categorical
from rw_utils import write_file
from utils import list_recursive

warnings.simplefilter("ignore")


def local_0(args):
    """ The first function in the local computation chain
    """
    input_ = args["input"]
    state_ = args["state"]
    cache_dir = state_["cacheDirectory"]

    threshold = input_["threshold"]
    voxel_size = input_["voxel_size"]
    lamb = args['input']['lambda']

    categorical_dict = parse_for_categorical(args)
    covar_x = average_nifti(args)

    write_file(args, args, 'cache', 'args_file')
    covar_x.to_csv(os.path.join(cache_dir, 'X_df'))

    output_dict = {
        "categorical_dict": categorical_dict,
        "threshold": threshold,
        "voxel_size": voxel_size,
        "avg_nifti": "avg_nifti.nii",
        "computation_phase": "local_0"
    }
    cache_dict = {
        "covariates": 'X_df',
        "voxel_size": voxel_size,
        "lambda": lamb
    }

    computation_output_dict = {"output": output_dict, "cache": cache_dict}

    return json.dumps(computation_output_dict)


def local_1(args):
    """ The second function in the local computation chain
    """
    cache_ = args["cache"]
    state_ = args["state"]
    output_dir = state_["transferDirectory"]
    cache_dir = state_["cacheDirectory"]

    X = pd.read_csv(os.path.join(cache_dir, cache_['covariates']), index_col=0)
    regularizer_l2 = cache_['lambda']

    # Local Statistics
    encoded_X, y = vbm_parser(args, X)
    meanY_vector, lenY_vector = mean_and_len_y(y)
    _, local_stats_list = local_stats_to_dict_numba(args, encoded_X, y)

    # Global Statistics
    augmented_X = add_site_covariates(args, X)
    X_labels = list(augmented_X.columns)
    biased_X = augmented_X.values.astype('float64')

    XtransposeX_local = multiply(biased_X, biased_X)
    Xtransposey_local = multiply(biased_X, y)

    # Writing covariates and dependents to cache as files
    np.save(os.path.join(cache_dir, 'X.npy'), biased_X)
    np.save(os.path.join(cache_dir, 'y.npy'), y)

    # Writing XTX and XTy to output as files
    np.save(os.path.join(output_dir, 'XTX.npy'), XtransposeX_local)
    np.save(os.path.join(output_dir, 'XTy.npy'), Xtransposey_local)

    output_dict = {
        "XtransposeX_local": 'XTX.npy',
        "Xtransposey_local": 'XTy.npy',
        "mean_y_local": meanY_vector.tolist(),
        "count_local": lenY_vector.tolist(),
        "local_stats_list": local_stats_list,
        "X_labels": X_labels,
        "lambda": regularizer_l2
    }
    cache_dict = {"covariates": "X.npy", "dependents": "y.npy"}

    write_file(args, output_dict, 'output', 'local_output')

    computation_output_dict = {
        "output": {
            "computation_phase": "local_1"
        },
        "cache": cache_dict,
    }

    return json.dumps(computation_output_dict)


def local_2(args):
    """Computes the SSE_local, SST_local and varX_matrix_local
    Args:
        args (dictionary): {"input": {
                                "avg_beta_vector": ,
                                "mean_y_global": ,
                                "computation_phase":
                                },
                            "cache": {
                                "covariates": ,
                                "dependents": ,
                                "lambda": ,
                                "dof_local": ,
                                }
                            }
    Returns:
        computation_output (json): {"output": {
                                        "SSE_local": ,
                                        "SST_local": ,
                                        "varX_matrix_local": ,
                                        "computation_phase":
                                        }
                                    }
    Comments:
        After receiving  the mean_y_global, calculate the SSE_local,
        SST_local and varX_matrix_local
    """
    input_ = args["input"]
    cache_ = args["cache"]
    state_ = args["state"]
    cache_dir = state_["cacheDirectory"]

    biased_X = np.load(os.path.join(cache_dir, cache_["covariates"]))
    y = np.load(os.path.join(cache_dir, cache_["dependents"]))

    avg_beta_vector = input_["avg_beta_vector"]
    mean_y_global = input_["mean_y_global"]

    varX_matrix_local = multiply(biased_X, biased_X)

    sse_local, sst_local = stats_calculation(biased_X, y,
                                             np.array(avg_beta_vector),
                                             np.array(mean_y_global))

    output_dict = {
        "SSE_local": sse_local.tolist(),
        "SST_local": sst_local.tolist(),
        "varX_matrix_local": varX_matrix_local.tolist()
    }

    write_file(args, output_dict, 'output', 'local_output')

    output_dict = {"computation_phase": 'local_2'}
    cache_dict = {}
    computation_output_dict = {"output": output_dict, "cache": cache_dict}

    return json.dumps(computation_output_dict)


if __name__ == '__main__':

    PARAM_DICT = json.loads(sys.stdin.read())
    PHASE_KEY = list(list_recursive(PARAM_DICT, 'computation_phase'))

    if not PHASE_KEY:
        sys.stdout.write(local_0(PARAM_DICT))
    elif "remote_0" in PHASE_KEY:
        sys.stdout.write(local_1(PARAM_DICT))
    elif "remote_1" in PHASE_KEY:
        sys.stdout.write(local_2(PARAM_DICT))
    else:
        raise ValueError("Error occurred at Local")
