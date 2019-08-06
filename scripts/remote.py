#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the remote computations for decentralized
regression with decentralized statistic calculation
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import ujson as json
from scipy import stats

from ancillary import encode_png, print_beta_images, print_pvals
from nipype_utils import calculate_mask
from remote_ancillary import remote_stats, return_uniques_and_counts
from rw_utils import read_file
from utils import list_recursive

warnings.simplefilter("ignore")
OUTPUT_FROM_LOCAL = 'local_output'


def remote_0(args):
    """The first function in the remote computation chain
    """
    calculate_mask(args)
    input_ = args["input"]
    site_info = {
        site: input_[site]['categorical_dict']
        for site in input_.keys()
    }

    df = pd.DataFrame.from_dict(site_info)
    covar_keys, unique_count = return_uniques_and_counts(df)

    computation_output_dict = {
        "output": {
            "covar_keys": covar_keys,
            "global_unique_count": unique_count,
            "mask": 'mask.nii',
            "computation_phase": "remote_0"
        },
        "cache": {}
    }

    return json.dumps(computation_output_dict)


def remote_1(args):
    """ The second function in the local computation chain
    """
    input_ = args["input"]
    state_ = args["state"]
    input_dir = state_["baseDirectory"]
    cache_dir = state_["cacheDirectory"]

    site_list = input_.keys()
    user_id = list(site_list)[0]

    input_list = dict()

    for site in site_list:
        file_name = os.path.join(input_dir, site, OUTPUT_FROM_LOCAL)
        input_list[site] = read_file(args, "input", file_name)

    X_labels = input_list[user_id]["X_labels"]

    all_local_stats_dicts = [
        input_list[site]["local_stats_list"] for site in input_list
    ]

    beta_vector_0 = sum([
        np.load(
            os.path.join(input_dir, site,
                         input_list[site]["XtransposeX_local"]))
        for site in input_list
    ])

    beta_vector_1 = sum([
        np.load(
            os.path.join(input_dir, site,
                         input_list[site]["Xtransposey_local"]))
        for site in input_list
    ])

    all_lambdas = [input_list[site]["lambda"] for site in input_list]

    if np.unique(all_lambdas).shape[0] != 1:
        raise Exception("Unequal lambdas at local sites")

    avg_beta_vector = np.transpose(
        np.dot(np.linalg.inv(beta_vector_0), beta_vector_1))

    mean_y_local = [input_list[site]["mean_y_local"] for site in input_list]
    count_y_local = [
        np.array(input_list[site]["count_local"]) for site in input_list
    ]
    mean_y_global = np.array(mean_y_local) * np.array(count_y_local)
    mean_y_global = np.sum(mean_y_global, axis=0) / np.sum(count_y_local,
                                                           axis=0)

    dof_global = sum(count_y_local) - avg_beta_vector.shape[1]

    np.save(
        os.path.join(args["state"]["transferDirectory"],
                     'avg_beta_vector.npy'), avg_beta_vector)
    np.save(
        os.path.join(args["state"]["transferDirectory"], 'mean_y_global.npy'),
        mean_y_global)

    np.save(
        os.path.join(args["state"]["cacheDirectory"],
                     'avg_beta_vector.npy'), avg_beta_vector)

    output_dict = {
        "avg_beta_vector": 'avg_beta_vector.npy',
        "mean_y_global": 'mean_y_global.npy',
        "computation_phase": "remote_1"
    }

    cache_dict = {
        "avg_beta_vector": 'avg_beta_vector.npy',
        "dof_global": dof_global.tolist(),
        "X_labels": X_labels,
        "local_stats_dict": all_local_stats_dicts
    }

    computation_output_dict = {"output": output_dict, "cache": cache_dict}

    file_name = os.path.join(cache_dir, 'remote_cache')
    with open(file_name, 'w') as file_h:
        input_list[site] = json.dump(cache_dict, file_h)

    return json.dumps(computation_output_dict)


def remote_2(args):
    """
    Computes the global model fit statistics, r_2_global, ts_global, ps_global

    Args:
        args (dictionary): {"input": {
                                "SSE_local": ,
                                "SST_local": ,
                                "varX_matrix_local": ,
                                "computation_phase":
                                },
                            "cache":{},
                            }

    Returns:
        computation_output (json) : {"output": {
                                        "avg_beta_vector": ,
                                        "beta_vector_local": ,
                                        "r_2_global": ,
                                        "ts_global": ,
                                        "ps_global": ,
                                        "dof_global":
                                        },
                                    "success":
                                    }
    Comments:
        Generate the local fit statistics
            r^2 : goodness of fit/coefficient of determination
                    Given as 1 - (SSE/SST)
                    where   SSE = Sum Squared of Errors
                            SST = Total Sum of Squares
            t   : t-statistic is the coefficient divided by its standard error.
                    Given as beta/std.err(beta)
            p   : two-tailed p-value (The p-value is the probability of
                  seeing a result as extreme as the one you are
                  getting (a t value as large as yours)
                  in a collection of random data in which
                  the variable had no effect.)

    """
    cache_ = args["cache"]
    state_ = args["state"]
    input_dir = state_["baseDirectory"]
    cache_dir = state_["cacheDirectory"]

    input_list = dict()
    site_list = args["input"].keys()
    for site in site_list:
        file_name = os.path.join(input_dir, site, OUTPUT_FROM_LOCAL)
        with open(file_name, 'r') as file_h:
            input_list[site] = json.load(file_h)

    cache_list = read_file(args, "cache", 'remote_cache')

    X_labels = args["cache"]["X_labels"]

    all_local_stats_dicts = cache_["local_stats_dict"]

#    avg_beta_vector = cache_list["avg_beta_vector"]
#    dof_global = cache_list["dof_global"]
    
    avg_beta_vector = np.load(os.path.join(cache_dir, cache_list["avg_beta_vector"]))
    dof_global = cache_list["dof_global"]
    
    SSE_global = sum(
        [np.array(input_list[site]["SSE_local"]) for site in input_list])
    SST_global = sum(
        [np.array(input_list[site]["SST_local"]) for site in input_list])
    varX_matrix_global = sum([
        np.array(input_list[site]["varX_matrix_local"]) for site in input_list
    ])

    r_squared_global = 1 - (SSE_global / SST_global)
    MSE = SSE_global / np.array(dof_global)
    ts_global = remote_stats(MSE, varX_matrix_global,
                             avg_beta_vector)
    ps_global = [
        2 * stats.t.sf(np.abs(t), df) for t, df in zip(ts_global, dof_global)
    ]

    print_pvals(args, ps_global, ts_global, X_labels)
    print_beta_images(args, avg_beta_vector, X_labels)

    # Block of code to print local stats as well
    sites = [site for site in input_list]

    all_local_stats_dicts = dict(zip(sites, all_local_stats_dicts))

    # Block of code to print just global stats
    global_dict_list = encode_png(args)

    # Print Everything
    keys2 = ["global_stats", "local_stats"]
    output_dict = dict(zip(keys2, [global_dict_list, all_local_stats_dicts]))

    computation_output_dict = {"output": output_dict, "success": True}

    return json.dumps(computation_output_dict)


if __name__ == '__main__':

    PARAM_DICT = json.loads(sys.stdin.read())
    PHASE_KEY = list(list_recursive(PARAM_DICT, 'computation_phase'))

    if "local_0" in PHASE_KEY:
        sys.stdout.write(remote_0(PARAM_DICT))
    elif "local_1" in PHASE_KEY:
        sys.stdout.write(remote_1(PARAM_DICT))
    elif "local_2" in PHASE_KEY:
        sys.stdout.write(remote_2(PARAM_DICT))
    else:
        raise ValueError("Error occurred at Remote")
