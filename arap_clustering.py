import sys
sys.path.append('../../Box/data-analysis/libraries')

import scipy as sp
import numpy as np
import pandas as pd

import pickle
import time
import copy
import re
import os

from copy import deepcopy
from scipy.stats import norm
from online_affinity_propagation_ import OnlineAffinityPropagation
from pairwise import custom_distances
from match_names import cluster_from_file

"""
Remove all of the stop words in a dataframe using a list of stopwords and
regular expressions. Also cleans up extra whitespace.
"""
def remove_stops(df, stops):
    stopdf = deepcopy(df)

    for col in stopdf.columns:
        stopdf[col] = stopdf[col].str.lower().str.strip()

    restring = ''
    for stop in stops:
        if(stop.strip() != ''):
            restring+=stop.lower().strip() + '|'
    restring = restring.strip('|')
    restring = "(?i)\\b(" + restring + ")\\b"
    #exp = re.compile(restring)
    stopdf.replace(to_replace=restring, inplace=True, regex=True, value = ' ')
    restring = '[^A-z]+'
    # exp = re.compile(restring)
    stopdf.replace(to_replace=restring, inplace=True, regex=True, value = ' ')
    restring = ' {2,}'
    # exp = re.compile(restring)
    stopdf.replace(to_replace=restring, inplace=True, regex=True, value = ' ')
    restring = ' +$'
    # exp = re.compile(restring)
    stopdf.replace(to_replace=restring, inplace=True, regex=True, value = '')
    restring = '^ *$'
    # exp = re.compile(restring)
    stopdf.replace(to_replace=restring, inplace=True, regex=True,
                                    value='empty after stopword removal')
    return stopdf

"""
get a list of stopwords from a file with one stopword per line.
This file supports python single-line comments (with '#')
"""
def get_stoplist(filename):
    try:
        f = open(filename)
        stoplist = f.read().split('\n')
    except:
        print("file not found, defaulting stopwords to empty list")
        stoplist = []
    test = {}
    for stop in stoplist:
        if(stop.strip() == ''):
            continue
        if(stop.strip()[0] == '#'):
            continue
        if stop in test:
            print("extra {} found".format(stop))
            stoplist.remove(stop)
        else:
            test[stop] = True
    return stoplist
"""
cleans up a file of arap data. Only cleans columns with headers given by
headers. gets rid of rows with nan values. These can be examined by humans
"""
def get_clean_data(datafolder, filename, stoplistfile, headers, datahead):
    cols = copy.deepcopy(headers)
    filepath = datafolder+filename
    stopwords = get_stoplist(stoplistfile)
    df = pd.read_csv(filepath, dtype=str)
    for header in cols:
        df[header] = df[header].str.lower()
        df[header] = df[header].str.strip()
        df[header] = df[header].str.replace('[ ]+', ' ')
    df.drop_duplicates(subset=cols, inplace=True)
    tmpdf = remove_stops(df, stopwords)[cols]
    for i in range(len(cols)):
        cols[i] = 'nostops_' + cols[i]
    tmpdf.columns=cols
    df = pd.concat([df,tmpdf], axis=1)
    df.drop_duplicates(subset=datahead, inplace=True)
    df.dropna(subset=cols, inplace=True)
    return df

if __name__ == '__main__':
    # check that we get the right output for strings

    def jaccard_distance(a, b, N=3):
        """Calculate the jaccard distance between ngrams from words A and B"""
        while(len(a) < N):
            a += ' '
        while(len(b) < N):
            b += ' '
        a = [a[i:i+N] for i in range(len(a)-N+1)]
        b = [b[i:i+N] for i in range(len(b)-N+1)]
        a = set(a)
        b = set(b)
        return float(1.0 * len(a&b)/len(a|b))

    data_folder = '/home/mathiaskallick/Documents/Box/data/cust_financials/'
    file_name = 's3_arap_data.csv'
    headers = ['Borrower Name', 'Customer or Vendor']

    stoplist_file = 'arap_cust_vend_stopwords.txt'
    datahead = ['Customer or Vendor']

    X = get_clean_data(data_folder, file_name, stoplist_file, headers, datahead)

    X = X['Customer or Vendor'].values

    print("X: {}".format(X))

    print("X.shape: {}".format(X.shape))
    train_data = np.copy(X[:6000,]) #6000 for full data set
    test_data = np.copy(X[6000:,])

    init_cluster = True
    add_data_to_model = True
    recluster_model = True and (init_cluster or add_data_to_model)

    oap=None

    if(init_cluster):
        try:
            os.remove('stored_oap')
        except Exception as e:
            print("stored model can't be deleted: {}".format(e))

        oap = OnlineAffinityPropagation(affinity='custom',
                                        affinity_function=jaccard_distance,
                                        max_iter=1000, verbose=True,
                                        preference=.8, convergence_iter=25)

        starttime = time.clock()
        oap.fit(train_data)
        endtime = time.clock()

        print("fit runtime: {}".format(endtime-starttime))

        oap.to_json('stored_oap')

    if(add_data_to_model):
        oap = OnlineAffinityPropagation()

        oap.read_from_file('stored_oap', jaccard_distance)
        starttime = time.clock()
        oap.add_data(test_data, verbose=False)
        endtime = time.clock()

        oap.to_json('stored_oap')

        print("add data runtime: {}".format(endtime-starttime))

    if(recluster_model):
        starttime = time.clock()
        oap.recluster()
        endtime = time.clock()

        print("recluster runtime: {}".format(endtime-starttime))

    if(not (add_data_to_model or init_cluster)):
        oap = OnlineAffinityPropagation()
        oap.read_from_file('stored_oap', jaccard_distance)

    print("oap.data_.shape: {}".format(oap.data_.shape))
    print("oap.labels_.shape: {}".format(oap.labels_.shape))
    # print("tmpnames.shape: {}".format(tmpnames.shape))
    cluster_array = np.concatenate([
                    np.reshape(oap.data_,(oap.data_.shape[0],1)),
                    np.reshape(oap.labels_,(oap.labels_.shape[0],1))], axis=1)
                    # np.reshape(tmpnames, (tmpnames.shape[0], 1))],axis=1)

    # cluster_array = cluster_array[np.where(cluster_array[:,1] != -1)]
    #
    #cluster_array = cluster_array[cluster_array[:,1].argsort()]

    print("clusters:\n{}".format(cluster_array))

    txt = pd.DataFrame(cluster_array)
    txt.to_csv('no_re_compile_clusters.csv')
