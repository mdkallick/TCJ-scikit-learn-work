"""Online Affinity Propagation clustering algorithm."""

# Author: Alexandre Gramfort alexandre.gramfort@inria.fr
#        Gael Varoquaux gael.varoquaux@normalesup.org
#       Mathias Dyssegaard Kallick mdkallick@gmail.com

# License: BSD 3 clause

# TODO: Rename custom_distances to a more reasonable name

import numpy as np
import scipy as sp

from inspect import getsource
from pickle import load
from pickle import dump
from pickle import dumps

from scipy.stats import norm
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import as_float_array, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import euclidean_distances
from sklearn.metrics import pairwise_distances_argmin
from pairwise import custom_distances

def affinity_propagation(S, preference=None, convergence_iter=15, max_iter=200,
                         damping=0.5, copy=True, verbose=False,
                         return_n_iter=False):
    """Perform Affinity Propagation Clustering of data
    Read more in the :ref:`User Guide <affinity_propagation>`.
    Parameterssudo dnf install sublime-text
    ----------
    S : array-like, shape (n_samples, n_samples)
        Matrix of similarities between points
    preference : array-like, shape (n_samples,) or float, optional
        Preferences for each point - points with larger values of
        preferences are more likely to be chosen as exemplars. The number of
        exemplars, i.e. of clusters, is influenced by the input preferences
        value. If the preferences are not passed as arguments, they will be
        set to the median of the input similarities (resulting in a moderate
        number of clusters). For a smaller amount of clusters, this can be set
        to the minimum value of the similarities.
    convergence_iter : int, optional, default: 15
        Number of iterations with no change in the number
        of estimated clusters that stops the convergence.
    max_iter : int, optional, default: 200
        Maximum number of iterations
    damping : float, optional, default: 0.5
        Damping factor between 0.5 and 1.
    copy : boolean, optional, default: True
        If copy is False, the affinity matrix is modified inplace by the
        algorithm, for memory efficiency
    verbose : boolean, optional, default: False
        The verbosity level
    return_n_iter : bool, default False
        Whether or not to return the number of iterations.
    Returns
    -------
    cluster_centers_indices : array, shape (n_clusters,)
        index of clusters centers
    labels : array, shape (n_samples,)
        cluster labels for each point
    n_iter : int
        number of iterations run. Returned only if `return_n_iter` is
        set to True.
    Notes
    -----
    For an example, see :ref:`examples/cluster/plot_affinity_propagation.py
    <sphx_glr_auto_examples_cluster_plot_affinity_propagation.py>`.
    References
    ----------
    Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
    Between Data Points", Science Feb. 2007
    """
    S = as_float_array(S, copy=copy)
    n_samples = S.shape[0]

    if S.shape[0] != S.shape[1]:
        raise ValueError("S must be a square array (shape=%s)" % repr(S.shape))

    if preference is None:
        preference = np.median(S)
    if damping < 0.5 or damping >= 1:
        raise ValueError('damping must be >= 0.5 and < 1')

    random_state = np.random.RandomState(0)

    # Place preference on the diagonal of S
    S.flat[::(n_samples + 1)] = preference

    A = np.zeros((n_samples, n_samples))
    R = np.zeros((n_samples, n_samples))  # Initialize messages
    # Intermediate results
    tmp = np.zeros((n_samples, n_samples))

    # Remove degeneracies
    S += ((np.finfo(np.double).eps * S + np.finfo(np.double).tiny * 100) *
          random_state.randn(n_samples, n_samples))

    # Execute parallel affinity propagation updates
    e = np.zeros((n_samples, convergence_iter))

    ind = np.arange(n_samples)

    for it in range(max_iter):
        # tmp = A + S; compute responsibilities
        np.add(A, S, tmp)
        I = np.argmax(tmp, axis=1)
        Y = tmp[ind, I]  # np.max(A + S, axis=1)
        tmp[ind, I] = -np.inf
        Y2 = np.max(tmp, axis=1)

        # tmp = Rnew
        np.subtract(S, Y[:, None], tmp)
        tmp[ind, I] = S[ind, I] - Y2

        # Damping
        tmp *= 1 - damping
        R *= damping
        R += tmp

        # tmp = Rp; compute availabilities
        np.maximum(R, 0, tmp)
        tmp.flat[::n_samples + 1] = R.flat[::n_samples + 1]

        # tmp = -Anew
        tmp -= np.sum(tmp, axis=0)
        dA = np.diag(tmp).copy()
        tmp.clip(0, np.inf, tmp)
        tmp.flat[::n_samples + 1] = dA

        # Damping
        tmp *= 1 - damping
        A *= damping
        A -= tmp

        # Check for convergence
        E = (np.diag(A) + np.diag(R)) > 0
        e[:, it % convergence_iter] = E
        K = np.sum(E, axis=0)

        if it >= convergence_iter:
            se = np.sum(e, axis=1)
            unconverged = (np.sum((se == convergence_iter) + (se == 0))
                           != n_samples)
            if (not unconverged and (K > 0)) or (it == max_iter):
                if verbose:
                    print("Converged after %d iterations." % it)
                break
    else:
        if verbose:
            print("Did not converge")

    I = np.where(np.diag(A + R) > 0)[0]
    K = I.size  # Identify exemplars

    if K > 0:
        c = np.argmax(S[:, I], axis=1)
        c[I] = np.arange(K)  # Identify clusters
        # Refine the final set of exemplars and clusters and return results
        for k in range(K):
            ii = np.where(c == k)[0]
            j = np.argmax(np.sum(S[ii[:, np.newaxis], ii], axis=0))
            I[k] = ii[j]

        c = np.argmax(S[:, I], axis=1)
        c[I] = np.arange(K)
        labels = I[c]
        # Reduce labels to a sorted, gapless, list
        cluster_centers_indices = np.unique(labels)
        labels = np.searchsorted(cluster_centers_indices, labels)

        STRAP = []
        dists = 1 - S.copy()
        np.fill_diagonal(dists, 0)
        #print("dists: {}".format(dists))

        empty_clusters = []

        # calculate the true cluster centers (i.e. the exemplars), by
        # finding the element in each cluster with the highest sum of
        # similarities with every other element in the cluster
        for k in range(labels.max()+1):
            locs = np.where(labels==k)[0]
            cluster_size = locs.shape[0]
            # if the cluster only has one element, throw it in the reservoir
            if(cluster_size == 1):
                #print("locs: {}".format(locs))
                labels[locs] = -1
                empty_clusters.append(k)
                continue
            tmp_cluster_distances = np.zeros((cluster_size,cluster_size))
            i=0
            for r in np.nditer(locs):
                j=0
                for c in np.nditer(locs):
                    tmp_cluster_distances[i,j] = dists[r,c]
                    j+=1
                i+=1
            tmp_sums = np.sum(tmp_cluster_distances, axis=1)
            sq_tmp_sums = np.sum(np.square(tmp_cluster_distances), axis=1)
            ex_loc = np.argmax(tmp_sums)
            exemplar_distance = tmp_sums[ex_loc]
            sq_exemplar_distance = sq_tmp_sums[ex_loc]
            cluster_centers_index = locs[ex_loc]
            # create the STRAP characterization of each examplar (4-tuple) in an array
            # (exemplar, clus_size, sum_distances, sum_squared_distances)
            STRAP.append([cluster_centers_index, cluster_size-1,
                                exemplar_distance, sq_exemplar_distance])

        # make sure there aren't missing clusters #TODO: uncomment this maybe?
        for i in range(len(empty_clusters)):
            empty_clusters[i] -= i

        for k in empty_clusters:
            for l in np.nditer(labels, op_flags=['writeonly']):
                if(l > k):
                    l[...] -= 1
        # print("STRAP: {}".format(STRAP))

    else:
        labels = np.empty((n_samples, 1))
        cluster_centers_indices = None
        labels.fill(np.nan)
    #print("labels: {}".format(labels))
    if return_n_iter:
        return cluster_centers_indices, labels, STRAP, it + 1
    else:
        return cluster_centers_indices, labels, STRAP

###############################################################################

class OnlineAffinityPropagation(BaseEstimator, ClusterMixin):
    """Perform Affinity Propagation Clustering of data.
    Read more in the :ref:`User Guide <affinity_propagation>`.
    Parameters
    ----------
    damping : float, optional, default: 0.5
        Damping factor (between 0.5 and 1) is the extent to
        which the current value is maintained relative to
        incoming values (weighted 1 - damping). This in order
        to avoid numerical oscillations when updating these
        values (messages).
    max_iter : int, optional, default: 200
        Maximum number of iterations.
    convergence_iter : int, optional, default: 15
        Number of iterations with no change in the number
        of estimated clusters that stops the convergence.
    copy : boolean, optional, default: True
        Make a copy of input data.
    preference : array-like, shape (n_samples,) or float, optional
        Preferences for each point - points with larger values of
        preferences are more likely to be chosen as exemplars. The number
        of exemplars, ie of clusters, is influenced by the input
        preferences value. If the preferences are not passed as arguments,
        they will be set to the median of the input similarities.
    affinity : string, optional, default=``euclidean``
        Which affinity to use. At the moment ``precomputed`` and
        ``euclidean`` are supported. ``euclidean`` uses the
        negative squared euclidean distance between points.
    verbose : boolean, optional, default: False
        Whether to be verbose.
    Attributes
    ----------
    cluster_centers_indices_ : array, shape (n_clusters,)
        Indices of cluster centers
    cluster_centers_ : array, shape (n_clusters, n_features)
        Cluster centers (if affinity != ``precomputed``).
    labels_ : array, shape (n_samples,)
        Labels of each point
    affinity_matrix_ : array, shape (n_samples, n_samples)
        Stores the affinity matrix used in ``fit``.
    n_iter_ : int
        Number of iterations taken to converge.
    Notes
    -----
    For an example, see :ref:`examples/cluster/plot_affinity_propagation.py
    <sphx_glr_auto_examples_cluster_plot_affinity_propagation.py>`.
    The algorithmic complexity of affinity propagation is quadratic
    in the number of points.
    References
    ----------
    Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
    Between Data Points", Science Feb. 2007
    """

    def __init__(self, damping=.5, max_iter=200, convergence_iter=15,
                 copy=True, preference=None, affinity='euclidean',
                 affinity_function=None, verbose=False):

        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.copy = copy
        self.verbose = verbose
        self.preference = preference
        self.affinity = affinity
        self.affinity_function = affinity_function

    @property
    def _pairwise(self):
        return self.affinity == "precomputed"

    def fit(self, X, y=None):
        """ Create affinity matrix from negative euclidean distances, then
        apply affinity propagation clustering.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or (n_samples, n_samples)
            Data matrix or, if affinity is ``precomputed``, matrix of
            similarities / affinities.
        """
        #X = check_array(X, accept_sparse='csr', dtype=X.dtype)
        print("X.shape: {}".format(X.shape))
        self.data_ = X
        if self.affinity == "precomputed":
            self.affinity_matrix_ = X
        elif self.affinity == "euclidean":
            self.affinity_matrix_ = -euclidean_distances(X, squared=True)
        elif self.affinity == "custom":
            if(self.affinity_function==None):
                raise ValueError("A distance function must be specified"
                                 "for 'custom' Affinity.")
            self.affinity_matrix_ = custom_distances(self.affinity_function, X)
        else:
            raise ValueError("Affinity must be 'precomputed' or "
                             "'euclidean'. Got %s instead"
                             % str(self.affinity))
        #print("self.affinity_matrix_: {}".format(self.affinity_matrix_))
        self.cluster_centers_indices_, self.labels_, self.STRAP, self.n_iter_ = \
            affinity_propagation(
                self.affinity_matrix_, self.preference, max_iter=self.max_iter,
                convergence_iter=self.convergence_iter, damping=self.damping,
                copy=self.copy, verbose=self.verbose, return_n_iter=True)

        # print("self.labels_: {}".format(self.labels_))
        # print("self.STRAP: {}".format(self.STRAP))
        if self.affinity != "precomputed":
            self.cluster_centers_ = X[self.cluster_centers_indices_].copy()

        return self

    def add_data(self, Xin, copy=True, verbose=False):
        """Add data to the clusters - will attempt to add each point of data to
        a cluster, and if it fails, will temporarily store unclustered data in
        a reservoir. When the reservoir gets full, will recluster with the
        reservoir data.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape(n_samples, n_features)
        Returns
        -------
        """
        check_is_fitted(self, "cluster_centers_indices_")
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("Add Data method is not supported when "
                             "affinity='precomputed'.")

        # copy the data
        if(copy):
            X = np.copy(Xin)

        # get a numpy finfo for floats
        f = np.finfo(np.float32)

        #print("X.shape: {}".format(X.shape))
        #print("X: {}".format(X))
        probs = np.zeros((X.shape[0], len(self.STRAP)))
        assigned_clusters = np.zeros((X.shape[0],))

        if(self.affinity == 'euclidean'):
            for exemplar in np.nditer(self.cluster_centers_):
                #TODO: implement for euclidean distances
                pass
        #TODO: generalize to work with more than just strings
        elif(self.affinity == 'custom'):
            i=0
            for exemplar, n, M, Sigma in self.STRAP:
                exemplar = self.data_[exemplar]
                mu = M/n
                var = np.sqrt((Sigma/n) - (np.square(M/n))) + f.eps
                g = norm(mu, var)
                if(verbose):
                    print("n: {}".format(n))
                    print("M: {}".format(M))
                    print("Sigma: {}".format(Sigma))
                    print("mu: {}".format(mu))
                    print("var: {}".format(var))
                j=0
                for data in np.nditer(X, flags=['refs_ok'], op_dtypes=X.dtype):
                    #data = str(data)

                    # calculate the distance between the data point and the
                    # exemplar
                    affinity = self.affinity_function(exemplar, str(data))
                    # print("exemplar, data: {}, {}".format(exemplar, data))
                    # print("affinity: {}".format(affinity))

                    ## if this data point is VERY close to the exemplar, include
                    ## it in the cluster. Using this because excluding it makes
                    ## adding a data point identical to the exemplar not cluster
                    ## into the exemplars cluster, because it is outside of the
                    ## variance from the mean distance.
                    if(1 - affinity < abs(mu) + abs(var)):
                        probs[j,i] = affinity
                        # calculate the mean of the gaussian for this cluster
                        # probs[j,i] = g.pdf(distance)
                        if(verbose):
                            print("affinity: {}".format(affinity))
                            print("g.pdf(affinity): {}".format(g.pdf(affinity)))
                            print("data: {}".format(data))
                            print("exemplar: {}".format(exemplar))
                    print('\rprogress: {}\r'.format((((i*X.shape[0])+
                            j)/(X.shape[0]*len(self.STRAP))) * 100), end='')

                    j+=1

                i+=1
            print()
        max_indices = probs.argmax(axis=1)
        for i in range(max_indices.shape[0]):
            max_proba = probs[i, max_indices[i]]
            if(max_proba > f.eps):
                assigned_clusters[i] = max_indices[i]
                print("string {} was assigned to cluster {}".format(X[i], max_indices[i]))
            else:
                assigned_clusters[i] = -1

        #print("assigned_clusters: {}".format(assigned_clusters))
        self.data_ = np.concatenate([self.data_,X])
        self.labels_ = np.concatenate([self.labels_,assigned_clusters])

        return assigned_clusters

    def recluster(self):
        unclustered_loc = np.where(self.labels_==-1)
        clustered_loc = np.where(self.labels_!=-1)
        data_len = len(self.data_)-1
        reservoir = self.data_[unclustered_loc]
        new_affinity_matrix_ = custom_distances(self.affinity_function,
                                                        reservoir)
        new_cluster_centers_indices_, new_labels_, new_STRAP = \
            affinity_propagation(
                new_affinity_matrix_, self.preference, max_iter=self.max_iter,
                convergence_iter=self.convergence_iter, damping=self.damping,
                copy=self.copy, verbose=self.verbose, return_n_iter=False)
        max_label = int(np.max(self.labels_))+1
        print("max_label: {}".format(max_label))
        print("new_labels_: {}".format(new_labels_))
        for l in np.nditer(new_labels_, op_flags=['writeonly']):
            if(l != -1):
                l[...] += max_label
        self.data_ = np.concatenate([self.data_[clustered_loc],
                                     self.data_[unclustered_loc]])
        self.labels_ = np.concatenate([self.labels_[clustered_loc],
                                        new_labels_])
        for i in range(len(new_STRAP)):
            new_STRAP[i][0]+=data_len
        self.STRAP += new_STRAP
        # print("self.data_: {}".format(self.data_))
        # print("self.labels_: {}".format(self.labels_))
        # print("self.STRAP: {}".format(self.STRAP))

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data to predict.
        Returns
        -------
        labels : array, shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self, "cluster_centers_indices_")
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("Predict method is not supported when "
                             "affinity='precomputed'.")

        return pairwise_distances_argmin(X, self.cluster_centers_)

    def to_json(self, filename, to_file=True):
        json_data = {
                     'affinity': self.affinity,
                     'convergence_iter': self.convergence_iter,
                     'copy': self.copy,
                     'damping': self.damping,
                     'max_iter': self.max_iter,
                     'preference': self.preference,
                     'verbose': self.verbose,
                     'data': self.data_,
                     'STRAP': self.STRAP,
                     'labels': self.labels_,
                     'cluster_centers_indices_': self.cluster_centers_indices_,
                     'cluster_centers_': self.cluster_centers_
                    }

        if(to_file):
            with open(filename, 'w+b') as pickle_file:
                 dump(json_data, pickle_file)
        else:
            return dumps(json_data)

    def read_from_file(self, filename, function=None):
        with open(filename, 'rb') as pickle_file:
            newdata = load(pickle_file)

        if(newdata['affinity'] is not None):
            self.affinity = newdata['affinity']
        if(newdata['convergence_iter'] is not None):
            self.convergence_iter = newdata['convergence_iter']
        if(newdata['copy'] is not None):
            self.copy = newdata['copy']
        if(newdata['damping'] is not None):
            self.damping = newdata['damping']
        if(newdata['max_iter'] is not None):
            self.max_iter = newdata['max_iter']
        if(newdata['preference'] is not None):
            self.preference = newdata['preference']
        if(newdata['verbose'] is not None):
            self.verbose = newdata['verbose']
        if(newdata['data'] is not None):
            self.data_ = newdata['data']
        if(newdata['STRAP'] is not None):
            self.STRAP = newdata['STRAP']
        if(newdata['labels'] is not None):
            self.labels_ = newdata['labels']
        if(newdata['cluster_centers_indices_'] is not None):
            self.cluster_centers_indices_ = newdata['cluster_centers_indices_']
        if(newdata['cluster_centers_'] is not None):
            self.cluster_centers_ = newdata['cluster_centers_']

        if(self.affinity == 'custom' and function == None):
            raise ValueError("In order to use custom affinity,"
                             "you must input a valid function.")

        self.affinity_function = function
