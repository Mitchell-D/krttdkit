
import numpy as np
from scipy import stats
from . import enhance

def minimum_distance(X:np.ndarray, categories:dict):
    """
    Given a dictionary mapping category labels to lists of pixel coordinates
    for axes 0 and 1 of a (M,N,C) ndarray (for C bands on the same domain),
    categorizes every pixel, and returns an integer-coded categorization.
    """
    labels, pixel_lists = zip(*categories.items())
    means = [] #
    for i in range(X.shape[2]):
        X[:,:,i] = enhance.linear_gamma_stretch(X[:,:,i])
    for i in range(len(pixel_lists)):
        means.append(np.array([
            sum([ X[y,x,j] for y,x in pixel_lists[i] ])/len(pixel_lists[i])
            for j in range(X.shape[2])
            ]))

    means_sq = [np.dot(means[i], means[i]) for i in range(len(means))]

    classified = np.full_like(X[:,:,0], fill_value=np.nan, dtype=np.uint8)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            px = X[i,j,:]
            disc = [ np.dot(px, px) + means_sq[m] - 2*np.dot(means[m], px)
                    for m in range(len(means)) ]
            classified[i,j] = disc.index(min(disc))
    return classified, labels

def mlc(X:np.ndarray, categories:dict, thresh:float=None):
    """
    Do maximum likelihood classification using the discriminant function

    :@param X: (M,N,b) ndarray with b independent variables
    :@param X: Dictionary mapping category labels to a set of 2-tuple pixel
            indeces of pixels in X belonging to that class.
    :@param thresh: Pixel confidence threshold in percent [0,1] Pixels
            classified with less confidence than the threshold will be
            added to a new "uncertain" category.
    :@return: 2-tuple like (classified, keys) containing the integer-
            -classified array, and a list of keys with indeces corresponding
            to the values in the array labeled by that category.
    """
    cat_keys = list(categories.keys())
    # Chi threshold depends on degrees of freedom and significance threshold
    chi_thresh = None if not thresh else stats.chi2.ppf(thresh, df=X.shape[2])
    cats = [X[tuple(map(np.asarray, tuple(zip(*categories[cat]))))]
            for cat in cat_keys]
    means = [ np.mean(c, axis=0) for c in cats ]
    covs = [ np.cov(c.transpose()) for c in cats ]
    nln_covs = [ -1*np.log(np.linalg.det(C)) for C in covs ]
    inv_covs = [ np.linalg.inv(C) for C in covs ]
    if thresh:
        cat_keys.append("uncertain")
    def mlc_disc(px):
        G = np.zeros_like(np.arange(len(means)))
        chi = np.zeros_like(np.arange(len(means)))
        for i in range(len(means)):
            obs_cov = np.dot(inv_covs[i], px-means[i])
            # If pixel brightnesses are normally distributed, obs_cov
            # should have a chi-squared distribution.
            obs_cov = np.dot((px-means[i]).transpose(), obs_cov)
            #obs_cov = np.dot((px-means[i]).transpose(), inv_covs[i])
            G[i] = nln_covs[i]-obs_cov
            chi[i] = nln_covs[i]-obs_cov
        idx = np.argmax(G)
        if not thresh:
            return idx
        if G[idx] <= -.5*chi_thresh+.5*nln_covs[idx]:
            return len(means)
        return idx

    classified = np.zeros_like(X[:,:,0])
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            classified[i,j] = mlc_disc(X[i,j])
    return classified, cat_keys

def k_means(X:np.ndarray, cluster_count:int, tolerance=1e-3,
            get_sse:bool=False, debug:bool=False):
    """
    Perform k-means clustering on the input dataset with a provided number
    of clusters and a decimal tolerance for cluster mean equality.

    :@return: list of lists containing the indeces of each pixel belonging
            to a cluster.
    """
    px_mean = np.zeros_like(np.arange(cluster_count))
    def new_centroid():
        """ Randomize centroid locations"""
        nonlocal X
        #return np.random.rand(X.shape[2])
        #'''
        rand_y = np.random.randint(0, X.shape[0])
        rand_x = np.random.randint(0, X.shape[1])
        return X[rand_y, rand_x]
        #'''

    # Pick random pixels to initialize the means
    centroids = [ new_centroid() for c in range(cluster_count) ]
    all_valid = False
    pc_pass = 0
    sse = [] # sum of squared error
    while not all_valid:
        '''
        if any([ np.any(np.isnan(c)) for c in centroids ]):
            if debug: print(f"resetting centroids...")
            pc_pass = 0
            centroids = [ new_centroid() for c in range(cluster_count) ]
        elif pc_pass != 0:
        '''
        tmp_sse = 0
        for c in range(cluster_count):
            if np.all(np.isnan(centroids[c])):
                centroids[c] = new_centroid()
                if debug:
                    print(f"New centroid for class {c}: {new_centroids[c]}")
        if pc_pass != 0 and debug:
            print([f"({c[0]:.4f}, {c[1]:.4f})" for c in centroids])
        pc_pass += 1
        new_centroids = []
        clusters = [ [] for i in range(cluster_count)]
        cluster_idx = [ [] for i in range(cluster_count)]
        if debug: print(f"\nK-means pass {pc_pass}")
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for c in range(cluster_count):
                    px_mean[c] = np.linalg.norm(X[i,j]-centroids[c])
                # Get the index of the closest centroid and assign this pixel
                cidx = np.argmin(px_mean)
                clusters[cidx].append(X[i,j])
                cluster_idx[cidx].append((i,j))
                if get_sse:
                    tmp_sse += np.linalg.norm(X[i,j]-centroids[cidx])**2
        # Collect centroid pixels
        for c in range(cluster_count):
            # Average all pixels in each centroid per band
            new_centroids.append(np.average(np.asarray(clusters[c]), axis=0))
            # Reset a centroid if it had no members.
        all_valid = all([np.allclose(oldc,newc,tolerance) for oldc, newc
                         in zip(centroids, new_centroids)])
        centroids = new_centroids
        if get_sse:
            print(f"SSE: {tmp_sse}")
            sse.append(tmp_sse)
    '''
    Y = np.zeros_like(X[:,:,0])
    for c in range(cluster_count):
        if debug: print(f"Cluster {c}: {centroids[c]} {len(clusters[c])}")
        for i,j in cluster_idx[c]:
            Y[i,j] = c
    '''
    if not get_sse:
        return cluster_idx
    return cluster_idx, sse

def pca(X:np.ndarray, print_table:bool=False):
    """
    Perform principle component analysis on the provided array, and return
    the transformed array of principle components
    """
    flatX = np.copy(X).transpose(2,0,1).reshape(X.shape[2],-1)
    # Get a vector of the mean value of each band
    means = np.mean(flatX, axis=0)
    # Get a bxb covariance matrix for b bands
    covs = np.cov(flatX)
    # Calculate and sort eigenvalues and eigenvectors
    eigen = list(np.linalg.eig(covs))
    eigen[1] = list(map(list, eigen[1]))
    eigen = list(zip(*eigen))
    eigen.sort(key=lambda e: e[0])
    evals, evecs = zip(*eigen)
    # Get a diagonal matrix of eigenvalues
    transform = np.dstack(evecs).transpose().squeeze()
    Y = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i,j,:] = np.dot(transform, X[i,j,:])
    if print_table:
        cov_string = ""
        ev_string = ""
        for i in range(covs.shape[0]):
            cov_string+=" & ".join(
                    [f"{x:.4f}" for x in covs[i,:]])
            ev_string+=f"{evals[i]:.4f}"+" & "+" & ".join(
                    [f"{x:.4f}"for x in evecs[i]])
            ev_string += " \\\\ \n"
            cov_string += " \\\\ \n"
        print("Covariance matrix:")
        print(cov_string)
        print("Eigenvalue and Eigenvector table:")
        print(ev_string)

    return Y

