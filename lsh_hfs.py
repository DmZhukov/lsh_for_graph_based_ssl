import numpy as np
import _pickle as cPickle
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors
import scipy.sparse
import scipy.sparse.linalg
from sklearn.decomposition import PCA
import math
import nearpy

plt.style.use('ggplot')

def unpickle(file):
    fo = open(file, 'rb')
    data = cPickle.load(fo, encoding='latin1')
    fo.close()
    return data

def mask(n, p):
# n - number of samples
# p - probability of masking a label
# randomly choose which labels to mask
    return np.array(np.random.rand(n,1) < p, dtype=np.int32)

def build_knn_graph(similarities, k):
    weights = np.zeros(similarities.shape)
    for l in range(k):
        idx = np.argmax(similarities, axis = 1)
        for i,j in enumerate(idx):
            weights[i,j] = weights[j,i] = similarities[i,j]
            similarities[i,j] = similarities[j,i] = 0
    return weights

def gaussian_similarity(distance, sigma):
    return np.exp(-distance*distance / (2*sigma**2))

def get_similarities(weights):
    row, col, distances = scipy.sparse.find(weights)
    similarities = gaussian_similarity(distances, sigma)
    return  scipy.sparse.coo_matrix((similarities, (row, col)), shape=weights.shape)

def get_laplacian(weights):
    return scipy.sparse.diags(np.squeeze(np.array(weights.sum(axis=1))), 0) - weights

def get_approximate_neighbors(query, data, engines_list, k):
# k - number of neighbors
    L = len(engines_list)
    neighbors = []
    distances = []
    idxs = np.zeros(L, dtype=np.int32)
    candidate_indexes = set()
    for l in range(L):
        bucket = engines_list[l].neighbours(query)
        candidate_indexes = candidate_indexes.union({el[1] for el in bucket})
    candidate_indexes = list(candidate_indexes)
    candidates = data[candidate_indexes,:]
    distances, neighbors = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(candidates).kneighbors(query.reshape([1,-1]))
    return neighbors.squeeze(), distances.squeeze()

def build_approx_graph(data, k, L, projection_count=20):
    n, d = data.shape
    engine =[]
    for l in range(L):
        engine.append(nearpy.Engine(d, lshashes=[ nearpy.hashes.RandomBinaryProjectionTree('rbp',projection_count, k+1) ],
            distance=nearpy.distances.EuclideanDistance()))
    for i in range(n):
        for l in range(L):
            engine[l].store_vector(data[i,:], i)
    weights = scipy.sparse.dok_matrix((n,n), dtype=np.float32)
    for i in range(n):
        neighbors, distances = get_approximate_neighbors(data[i,:], data, engine, k+1)
        neighbors = neighbors[1:] # get rid of the first neighbor that is a query itself
        distances = distances[1:]
        for j in range(k):
            weights[i,neighbors[j]] = distances[j]
            weights[neighbors[j],i] = distances[j]
    return weights

def build_graph(data, k):
    n, d = data.shape
    #knn = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(data)
    all_distances, all_neighbors = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(data).kneighbors(data)
    weights = scipy.sparse.dok_matrix((n,n), dtype=np.float32)
    for i in range(n):
        neighbors = all_neighbors[i,1:] # get rid of the first neighbor that is a query itself
        distances = all_distances[i,1:]
        for j in range(k):
            weights[i,neighbors[j]] = distances[j]
            weights[neighbors[j],i] = distances[j]
    return weights

def solve_HFS(laplacian, c_u, c_l, gamma_g, y):
    C_inv_array = np.array(1./c_u*(y[:,0]==0) + 1./c_l*(y[:,0]!=0), dtype=np.float32)
    C_inv = scipy.sparse.diags(C_inv_array, 0)
    Q = laplacian + gamma_g*scipy.sparse.eye(n)
    return scipy.sparse.linalg.spsolve(C_inv.dot(Q) + scipy.sparse.eye(n), y)

def HFS(data, y, k, gamma_g, sigma, c_u, c_l, approx=False, L=5, projection_count=20, laplacian = None):
    if not approx:
        weights = build_graph(data, k)
    else:
        weights = build_approx_graph(data, k, L, projection_count)
    weights = get_similarities(weights)
    laplacian = get_laplacian(weights)
    return solve_HFS(laplacian, c_u,c_l, gamma_g,y), laplacian


if __name__ == '__main__':
	# Reading data
	data = []
	labels = []
	for i in range(5):
	    batch = unpickle('./cifar-10-batches-py/data_batch_%d' % (i+1))
	    data.append(batch['data'])
	    labels.append(np.array(batch['labels']))
	data = np.concatenate(data, axis=0)
	labels = np.concatenate(labels, axis=0)
	labels = OneHotEncoder(sparse=False).fit_transform(labels.reshape([-1,1]))
	labels = 2*labels-1

	n = 5000 # number of samples
	p = 0.1 # probability of unmasking a label
	idxs = np.random.permutation(np.arange(data.shape[0]))[:n]
	data = data[idxs,:]
	labels = labels[idxs]
	_mask = mask(n, p)
	y = labels*_mask # masked labels
	n_l = np.sum(_mask)

	dimension = 100
	pca = PCA(n_components=100)
	data = pca.fit_transform(data)

	k = 10
	sigma = 1000.

	gamma_g = math.sqrt(n_l**3)
	c_u = 1
	c_l = 1
	L = 5


	l, laplacian = HFS(data, y, k, gamma_g, sigma, c_u, c_l, approx=False)
	l_error = []
	laplacian_error = []
	for L in range(2,50,2):
	    print('L = %d' % L)
	    l_approx, laplacian_approx = HFS(data, y, k, gamma_g, sigma, c_u, c_l, approx=True, L = L)
	    l_error.append(np.sum((l_approx - l)**2))
	    laplacian_error.append(scipy.sparse.linalg.norm(laplacian-laplacian_approx, ord='fro'))

	np.savetxt('l_error_L.txt', np.array(l_error, dtype=np.float32))
	np.savetxt('laplacian_error_L.txt', np.array(l_error, dtype=np.float32))

	plt.figure()
	plt.plot(l_error)
	plt.show()
	plt.figure()
	plt.plot(laplacian_error)
	plt.show()


	laplacian = get_laplacian(get_similarities(build_graph(data, k)))
	laplacian_approx = get_similarities(get_similarities(build_approx_graph(data, k, L=15)))

	error = []
	for gamma_g in range(1,1000, 10):
	    print('gamma_g = %f' % gamma_g)
	    l = solve_HFS(laplacian, c_u, c_l, gamma_g, y)
	    l_approx = solve_HFS(laplacian_approx, c_u, c_l, gamma_g, y)
	    error.append(np.sum((l_approx - l)**2))

	np.savetxt('l_error_gamma_g.txt', np.array(error, dtype=np.float32))

	plt.figure()
	plt.plot(np.array(error))
	plt.plot(2576*np.power(1./np.arange(1,1000,1), 4))
	plt.show()