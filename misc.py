import numpy as np
import torch

def letters_to_indices(letters):
    # Convert a string of letters into a list of indices
    # 'A' corresponds to 0, 'B' to 1, 'C' to 2, etc.
    # Assumes letters are provided in uppercase and are in a valid range.
    return [ord(char) - ord('A') for char in letters]


def convert_to_ranking(X,Y):
    #Inpuformat:
    #X,Y #X.shape = (k,D,N) = (num alternatives,space dimensions, number of rankings)
    #Y is either vector of alphabets representing the order Y.shape = (N,1) or Y is ranking order as indices Y.shape = (N,k)
    #Outputformat:
    #X #X.shape = (k,D,N) = (comp,space dimensions, number of rankings)
    #k = X.shape[0]
    N = X.shape[2]
    #D = X.shape[1]
    newX = X.copy()
    for i in range(N):
        if np.issubdtype(Y.dtype, np.str_) or np.issubdtype(Y.dtype, np.object_):
            rankingorder = letters_to_indices(Y[i].item())
        else:
            rankingorder = Y[i,:]
        newX[:,:,i] = np.take(X[:,:,i], rankingorder, axis=0) #Reorder alternatives to match ranking order
    return newX


def convert_to_ranking_and_change_k(X,Y,k):
    #Inpuformat:
    #X,Y #X.shape = (k,D,N) = (num alternatives,space dimensions, number of rankings)
    #Y is either vector of alphabets representing the order Y.shape = (N,1) or Y is ranking order as indices Y.shape = (N,k)
    #Outputformat:
    #X #X.shape = (k,D,N) = (comp,space dimensions, number of rankings)
    k_original = X.shape[0]
    if k > k_original:
        raise ValueError('Desired k is higher than the original k!')
    N = X.shape[2]
    #D = X.shape[1]
    X = X[:k,:,:]  #Discard alternatives except first k
    newX = X.copy()
    for i in range(N):
        rankingorder = letters_to_indices(Y[i].item())
         #Discard alternatives except first k
        rankingorder = rankingorder[:k]
        sorted_lst = sorted(rankingorder)
        label_map = {num: i for i, num in enumerate(sorted_lst)}
        rankingorder = [label_map[num] for num in rankingorder]
        newX[:,:,i] = np.take(X[:,:,i], rankingorder, axis=0) #Reorder alternatives to match ranking order
    return newX