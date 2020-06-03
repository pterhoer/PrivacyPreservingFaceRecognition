# Implementation of the Negative Face Recognition approach to 
# enhance privacy of face recognition
# 
# For further information please refer to:
# PE-MIU: A Training-Free Privacy-Enhancing Face Recognition Approach 
# Based on Minimum Information Units by
# Philipp Terhörst, Kevin Riehl, Naser Damer, Peter Rot, Blaz Bortolato, 
# Florian Kirchbuchner, Vitomir Struc and Arjan Kuijper (2020)
#

import numpy as np
import random
import math

from scipy.optimize import linear_sum_assignment

class PrivacyEnhancingMIU:
    
    def __init__(self, block_size):
    
        self.block_size = block_size

    def __get_block(self, feature, index):
        """ returns the i-th block of a fearure vector, depending on the 
        provided block size
        
        feature: array, feature vector
        index: index of the block to be returned
        
        """
        startIndex = self.block_size * index
        endIndex = startIndex + self.block_size
        return feature[startIndex : endIndex]
            
    def shuffle(self, features):
        """ returns blockwise shuffle of the feature vectors
        
        features: array of feature vectors, (n_samples, n_features)
        """
        
        X_alt = np.copy(features)
        for x in range(0, len(X_alt)):
            X_alt[x] = self.__shuffle_feature(X_alt[x])
        return X_alt
    
    
    def __shuffle_feature(self, vec):
        """ blockwise shuffle of a single feature vector
        
        vec: feature vector (n_features)
        """
        
        vec_alt = np.copy(vec)
        blocks =  [vec_alt[i:i + self.block_size] for i in range(0,len(vec_alt), self.block_size)]
        random.shuffle(blocks)
        vec_alt = [b for bs in blocks for b in bs]
        return vec_alt
    
    
    def __create_cost_matrix(self, vec_1, vec_2, num_blocks):
        """ calculates the cost matrix for the given feature vectors
        
        vec_1: feature vector (n_features)
        vec_2: feature vector (n_features)
        """
        
        C = np.zeros((num_blocks,num_blocks))
        for x in range(0,num_blocks):
            for y in range(0,num_blocks):
                C[x][y] = self.__l2dist(self.__get_block(vec_1, x),self.__get_block(vec_2, y))
        return C
    
    def __sort(self, feature, num_blocks, assignment):
        """ sort blocks of the feature vector depending on the assignment.
       
        
        feature: feature vector, (n_features)
        assignment: index of blocks  
        """
        
        vec = np.zeros_like(feature)
        for x in range(0, num_blocks):
            part = self.__get_block(feature, assignment[x])
            for l in range(0, part.shape[0]):
                vec[x * self.block_size + l] = part[l]
        return vec
        
        
    
    def reconstruct(self, ref_vec, probe_vec):
        """ adjusts a given probe vector to a given reference vector
        
        ref_vec: reference vector, (n_features)
        probe_vec: probe vector, (n_features)
        
        returns an adjusted probe vector with its blocks adjusted to the 
        blocks of the reference vector, (n_features)
        
        """
        dim = ref_vec.shape[0]
        num_blocks = math.ceil(dim / self.block_size)
        cost_matrix = self.__create_cost_matrix(ref_vec, probe_vec, num_blocks)

        row_ind, assignment = linear_sum_assignment(cost_matrix)
        adj_vec = self.__sort(probe_vec, num_blocks, assignment)

        return adj_vec
    
    def cos_sim(self, a,b):
        """ calculates cosine similarity between vector a and b
        
        a: vector
        b: vector
        """
    
        a, b = a.reshape(-1), b.reshape(-1)
        return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))


    def __l2dist(self, a,b):
        """ calculates euclidean distance between vector a and b
        
        a: vector
        b: vector
        """
        return np.linalg.norm(a-b)
    
