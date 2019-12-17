# Implementation of the unsupervised similarity-sensitive noise transformation 
# approach to enhance privacy of face representations
# 
# For further information please refer to:
# "Unsupervised privacy-enhancement of face representations using 
# similarity-sensitive noise transformations" by
# Philipp Terhörst, Naser Damer, Florian Kirchbuchner and Arjan Kuijper, 
# Applied Intelligence, 2018
#

import numpy as np
import random 

class SimilaritySensitiveNoiseTransformation:
    
    def csn_transform(self, v, similarity, len_intervall=(1,101), epsilon=0.0001):
        """ cosine sensitive noise transformation.
            returns an transformed vector x which has a cosine similarity of similarity to v
    
            v: numpy vector
            similarity: values in [-1,1]
            epsilon: defines the quality (lower values higher quality)
        """
        
        # dimensions
        size = len(v)
            
        # normalize vector
        v_norm = v / np.linalg.norm(v)
        quality = False
    
        while(quality==False):
            
            # create normalized random vector
            r = np.random.randn(size)
            r += v_norm
            r_norm = r / np.linalg.norm(r) # must be normalized
        
            # calculate coefficients
            O = similarity
            R = np.dot(r_norm, v_norm)
            a = self.__a(O,R)
            b = self.__b(a, R)
        
            # create new vector
            x = a * v_norm + b * r_norm
        
            # change length of the vector
            lengths = range(*len_intervall)
            length = random.sample(lengths, 1)
            x = length * x
        
            # compute and check similarity
            sim = self.cos_sim(x, v)
            if np.abs(sim - similarity) <= epsilon:
                quality = True
            
        return x
    
    
    def csn_transform_matrix(self, X, similarity, len_intervall=(1,101), epsilon=0.0001):
        """	Equal to cosine sensitive noise transformation, but for multiple samples (matrices) """
        
        m, n = X.shape
        X_noise = np.zeros((m,n))
        for i in range(m):
            v = X[i,:]
            v_noise = self.csn_transform(v, similarity, len_intervall, epsilon)
            X_noise[i,:] = v_noise
        return X_noise
    
    def esn_transform(self, v, distance):
        """  euclidean sensitive noise transformation.
             returns an transformed vector x which has a euclidean distance of distance to v
             
             v: numpy vector
             distance: float
        """
    
        size = len(v)
        x = np.random.normal(size=size) # x_n ~ N(0,1)
        r = np.linalg.norm(x)
        x_unit = 1./r * x
        return v + x_unit * distance
    
    def esn_transform_matrix(self, X, distance):
        """	Equal to euclidean sensitive noise transformation, but for multiple samples (matrices) """
           
        m, n = X.shape
        X_noise = np.zeros((m,n))
        for i in range(m):
            v = X[i,:]
            v_noise = self.esn_transform(v, distance)
            X_noise[i,:] = v_noise
        return X_noise
    
    def cos_sim(self, a,b):
        """ Calculates the cosine similarity between vector a and b """
        return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def __a(self, O, R, sign=+1):
        """ Calculate parameter a for x = a*v + b*r """
        s = O**2*R**4 - O**2*R**2 - R**4 + R**2
    
        if s <= 0:
            s = 0.000001
    
        return (sign*np.sqrt(s) + O*R**2 - O)/(R**2 -1)

    def __b(self, a, R, sign=+1):
        """ Calculate parameter b for x = a*v + b*r """
        s = a**2*R**2 - a**2 +1
        return sign*np.sqrt(s) - a * R

