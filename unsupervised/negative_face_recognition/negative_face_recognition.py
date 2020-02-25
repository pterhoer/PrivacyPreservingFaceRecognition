# Implementation of the Negative Face Recognition approach to 
# enhance privacy of face recognition
# 
# For further information please refer to:
# "Unsupervised Enhancement of Soft-biometric Privacy with Negative Face Recognition" by
# Philipp Terhörst, Marco Huber, Naser Damer, Florian Kirchbuchner and Arjan Kuijper (2020)
#

import random
import numpy as np

from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

class NegativeFaceRecognition:

    def __transform_random(self, features, bins):
        """ transform the given discrete vectors into a new random representation.
            No discrete value is retained.
        
            features: array (n_samples, n_features)
            bins: number of bins/discrete values
        """
        neg_templates = []
        for template in features:
            neg_t = []
            for f in template:
                bin_space = [x for x in list(range(0,bins)) if x != f]
                f = random.choice(bin_space)
                neg_t.append(f)
            neg_templates.append(neg_t)
        return np.asarray(neg_templates, dtype=int)
    
    def get_positive_template(self, features, bins):
        """ Returns the generated positive template
			
			features: array (n_samples, n_features)
			bins: number of bins/discrete values
		"""
        scaled_features = self.__scale(features)
        positive_template = self.__discretize(scaled_features, bins)
        return positive_template.astype(int)
        
    def get_negative_template(self, features, bins):
        """ Returns the generated negative template

			features: array (n_samples, n_features)
			bins: number of bins/discrete values
		"""
        scaled_features = self.__scale(features)
        positive_template = self.__discretize(scaled_features, bins)
        negative_template = self.__transform_random(positive_template, bins)
        return negative_template
    
    def pn_comparison_score(self, pos, neg):
        """ positive-negative comparison score:
            calculates and returns the dissimilarity between a positive 
            and a negative template that acts as a comparison score
    
            pos: positive template
            neg: negative template
        """
    
        parity = neg == pos
        parity_score = 0
        unique, counts = np.unique(parity, return_counts=True)
        for i in range(0,len(unique)):
            if unique[i] == True:
                parity_score = counts[i]
        sim = 1 - (parity_score/parity.size)
        return sim
    
    def __scale(self, features):
        """
            standardize features by removing the mean and scaling to unit variance
            
            features: array (n_samples, n_features)
        """
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        return features
    
    def __discretize(self, features, bins):
        """
            discretize the given features into bins intervals.
            
            features: array (n_samples, n_features)
            bins: number of bins/discrete values
        """
        discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
        discrete_features = discretizer.fit_transform(features)
        return discrete_features
    