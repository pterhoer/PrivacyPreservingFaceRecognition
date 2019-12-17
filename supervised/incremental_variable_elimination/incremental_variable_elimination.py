# Implementation of the supervised Incremental Variable Elimination approach to 
# suppress gender and age in face templates
# 
# For further information please refer to:
# "Suppressing Gender and Age in Face Templates Using Incremental Variable Elimination" by
# Philipp Terhörst, Naser Damer, Florian Kirchbuchner and Arjan Kuijper,
# International Conference on Biometrics (ICB), 2019
# 


import numpy as np

class IncrementalVariableElimination():
    
    def __init__(self, model_train, num_eliminations_perStep, num_steps):
        """
            model_train: sklearn model that contains .feature_importance
            num_eliminations_perSteps: number of variable eliminations in each step
            num_steps: number of iterations of eliminations
            
            num_steps and num_eliminations_perSteps have to be adjusted to the
            feature sizes
        """
        
        self.__model_train = model_train
        self.__num_eliminations_perStep = num_eliminations_perStep
        self.__num_steps = num_steps
        
    def fit(self, X, y_att):
        """ Model is trained to determine the feature mask to filter the variables
            that contain the most information regrading y_att.
            
            calculates the list of features masks
        
            X: array of (n_samples, n_features)
            y: target values / class labels, to be suppressed
        """
        self.__num_features = X.shape[1]
        self.__mask_list = []
        self.__calculateMaskList(X, y_att)
    
    def __calculateMask(self, X, y):
        """ Calculates the feature mask depending on the feature importance
        
            X: array of (n_samples, n_features)
            y: target values / class labels
        """
        
        model = self.__model_train
        
        # train model
        model.fit(X, y)
        
        # calculate feature importance
        f_imp = model.feature_importances_
        f_imp_argsort = np.argsort(f_imp)[::-1]

        # get list of values to remove
        thr_list = f_imp_argsort[0:self.__num_eliminations_perStep]

        # define mask
        mask = np.full(f_imp.shape, True)
        for thr in thr_list: 
            mask[thr] = False
      
        return mask
    
    def __calculateMaskList(self, X, y):
        """ Calculates the list of features masks
        
            X: array of (n_samples, n_features)
            y: target values / class labels
        """
        
        mask_list = [np.ones(self.__num_features, dtype="bool")]
        for cur_step in range(self.__num_steps):
            X_red = self.__transform(X, mask_list)
            mask = self.__calculateMask(X_red, y)
            mask_list.append(mask)
        self.__mask_list = mask_list
        
    def __transform(self, X, mask_list):
        """ Private method used to transform the data during the calculation of the mask
        
            X: array of (n_samples, n_features), to be transformed
            mask_list: current list of masks
        """

        for mask in mask_list:
            X = X[:,mask]
        return X
    
    def transform(self, X):
        """ Transform the given array of samples and features into the incremental
            variable eliminated form. .fit(X, y_att) has to be called first.
        
            X: array of (n_samples, n_features), to be transformed
        """
        
        for mask in self.__mask_list:
            X = X[:, mask]
        return X