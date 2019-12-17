# Example of Incremental Variable Elimination
# 
# For further information please refer to:
# "Suppressing Gender and Age in Face Templates Using Incremental Variable Elimination" by
# Philipp Terhörst, Naser Damer, Florian Kirchbuchner and Arjan Kuijper,
# International Conference on Biometrics (ICB), 2019
# 

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from incremental_variable_elimination import IncrementalVariableElimination as IVE

# load data
# in this example we use only a small subset
X = np.load("sample_features.npy")
Y = np.load("sample_gender_labels.npy")

# feature normalization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# define classifier
model_train = RandomForestClassifier(n_estimators=30)

# define params
# number of steps and number of eliminations per step have to be
# adjusted depending on the feature size
num_steps = 20
num_eliminations = 5

# init, fit and transform
ive = IVE(model_train, num_eliminations, num_steps)
ive.fit(X,Y)
X_new = ive.transform(X)


