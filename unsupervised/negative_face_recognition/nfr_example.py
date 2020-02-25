# Example for Negative Face Recognition
# 
# For further information please refer to:
# "Unsupervised Enhancement of Soft-biometric Privacy with Negative Face Recognition" by
# Philipp Terhörst, Marco Huber, Naser Damer, Florian Kirchbuchner and Arjan Kuijper (2020)
#

import numpy as np
from keras.models import load_model

from net import net, get_representation_model
from negative_face_recognition import NegativeFaceRecognition

# Load Data
y_ID = np.load("sample_IDs.npy")
X = np.load("sample_features.npy")

# If necessary, you can train a new model that expand the feature space
# We recommend using feature sizes in the magnitude of order 1000 

# Expand templates

# num_IDs = len(np.unique(y_ID))
# model = net(num_IDs)
# model.fit(x=X, y=y_ID_t, batch_size=256, epochs=50, verbose=2)

# create templates
# get representation model ...
# rep_model = get_representation_model(model)

# ... or load
rep_model = load_model("representation_model.h5") 

# generate expanded template 
expanded_templates = rep_model.predict(X)

# Negative Face Recognition

bins = 4
nfr = NegativeFaceRecognition()

# get positive templates
pos_templates = nfr.get_positive_template(expanded_templates, bins)

# get negative templates
neg_templates = nfr.get_negative_template(expanded_templates, bins)

# calculate comparison score
score = nfr.pn_comparison_score(pos_templates[4], neg_templates[5])
print("Comparison Score - Genuine:", score)
score = nfr.pn_comparison_score(pos_templates[0], neg_templates[1])
print("Comparison Score - Imposter:", score)

