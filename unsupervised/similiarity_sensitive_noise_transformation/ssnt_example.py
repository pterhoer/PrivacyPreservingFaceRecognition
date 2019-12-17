# Example of similarity-sensitive noise transformation
# 
# For further information please refer to:
# "Unsupervised privacy-enhancement of face representations using 
# similarity-sensitive noise transformations" by
# Philipp Terhörst, Naser Damer, Florian Kirchbuchner and Arjan Kuijper,
# Applied Intelligence, 2018
#

import numpy as np

from similarity_sensitive_noise_transformation import SimilaritySensitiveNoiseTransformation as SSNT

# load sample templates
genuine = np.load("genuine_features.npy")
imposter = np.load("imposter_features.npy")

# Similarity Sensitive Noise Transformation
ssnt = SSNT()

# cosine-sensitive noise transformation
cos_gen = ssnt.csn_transform_matrix(genuine, 0.8)
cos_imp = ssnt.csn_transform_matrix(imposter, 0.8)

# euclidean-sensitive noise transformation
euc_gen = ssnt.esn_transform_matrix(genuine, 0.5)
euc_imp = ssnt.esn_transform_matrix(imposter, 0.5)


# Output
print("Cosine-sensitive noise transformation:")
print("="*30)

print("Cosine similarity of genuine:")
print(ssnt.cos_sim(cos_gen[0], cos_gen[1]))
print(ssnt.cos_sim(cos_gen[1], cos_gen[2]))
print(ssnt.cos_sim(cos_gen[0], cos_gen[2]))
print()
print("Cosine similarity of imposter:")
print(ssnt.cos_sim(cos_gen[0], cos_imp[0]))
print(ssnt.cos_sim(cos_gen[0], cos_imp[1]))
print(ssnt.cos_sim(cos_gen[0], cos_imp[2]))

print()
print("Euclidian-sensitive noise Transformation:")
print("="*30)

print("Cosine similarity of genuine:")
print(ssnt.cos_sim(euc_gen[0], euc_gen[1]))
print(ssnt.cos_sim(euc_gen[1], euc_gen[2]))
print(ssnt.cos_sim(euc_gen[0], euc_gen[2]))
print()
print("Cosine similarity of imposter:")
print(ssnt.cos_sim(euc_gen[0], euc_imp[0]))
print(ssnt.cos_sim(euc_gen[0], euc_imp[1]))
print(ssnt.cos_sim(euc_gen[0], euc_imp[2]))
