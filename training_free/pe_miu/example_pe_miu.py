# Example for Privacy-Enhancing Face Recognition based on Minimum Information Units

# For further information please refer to:
# PE-MIU: A Training-Free Privacy-Enhancing Face Recognition Approach 
# Based on Minimum Information Units by
# Philipp Terhörst, Kevin Riehl, Naser Damer, Peter Rot, Blaz Bortolato, 
# Florian Kirchbuchner, Vitomir Struc and Arjan Kuijper (2020)
#

import numpy as np

from privacy_enhancing_miu import PrivacyEnhancingMIU


# Load data
genuine = np.load("genuine_features.npy")       # 3 genuine
imposter = np.load("imposter_features.npy")     # 3 imposter
comb  = np.concatenate((genuine, imposter), axis=0)

# PrivacyEnhancing with MIU
pemiu = PrivacyEnhancingMIU(block_size=32)

# shuffle each vector depending on the block size
alt = pemiu.shuffle(comb)

# reconstruct a genuine to reference vector alt[0]
rec_gen = pemiu.reconstruct(alt[0], alt[1])

# try to reconstruct a imposter to reference vector alt[0]
rec_imp = pemiu.reconstruct(alt[0], alt[4])

# compare genuine
print("Genuine Comparison - original", pemiu.cos_sim(genuine[0], genuine[1]))
print("Genuine Comparison - not reconstructed", pemiu.cos_sim(alt[0], alt[1]))
print("Genuine Comparison - reconstructed:", pemiu.cos_sim(alt[0], rec_gen))

# compare imposter
print("Imposter Comparison - original", pemiu.cos_sim(genuine[0], imposter[1]))
print("Imposter Comparison - not reconstructed", pemiu.cos_sim(alt[0], alt[4]))
print("Imposter Comparison - 'reconstrucetd':", pemiu.cos_sim(alt[0], rec_imp))