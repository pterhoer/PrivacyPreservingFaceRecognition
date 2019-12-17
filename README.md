# Privacy-Preserving Face Recognition

***17.12.2019:*** _Two more methods are currently under review and will be added as soon as they are accepted._

This repository provides implementations of different supervised and unsupervised methods to enhance the soft-biometric privacy of face recognition models.

If you use this code for a scientific publication, we would appreciate citations of the associated papers.

For further questions, please contact the provided e-mail adresses in the papers.

## Incremental Variable Elimination (supervised)
* [Research Paper](https://www.researchgate.net/publication/334304317_Suppressing_Gender_and_Age_in_Face_Templates_Using_Incremental_Variable_Elimination)
* [Implementation](./supervised/incremental_variable_elimination/incremental_variable_elimination.py)
* [Example](./supervised/incremental_variable_elimination/ive_example.py)

**Suppressing Gender and Age in Face Templates Using Incremental Variable Elimination**, International Conference on Biometrics (ICB), 2019

***Abstract***

Recent research on soft-biometrics showed that more information than just the person’s identity can be deduced from biometric data. Using face templates only, information about gender, age, ethnicity, health state of the person, and even the sexual orientation can be automatically obtained. Since for most applications these templates are expected to be used for recognition purposes only, this raises major privacy issues. Previous work addressed this problem purely on image level regarding function creep attackers without knowledge about the systems privacy mechanism. In this work, we propose a soft-biometric privacy enhancing approach that reduces a given biometric template by eliminating its most important variables for predicting soft-biometric attributes. Training a decision tree ensemble allows deriving a variable importance measure that is used to incrementally eliminate variables that allow predicting sensitive attributes. Unlike previous work, we consider a scenario of function creep attackers with explicit knowledge about the privacy mechanism and evaluated our approach on a publicly available database. The experiments were conducted to eight baseline solutions. The results showed that in many cases IVE is able to suppress gender and age to a high degree with a negligible loss of the templates recognition ability. Contrary to previous work, which is limited to the suppression of binary (gender) attributes, IVE is able, by design, to suppress binary, categorical, and continuous attributes.


```
@inproceedings{icb2019_PT,
  author    = {Philipp Terh{\"{o}}rst and
               Naser Damer and
               Florian Kirchbuchner and
               Arjan Kuijper},
  title     = {Suppressing Gender and Age in Face Templates Using Incremental Variable Elimination},
  booktitle = {2019 International Conference on Biometrics, {ICB} 2019, Crete,
               Greece, June 4-7, 2019},
  publisher = {{IEEE}},
  year      = {2019},
}
```

## Similarity-sensitive Noise Transformations (unsupervised)
* [Research Paper (Springer)](https://link.springer.com/article/10.1007/s10489-019-01432-5)
* [Research Paper (ResearchGate)](https://www.researchgate.net/publication/331307058_Unsupervised_privacy-enhancement_of_face_representations_using_similarity-sensitive_noise_transformations)
* [Implementation](./unsupervised/similiarity_sensitive_noise_transformation/similarity_sensitive_noise_transformation.py)
* [Example](./unsupervised/similiarity_sensitive_noise_transformation/ssnt_example.py)

**Unsupervised Privacy-enhancement of Face Representations Using Similarity-sensitive Noise Transformations**, Applied Intelligence, 2018

***Abstract***

Face images processed by a biometric system are expected to be used for recognition purposes only. 
However, recent work presented possibilities for automatically deducing additional information about an
individual from their face data. By using soft-biometric estimators, information about gender, age, ethnicity, sexual orientation or the health state of a person can be obtained. This raises 
a major privacy issue. Previous works presented supervised solutions that require large amount of private data in order to suppress a single attribute. In this work, we propose a privacy-preserving solution that does not require these sensitive information and thus, works in an unsupervised manner. 
Further, our approach offers privacy protection that is not limited to a single known binary attribute or classiﬁer. 
We do that by proposing similarity-sensitive noise transformations and investigate their effect and the effect of 
dimensionality reduction methods on the task of privacy preservation. 
Experiments are done on a publicly available database and contain analyses of the recognition performance, 
as well as investigations of the estimation performance of the binary attribute of gender and the continuous attribute of age. 
We further investigated the estimation performance of these attributes when the prior knowledge about the used privacy mechanism is explicitly utilized.
The results show that using this information leads to signiﬁcantly enhancement of the estimation quality. Finally, we proposed a metric to evaluate the trade-off between the privacy gain 
and the recognition loss for privacy-preservation techniques. Our experiments showed that the proposed cosine-sensitive noise transformation was successful in reducing the possibility 
of estimating the soft private information in the data, while having signiﬁcantly smaller effect on the intended recognition performance.

```
@article{DBLP:journals/apin/TerhorstDKK19,
  author    = {Philipp Terh{\"{o}}rst and
               Naser Damer and
               Florian Kirchbuchner and
               Arjan Kuijper},
  title     = {Unsupervised privacy-enhancement of face representations using similarity-sensitive
               noise transformations},
  journal   = {Appl. Intell.},
  volume    = {49},
  number    = {8},
  pages     = {3043--3060},
  year      = {2019},
  url       = {https://doi.org/10.1007/s10489-019-01432-5},
  doi       = {10.1007/s10489-019-01432-5},
  timestamp = {Wed, 25 Sep 2019 17:51:08 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/apin/TerhorstDKK19},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Acknowledgement
 
This work was supported by the German Federal Ministry of Education and Research (BMBF) as well as by the Hessen State Ministry for Higher Education, Research and the Arts (HMWK) within the National Research Center for Applied Cybersecurity ATHENE.

Portions of the research came from the Software Campus project supported by the German Federal Ministry of Education and Research (BMBF).

## Licence 

This project is licensed under the terms of the MIT license.
Copyright (c) 2019 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt
