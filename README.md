# Privacy-Preserving Face Recognition

***03.06.2020:*** _Privacy-Enhancement based on Minimum Information Units (PE-MIU) was added._

***25.02.2020:*** _Negative Face Recognition was added._

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

## Similarity-sensitive Noise Transformations (training-free)
* [Research Paper (Springer)](https://link.springer.com/article/10.1007/s10489-019-01432-5)
* [Research Paper (ResearchGate)](https://www.researchgate.net/publication/331307058_Unsupervised_privacy-enhancement_of_face_representations_using_similarity-sensitive_noise_transformations)
* [Implementation](./training_free/similiarity_sensitive_noise_transformation/similarity_sensitive_noise_transformation.py)
* [Example](./training_free/similiarity_sensitive_noise_transformation/ssnt_example.py)

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

## Negative Face Recognition (unsupervised)
* [Research Paper (arXiv)](https://arxiv.org/abs/2002.09181)
* [Implementation](./unsupervised/negative_face_recognition/negative_face_recognition.py)
* [Example](./unsupervised/negative_face_recognition/nfr_example.py)

**Unsupervised Enhancement of Soft-biometric Privacy with Negative Face Recognition**, 2020 

***Abstract*** 

Current research on soft-biometrics showed that privacy-sensitive information can be deduced from biometric templates of an individual. Since for many applications, these templates are expected to be used for recognition purposes only, this raises major privacy issues. Previous works focused on supervised privacy-enhancing solutions that require privacy-sensitive information about individuals and limit their application to the suppression of single and pre-defined attributes. Consequently, they do not take into account attributes that are not considered in the training. In this work, we present Negative Face Recognition (NFR), a novel face recognition approach that enhances the soft-biometric privacy on the template-level by representing face templates in a complementary (negative) domain. While ordinary templates characterize facial properties of an individual, negative templates describe facial properties that does not exist for this individual. This suppresses privacy-sensitive information from stored templates. Experiments are conducted on two publicly available datasets captured under controlled and uncontrolled scenarios on three privacy-sensitive attributes. The experiments demonstrate that our proposed approach reaches higher suppression rates than previous work, while maintaining higher recognition performances as well. Unlike previous works, our approach does not require privacy-sensitive labels and offers a more comprehensive privacy-protection not limited to pre-defined attributes. 

```
@misc{terhrst2020unsupervised,
    title={Unsupervised Enhancement of Soft-biometric Privacy with Negative Face Recognition},
    author={Philipp Terhörst and Marco Huber and Naser Damer and Florian Kirchbuchner and Arjan Kuijper},
    year={2020},
    eprint={2002.09181},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```


## Privacy-Enhancing Face Recognition based on Minimum Information Units (training-free)
* [Research Paper (IEEE)](https://ieeexplore.ieee.org/document/9094207)
* [Research Paper (ResearchGate)](https://www.researchgate.net/publication/341420117_PE-MIU_A_Training-Free_Privacy-Enhancing_Face_Recognition_Approach_Based_on_Minimum_Information_Units)
* [Implementation](./training_free/pe_miu/privacy_enhancing_miu.py)
* [Example](./training_free/pe_miu/example_pe_miu.py)

**PE-MIU: A Training-Free Privacy-Enhancing Face Recognition Approach Based on Minimum Information Units**, IEEE Access, 2020

***Abstract***

Research on soft-biometrics showed that privacy-sensitive information can be deduced from biometric data. Utilizing biometric templates only, information about a persons gender, age, ethnicity, sexual orientation, and health state can be deduced. For many applications, these templates are expected to be used for recognition purposes only. Thus, extracting this information raises major privacy issues. Previous work proposed two kinds of learning-based solutions for this problem. The first ones provide strong privacy-enhancements, but limited to pre-defined attributes. The second ones achieve more comprehensive but weaker privacy-improvements. In this work, we propose a Privacy-Enhancing face recognition approach based on Minimum Information Units (PE-MIU). PE-MIU, as we demonstrate in this work, is a privacy-enhancement approach for face recognition templates that achieves strong privacy-improvements and is not limited to pre-defined attributes. We exploit the structural differences between face recognition and facial attribute estimation by creating templates in a mixed representation of minimal information units. These representations contain pattern of privacy-sensitive attributes in a highly randomized form. Therefore, the estimation of these attributes becomes hard for function creep attacks. During verification, these units of a probe template are assigned to the units of a reference template by solving an optimal best-matching problem. This allows our approach to maintain a high recognition ability. The experiments are conducted on three publicly available datasets and with five state-of-the-art approaches. Moreover, we conduct the experiments simulating an attacker that knows and adapts to the systems privacy mechanism. The experiments demonstrate that PE-MIU is able to suppress privacy-sensitive information to a significantly higher degree than previous work in all investigated scenarios. At the same time, our solution is able to achieve a verification performance close to that of the unmodified recognition system. Unlike previous works, our approach offers a strong and comprehensive privacy-enhancement without the need of training.


```
@article{9094207,
  author={P. {Terhörst} and K. {Riehl} and N. {Damer} and P. {Rot} and B. {Bortolato} and F. {Kirchbuchner} and V. {Struc} and A. {Kuijper}},
  journal={IEEE Access}, 
  title={{PE-MIU}: A Training-Free Privacy-Enhancing Face Recognition Approach Based on Minimum Information Units}, 
  year={2020},
  volume={8},
  number={},
  pages={93635-93647}
}

```
## Acknowledgement
 
This work was supported by the German Federal Ministry of Education and Research (BMBF) as well as by the Hessen State Ministry for Higher Education, Research and the Arts (HMWK) within the National Research Center for Applied Cybersecurity ATHENE.

Portions of the research came from the Software Campus project supported by the German Federal Ministry of Education and Research (BMBF).

## Licence 

This project is licensed under the terms of the MIT license.
Copyright (c) 2019 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt
