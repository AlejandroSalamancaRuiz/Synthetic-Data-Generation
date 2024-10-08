# Synthetic Data Generation for Classification Problems

## Project Overview

This project explores and implements various techniques for generating synthetic data to address the issue of scarce datasets in classification problems. It compares the effectiveness of classical synthetic data generation methods, such as SMOTE and ADASYN, with a newly proposed GAN-based approach tailored for structured data. The primary focus is on generating synthetic categorical, numerical, and ordinal data to solve the heart disease classification problem using data from the CDC's 2020 survey.

## Dataset

The dataset used in this project comes from a cleaned version of the 2020 CDC survey on U.S. residents' health status, focusing on heart disease classification. It contains 401,958 rows and 279 columns, consisting of categorical, ordinal, and numerical variables.

The dataset has been preprocessed to handle categorical variables using one-hot encoding and to scale numerical values.

## Models and Methods

The project implements and compares the following techniques for synthetic data generation:

- SMOTE (Synthetic Minority Oversampling Technique): A widely used oversampling technique that generates new samples by interpolating between existing ones.
- ADASYN (Adaptive Synthetic Sampling Approach): An extension of SMOTE that adapts sample generation to focus on more difficult-to-classify cases.
- Borderline-SMOTE: Focuses on generating synthetic samples near the decision boundary.
- GAN (Generative Adversarial Networks): The newly proposed GAN architecture is designed to generate structured synthetic data. It consists of two neural networks:
    - Generator: Takes input from a latent space and generates structured data samples.
    - Discriminator: Trains to distinguish between real and fake data, helping improve the generator's output.

## Experimental Approach

A K-Nearest Neighbors (KNN) classifier with five neighbors is trained to predict heart disease based on synthetic datasets generated from various techniques. The classifier's performance is compared using accuracy as the evaluation metric. The project also explores how changes in the GAN architecture, dataset size, and latent space dimensions impact the quality of synthetic data.

### Steps:
- Data Preprocessing: The dataset is preprocessed using one-hot encoding for categorical/boolean variables and scaling for numerical variables.
- Classical Data Generation: Synthetic samples are generated using classical methods such as SMOTE, ADASYN, and Borderline-SMOTE.
- GAN-based Data Generation: The GAN model is trained and tested on different configurations to generate synthetic samples.
- Model Evaluation: The KNN model is trained on the generated synthetic data and tested against the original dataset to measure its performance.


## Results

The results show the accuracy of the KNN classifier using different synthetic data generation methods:

- SMOTE: Achieved accuracies up to 0.993 across different configurations.
- SVMSMOTE: Achieved accuracies up to 0.92.
- GAN: Showed varying performance, with accuracies between 0.13 and 0.92, depending on the configuration. While the GAN approach didn't consistently outperform the classical methods, it achieved promising results in several cases.

The experiments show that SMOTE consistently produced high-quality synthetic data across different dataset sizes, while the GAN approach remains unstable but has the potential to reach comparable performance in some configurations.

## Notebook

The Jupyter notebook titled Experiments.ipynb contains the full implementation of the project, including:

- Data Preprocessing: Steps to clean and preprocess the dataset.
- Model Training: Code for training the KNN classifier on the original and synthetic datasets.
- Synthetic Data Generation: Implementations of classical methods (SMOTE, ADASYN, etc.) and the proposed GAN architecture.
- Evaluation: Performance comparisons and accuracy results.

## Conclusion

The project concludes that while the GAN-based method shows potential, classical techniques like SMOTE are still highly effective for generating structured data for classification tasks. Further research into tuning the GAN model architecture and hyperparameters may yield better results in future iterations.

## References
- Pytlak, Kamil. "Personal Key Indicators of Heart Disease." Kaggle, 2022. Link
- Chawla, N. V., et al. "SMOTE: Synthetic Minority Over-sampling Technique." Journal of Artificial Intelligence Research, vol. 16, 2002, pp. 321-357.
- He, Haibo, et al. "ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning." 2008 IEEE International Joint Conference on Neural Networks, 2008, pp. 1322-1328.
- Goodfellow, Ian J., et al. "Generative Adversarial Nets." arXiv:1406.2661.