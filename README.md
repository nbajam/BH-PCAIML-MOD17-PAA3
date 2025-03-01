# Berkeley Haas - Professional Certificate in MLAI - Practical Assignment2 - Comparing Classifiers

## Background
 This assignment aims to train a classification models and to compare the performance of the classifiers:
 (k-nearest neighbors, logistic regression, decision trees, and support vector machines) 
 
The dataset is related to the marketing of bank products over the telephone.
This project aims to train a classification model to predict if a user will accept a coupon given his/her answers to some survey questions.

The data used in here comes from [UCI Machine Learning repository] (https://archive.ics.uci.edu/ml/datasets/bank+marketing )

In this project the work I have done is split into 3 parts.

### Part 1 - Preprocessing of the data
During this data processing, I have worked to:

* Use semicolon as the separator to load the CSV. Dots within column names
* Checked Duplicated Data
* Checked for null or missing values
* Feature Encoding - using Label Encoding to convert Categorical features to numerical features
* Finding relationships between features correlation matrix

### Part 2 Modeling training

Here in Part 2, I have transformed the features to make the data suitable for creating and training models. 

#### Data Tranformation and Encoding
Here I am performing data preprocessing for machine learning pipeline using LabelTransformer from sklearn. 

Encoded categorical variables using LabelEncoding, converting the features to numerical features

Fit the transformations to X_train.

Used the same transformations learned from X_train and applies them to X_test (without refitting).

All categorical features were Label encoded.
With this the data is now ready for machine learning models that require numerical input.

### Numerical Features
    ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
       'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
       'previous', 'poutcome', 'emp_var_rate', 'cons_price_idx',
       'cons_conf_idx', 'euribor3m', 'nr_employed', 'y']

### Categorical Features

The total number of features after LabelEncoding did not change and stands at 21 features. 

#### Creation and running performance of different models

 I created and tested the model performance for the following models:
 
 * Logistic Regression
 * Lasso Regression
 * Decision Trees
 * Gaussian Naive Bayes
 * Bernoulli Naive Bayes
 * K-nearest Neighbor
 * Linear SVM - Support Vector Machines

### Part 3 Applying PCA - Principal Component Analysis to reduce components / input features

Finally I explored PCA - Principal Component Analysis, to reduce the dimensions or in other words the number of features. 

Applying PCA reduced the features from 21 to 15 most influential ones listed below:
PC1 is most influenced by: euribor3m with influence ration 0.21106144872320615
PC2 is most influenced by: pdays with influence ration 0.0856805587270519
PC3 is most influenced by: age with influence ration 0.07975607099291482
PC4 is most influenced by: month_encoded with influence ration 0.0666490304836804
PC5 is most influenced by: job_encoded with influence ration 0.05704420774124356
PC6 is most influenced by: campaign with influence ration 0.054264905909507125
PC7 is most influenced by: duration with influence ration 0.053805168658536066
PC8 is most influenced by: duration with influence ration 0.05069210434549991
PC9 is most influenced by: previous with influence ration 0.049655554188672936
PC10 is most influenced by: housing_encoded with influence ration 0.04691501422387794
PC11 is most influenced by: default_encoded with influence ration 0.04544547049483743
PC12 is most influenced by: job_encoded with influence ration 0.0444204355560462
PC13 is most influenced by: campaign with influence ration 0.042580264950899976
PC14 is most influenced by: education_encoded with influence ration 0.03641928640808503
PC15 is most influenced by: age with influence ration 0.02894594575095166

Re-trained the basic models after applying PCA, to explore any improvement in performance. 

### Part 4 Summary and Evaluation of Model performance

The following tables summarize the results of:
 Validation Accuracy
 Testing Accuracy
 AUC - Area Under the Curve of Models (with Data before and after PCA) 

From the tables below it clear that All 3 Logistic Regression based classifiers along with Linear SVM classifier perform quite better than the rest of the models.(in both Categories - i.e, with PCA and without PCA)

The performance of the models was slightly better without PCA. 
One thing to note though, the models took comparatively lesser time to train with PCA.  

| Models with out PCA | Validation Accuracy|Testing Accuracy|AUC|
|-------|---------|-------|-----------|
|**Logistic Regression with No Penalty**|**0.916**|**0.911**|**0.90**|
|**Lasso Logistic Regression**|**0.917**|**0.910**|**0.90**|
|**Ridge Logistic Regression**|**0.917**|**0.911**|**0.90**|
|Decision Tree|0.915|0.905|0.87|
|Gaussian Naive Bayes|0.853|0.840|0.82|
|Bernoulli Naive Bayes|0.830|0.826|0.80|
|KNN Classifier|0.905|0.915|0.85|
|**Linear SVM**|**0.914**|**0.917**|**0.90**|

| Models with PCA | Validation Accuracy|Testing Accuracy|AUC|
|-------|---------|-------|-----------|
|**Logistic Regression with No Penalty**|**0.911**|**0.911**|**0.90**|
|**Lasso Logistic Regression**|**0.912**|**0.911**|**0.90**|
|**Ridge Logistic Regression**|**0.912**|**0.911**|**0.90**|
|Decision Tree|0.903|0.905|0.84|
|Bernoulli Naive Bayes|0.890|0.908|0.87|
|KNN Classifier|0.902|0.911|0.85|
|**Linear SVM**|**0.910**|**0.915**|**0.90**|


## Link to Notebook:
[Explore Coupon Acceptance Factors by Customers](https://github.com/nbajam/BH-PCAIML-MOD17-PAA3/blob/main/bh_pcaiml_mod17_prac_assign.ipynb)

## Data Used:
[In-Vehicle Coupon Recommendation](https://archive.ics.uci.edu/dataset/603/in+vehicle+coupon+recommendation)

