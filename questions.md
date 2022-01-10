**Table of Contents**:
- [k-Nearest Neighbors](#k-nearest-neighbors)
- [Decision Trees](#decision-trees)
- [Linear regression](#linear-regression)
- [Dimensionality reduction](#dimensionality-reduction)
- [NLP](#nlp)


## k-Nearest Neighbors
 - What is kNN? 
 - Can it be trained? If no, why?
 - How does it work?
 - What is majority vote?
 - What are the main parameters for kNN Classifier? 
      - n_neighbors
      - metric
      - algorithm
      - weights
 - What metrics can be used for determining distance?
     - Euclidean
     - Manhattan
     - Minkowski
     - Chebyshev
     - Mahalanobis
 - How to choose number of neighbors?
 <!-- Using hyper-parameter search and cross-validation to avoid over-fitting and under-fitting -->
 - Should we standardize the data?
 <!-- yes, so that each feature contributes equally to the distance -->
 - What is computational complexity of classifying new examples?
  <!-- grows linearly with the number of examples in the training dataset in the worst-case scenario -->
 - Does kNN suffer from the curse of dimensionality and if it does how so?
 - What are advantages and disadvantages of kNN?

## Decision Trees
 - What are Decision Trees?
 - How to build a tree?
 - What is impurity criterion (quality criterion)?
     - For classification
         - Gini (Gini impurity/uncertainty)
         - Entropy (information gain)
     - For regression
 - How a decision tree works with numerical features (chooses which splits to check)?
 - What are the main parameters for Decision Tree? 
     - max_depth <!-- the maximum depth of the tree -->
     - max_features <!-- the maximum number of features with which to search for the best partition (this is necessary with a large number of features because it would be "expensive" to search for partitions for all features) -->
     - min_samples_leaf <!-- the minimum number of samples in a leaf. This parameter prevents creating trees where any leaf would have only a few members. -->
 - How to deal with overfitting in decision trees?
 <!-- artificial limitation of the depth or a minimum number of samples in the leaves: the construction of a tree just stops at some point;
pruning the tree. -->
 - In what situation trees can be built to the maximum depth?
 <!-- 
Random Forest (a group of trees) averages the responses from individual trees that are built to the maximum depth (we will talk later on why you should do this)
Pruning trees. In this approach, the tree is first constructed to the maximum depth. Then, from the bottom up, some nodes of the tree are removed by comparing the quality of the tree with and without that partition (comparison is performed using cross-validation, more on this below). -->
 - What are advantages and disadvantages of decision trees?

## Linear regression
 - What is Linear Regression?
 - Which methods are used for optimization?
     - When is it suitable to use quantitative approach (gradient descent) instead of analytical solution?
     - What is analytical solution? How to derive the formula?
     - What is gradient descent and how can it be used to optimize parameters?
 - Which metrics are used to evaluate the performance of a regression model?
     - Describe MAE, MSE, RMSE, MAPE, MedAE
 - What is $R^2$ (coefficient of determination)?
     - In which case $R^2$ can be lower than 0?
 - What is regularization? And why is it needed?
     - What is Lasso?
     - What is Ridge?
     - What is the difference between Lasso and Ridge?
     - What is Elastic Net?
     - How to find the regularization parameter in LASSO or Ridge or Elastic Net?
     - In what situations LASSO performs better than Ridge?

## Dimensionality reduction
 - What is dimensionality reduction?
 - Which techniques for dimensionality reduction do you know?
 - What are SVD and truncated SVD? What is the difference between them?

## NLP
 - What is embedding?
 - 
