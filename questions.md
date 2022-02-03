**Table of Contents**:
- [K-Nearest Neighbors](#k-nearest-neighbors)
- [Decision Trees](#decision-trees)
- [Linear regression](#linear-regression)
- [Dimensionality reduction](#dimensionality-reduction)
- [NLP](#nlp)
  - [word2vec](#word2vec)


## K-Nearest Neighbors
 * What is kNN? 
 * Can it be trained? If no, why?
 * How does it work?
 * What is majority vote?
 * What are the main parameters for kNN Classifier? 
      * n_neighbors
      * metric
      * algorithm
      * weights
 * What metrics can be used for determining distance?
     * Euclidean
     * Manhattan
     * Minkowski
     * Chebyshev
     * Mahalanobis
 * How to choose number of neighbors?
 * Should we standardize the data?
 * What is computational complexity of classifying new examples?
 * Does kNN suffer from the curse of dimensionality and if it does how so?
 * What are advantages and disadvantages of kNN?

## Decision Trees
 * What are Decision Trees?
 * How to build a tree?
 * What is impurity criterion (quality criterion)?
     * For classification
         * Gini (Gini impurity/uncertainty)
         * Entropy (information gain)
     * For regression
 * How a decision tree works with numerical features (chooses which splits to check)?
 * What are the main parameters for Decision Tree? 
     * max_depth
     * max_features
     * min_samples_leaf
 * How to deal with overfitting in decision trees?
 * In what situation trees can be built to the maximum depth?
 * What are advantages and disadvantages of decision trees?

## Linear regression
 * What is Linear Regression?
 * Which methods are used for optimization?
     * When is it suitable to use quantitative approach (gradient descent) instead of analytical solution?
     * What is the analytical solution? How to derive the formula?
     * What is gradient descent, and how can it be used to optimize parameters?
 * Which metrics are used to evaluate the performance of a regression model?
     * Describe MAE, MSE, RMSE, MAPE, MedAE
 * What is $R^2$ (coefficient of determination)?
     * In which case $R^2$ can be lower than 0?
 * What is regularization? And why is it needed?
     * What is Lasso?
     * What is Ridge?
     * What is the difference between Lasso and Ridge?
     * What is Elastic Net?
     * How to find the regularization parameter in LASSO or Ridge or Elastic Net?
     * In what situations LASSO performs better than Ridge?

## Dimensionality reduction
 * What is dimensionality reduction?
 * Which techniques for dimensionality reduction do you know?
 * What are SVD and truncated SVD? What is the difference between them?

## NLP
 * What is embedding?

### word2vec
* 