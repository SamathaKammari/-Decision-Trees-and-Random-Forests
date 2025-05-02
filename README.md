https://towardsdatascience.com/decision-tree-and-random-forest-explained-8d20ddabc9dd/
Decision trees and random forests are supervised learning algorithms used for both classification and regression problems. These two algorithms are best explained together because random forests are a bunch of decision trees combined. There are ofcourse certain dynamics and parameters to consider when creating and combining decision trees. In this post, I will explain how decision trees and random forests work as well as the critical points to consider when using these models.

A decision tree builds upon iteratively asking questions to partition data. It is easier to conceptualize the partitioning data with a visual representation of a decision tree:

Figure source
Figure source
This represents a decision tree to classifiy animals. First split is based on the size of animal. Although the question seems to be "How big is the animal?", it is asked in the form of "Is the animal bigger than 1m?" because we want to split the data points in two groups at each step. The questions get more specific as the tree gets deeper.

What you ask at each step is the most critical part and greatly influences the performance of decision trees. For example, assume your dataset has "feature A" ranging from 0 to 100 but most of the values are above 90. In this case, the first question to ask is "Is feature A more than 90?". It does not make sense to ask "Is feature A more than 50?" because it will not give us much information about the dataset.

Our aim is to increase the predictiveness of the model as much as possible at each partitioning so that the model keeps gaining information about the dataset. Randomly splitting the features does not usually give us valuable insight about the dataset. Splits that increase purity of nodes are more informative. The purity of a node is inversely proportional to the distribution of different classes in that node. The questions to ask are chosen in a way that increases purity or decrease impurity.

There are two ways to measure the quality of a split: Gini Impurity and Entropy. They essentially measure the impurity and give similar results. Scikit-learn uses gini index by default but you can change it to entropy using criterion parameter.

Gini Impurity

As stated on wikipedia, "Gini impurity is a measure of how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset". It basically means that impurity increases with randomness. For instance, let’s say we have a box with ten balls in it. If all the balls are same color, we have no randomness and impurity is zero. However, if we have 5 blue balls and 5 red balls, impurity is 1.

Entropy and Information Gain

Entropy is a measure of uncertainty or randomness. The more randomness a variable has, the higher the entropy is. The variables with uniform distribution have the highest entropy. For example, rolling a fair dice has 6 possible outcomes with equal probabilities so it has a uniform distribution and high entropy.

Entropy vs Randomness
Entropy vs Randomness
Splits that result in more pure nodes are chosen. All these indicate "information gain" which is basically the difference between entropy before and after the split.


When choosing a feature to split, decision tree algorithm tries to achieve

More predictiveness
Less impurity
Lower entropy
Now we have an understanding of how the questions (or splits) are chosen. The next topic is the number of questions. How many questions do we ask? When do we stop? When is our tree sufficient to solve our classification problem? The answer for all these questions lead us to the one of most important concept of Machine Learning: overfitting. The model can keep asking questions until all the nodes are pure. Pure nodes include data points from only one class.


The model can keep asking questions (or splitting data) until all the leaf nodes are pure. However, this would be a too specific model and would not generalize well. It achieves high accuracy with training set but performs poorly on new, previously unseen data points.

As you can see in the visualization below, at a depth of 5, the model clearly overfits. The narrow regions between classes might be due to outliers or noise.


Depth = 5 (Figure source)
Depth = 5 (Figure source)
It is very important to control or limit the depth of a tree to prevent overfitting. Scikit-learn provides hyperparameters to control the structure of decision trees:

max_depth: The maximum depth of a tree. Depth of a tree starts from 0 (i.e. the depth on root node is zero). If not specified, the model keeps splitting until all leaves are pure or until all leaves contain less than min_samples_split samples.

min_samples_split: The minimum number of samples required to split an internal node. The algorithm keeps splitting the nodes as long as a node has more samples (data points) than the number specified with min_samples_split parameter.

min_impurity_decrease: The aim when doing a split is to reduce impurity (or uncertainty) but not all splits equally achieve this. This parameter sets a threshold to make a split. A node will be split if this split induces a decrease of the impurity greater than or equal to threshold value.

You can see the list of all hyperparametes of DecisionTreeClassifier() here.

Random Forests
Random Forest is an ensemble of many decision trees. Random forests are built using a method called bagging in which each decision trees are used as parallel estimators. If used for a classification problem, the result is based on majority vote of the results received from each decision tree. For regression, the prediction of a leaf node is the mean value of the target values in that leaf. Random forest regression takes mean value of the results from decision trees.

Random forests reduce the risk of overfitting and accuracy is much higher than a single decision tree. Furthermore, decision trees in a random forest run in parallel so that the time does not become a bottleneck.

The success of a random forest highly depends on using uncorrelated decision trees. If we use same or very similar trees, overall result will not be much different than the result of a single decision tree. Random forests achieve to have uncorrelated decision trees by bootstrapping and feature randomness.

Bootsrapping is randomly selecting samples from training data with replacement. They are called bootstrap samples. The following figure clearly explains this process:
Figure source
Figure source
Feature randomness is achieved by selecting features randomly for each decision tree in a random forest. The number of features used for each tree in a random forest can be controlled with max_features parameter.
Feature randomness
Feature randomness
Bootstrap samples and feature randomness provide the random forest model with uncorrelated trees.

There is an additional parameter introduced with random forests:

n_estimators: Represents the number of trees in a forest. To a certain degree, as the number of trees in a forest increase, the result gets better. However, after some point, adding additional trees do not improve the model. Please keep in mind that adding additional trees always mean more time for computation.

Pros and Cons
Decision Trees

Pros:

It is usually not needed to normalize or scale features
Suitable to work on a mixture of feature data types (continuous, categorical, binary)
Easy to interpret
Cons:

Prone to overfitting and need to be ensembled in order to generalize well
Random Forests

Pros:

A powerful, highly accurate model on many different problems
Like decision trees, does not require normalization or scaling
Like decision trees, can handle different feature types together
Runs the trees in parallel so the performance is not effected
Cons:

Not a good choice for high-dimensional data sets (i.e. text classification) compared fast linear models (i.e. Naive Bayes)
Example using Scikit-Learn
Decision trees and random forest can also be
