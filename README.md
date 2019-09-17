# Machine-Learning

#### Learning to Rank using Linear Regression:
The aim of the project is to use machine learning approach to learn to assign ranks 0, 1 or 2 to documents based on relevance to the query. Given the target rank for the documents along with 46 features, the supervised learning approach of linear regression is used for learning to rank(LeToR). Using both closed form solution and Stochastic Gradient Discent (SGD), it is observed that, on tuning hyperparameters, SGD has better accuracy than closed form solution.

#### Handwriting Recognition:
The aim of the project is to use machine learning approaches to determine the given 2 handwriting samples belong to the same writer or not. For this, ”AND” images samples extracted from CEDAR Letter data set (both human observed data and GSC data) is used. 
3 machine learning approaches linear regression ,logistic regression and neural network are used for this case. Comparing all the models, it is observed that, neural network has maximum accuracy followed by logistic regression and linear regression.

#### Classification
The aim of the project is to use 4 machine learning methods multivariate logistic regression, neural network, SVM and Random forest for classification of images into digits that they represent. Then the ensemble of the prediction from the 4 models are used as the final decision. Data sets used are MNIST (training and validation) and USPS (testing).
The 4 classification models are implemented and also the ensemble of them is implemented with accuracy of 0.9021 on the MNIST data set and 0.31 on USPS test data set. 
It answers the NO FREE LUNCH theorem as the model built for MNIST is not efficient on USPS data set.

#### Tom and Jerry in Reinforcement learning
The key concept of the project is to combine deep learning and reinforcement learning. Here the task is to decide on the shortest path based on four possible actions from each state. It is achieved by initially randomly moving the agent around the grid to gain information about how the reward is affected based on actions from different starting states to their corresponding end states.This process is called exploration.
Then over time, instead of randomly choosing actions, the observations made earlier are used to estimate the Q-value for all possible actions from the current state to make an informed decision i.e.. to make the best action to move closer towards the goal. This process is called exploitation. It is implemented using neural network.
For this task, the environment is setup initially. Then the agent (Tom) is defined along with the starting state. 
The brain defines the neural network used and the Memory is used to store the observations.
