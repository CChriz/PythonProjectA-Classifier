import numpy as np


class MyClassifier:
    """
    binary classifier for spam implementing logistic regression
    """
    def __init__(self, l_rate=0.1):
        """
        initialise class with a constant learning rate of 0.1, this being the amount 
        of change to be applied when adjusting parameters - weights & bias, during gradient descent
        """
        self.l_rate = l_rate
        
    def sigmoid(self, features):
        """
        method to calculate results of the sigmoid function : 1 / (1 + e^-(weights . input features + bias))
        for all data points in given numpy array of features
        @param features : numpy array of features with shape (n_data, 54)
        @return : numpy array of results of the sigmoid function
        """
        return 1 / (1 + np.exp(-(np.dot(features, self.weights) + self.bias)))
    
    def train(self, train_data, train_labels):
        """
        method to train the model using gradient descent with the training data provided
        @param train_data : numpy array of features from training data with shape (n_data, 54)
        @param train_labels : numpy array of training data's labels with shape (n_data,)
        """
        # initialise parameters
        # initialise numpy array of 0s for weight of each feature
        self.weights = np.zeros(54)
        # initialise bias to 0
        self.bias = 0
        
        # retrieve size of training data
        n_data = train_labels.size
        
        # loop iterates 1500 times to fit the model to the training data using gradient descent to minimise the cost function
        # through adjusting to find the optimum parameters - weights and bias
        for i in range(0, 1500):
            # estimate probability of spam for all data points in the training dataset using the sigmoid function
            spam_probs = self.sigmoid(train_data)
            
            
            # partial derivative of cost function with respect to weights = 1/n_data * (P(train_data=spam) - train_labels).train_data
            # partial derivative of cost function with respect to bias = 1/n_data * (P(train_data=spam) - train_labels)
            distance = spam_probs - train_labels
            # transposed train data so that each feature & their weight is aligned correctly, new shape (54, n_data)
            pd_weight = (1 / n_data) * np.dot(train_data.T, (distance))
            # sum up all distances between estimated probability of spam with actual label
            pd_bias = (1 / n_data) * np.sum(distance)
            
            # new weights = old weights - learning rate x partial derivative of cost function with respect to weights
            # new bias = old bias - learning rate x partial derivative of cost function with respect to bias
            self.weights -= self.l_rate * pd_weight
            self.bias -= self.l_rate * pd_bias
            
            
    def predict(self, test_data):
        """
        predicts the labels of given data set
        @param test_data : numpy array of input dataset of features with shape (n_data, 54)
        @return : numpy array of predictions with shape (n_data,) of binary values - 0(ham) and 1(spam)
        """
        # estimate probability of spam for all datapoints in the test dataset
        spam_probs = self.sigmoid(test_data)
        # return rounded value of estimated probabiblity of spam - 0(ham) or 1(spam)
        return np.round(spam_probs)

training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",").astype(int)
spam_classifier = MyClassifier(l_rate=0.1)
spam_classifier.train(training_spam[:, 1:], training_spam[:, 0])
