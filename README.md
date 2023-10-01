# PythonProjectA-Classifier
# Email Spam Classifier

This repository contains code for an email spam classifier. In this README, I will provide an overview of the project and explain the approach taken to build the classifier.

## Approach

I initially approached this task by implementing a Naive Bayes solution as guided in a separate notebook file. However, I was not satisfied with the accuracy of my classifier. Therefore, I explored an alternative approach, which was briefly discussed in the lectures - Logistic Regression.

I chose to use Logistic Regression as my final approach due to its good performance in binary classification, which is suitable for this task since an email can only be classified as spam (label 1) or ham (label 0).

### Algorithm

Here is an overview of the algorithm used to train the model:

1. Initialize weights of all 54 features to 0.
2. Initialize bias to 0.
3. Iterate 1500 times:
   - For each email:
     - Estimate the probability of spam using the sigmoid function [1].
     - Calculate the distance between the estimated probability and the true label.
     - Update weights and bias using gradient descent [2].

I chose 1500 iterations for training because it provided good results during testing and helped avoid overfitting or underfitting the model.

### Equations

#### Sigmoid Function [1]

The sigmoid function estimates the probability of an email being spam:

```
Y_hat = 1 / (1 + e^-z)
```

If the result is greater than 0.5, the email is classified as spam (label 1); otherwise, it is classified as ham (label 0).

Here, `z` represents the linear combination of weights, features, and bias:

```
z = w.x + b
```

- `w`: Weights
- `x`: Features
- `b`: Bias

The number of weights is equal to the number of features (54), and the bias is a single real value.

#### Gradient Descent [2]

To minimize the loss function, I adjusted the weights and bias during training using gradient descent:

- Update weights:
  ```
  new weights = old weights - Learning rate * partial derivative of cost function with respect to weights
  ```

- Update bias:
  ```
  new bias = old bias - Learning rate * partial derivative of cost function with respect to bias
  ```

Where:

- Partial derivative of cost function with respect to weights:
  ```
  1 / datapoints * (distance between estimate and true label) . Features
  ```

- Partial derivative of cost function with respect to bias:
  ```
  1 / datapoints * (distance between estimate and true label)
  ```

I used a learning rate of 0.1 to ensure that parameter adjustments during training were not too large or too small.

## Usage

Once training is complete, the model can predict the labels of given data using the `predict` function. The `predict` function applies the sigmoid function to the input features and the tuned parameters (weights and bias). The output of this function is the probability that the sample is a spam email, which is then rounded to the nearest label: 1 (spam) or 0 (ham).

Feel free to use this code to build your own email spam classifier!
