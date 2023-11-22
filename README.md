# CSE574_IntroToML_LogisticRegression
## Assignment 1

### Assignment Overview:
Assignment targets on implementation of linear models for regressions. It is basically 2 parts which concentrates on classification problems. That is to implement Logistic Regression using stochastic gradient descent. Further for the second part, it focuses on implementation of Neural Networks with two hidden layers with different regularization methods (l2, l1). The above implementations need to be executed over the give data set.
    
### Dataset:
The dataset given is Pima Indians Diabetes Database to predict the onset of diabetes based on diagnostic measures provided. Data is split into categories for training and testing of 614 for train size and 154 test sizes constituting 80% and 20% of data set.

### Implementation:
Implementation was done in Google Colab.
#### Part A: Implementation of Logistic Regression.
***

Logistic Regression is one of the simplest and commonly used ML algorithm approach for two-class classification. It can be employed on baseline models or on any binary classification problem. Logistic regression describes and estimates the relationship between one dependent binary variable and independent variables. The following are core concepts and implementation that are applied for training the model.

#### 1. Implementation
Dataset from csv is taken and implemented # Now since we want the valid and test size to be equal `(20% each of overall data)`. # we have to define `valid_size=0.5 (that is 50% of remaining data)`. A new class LogisticRegressionModel is introduced which internally implements all the following concepts to train and test the data. After initialization of the class, I called alog_fit method to validate the fit and predict method to trigger the classification model.

#### 2. Sigmoid Function: 
Sigmoid function is used to classify the binary dataset. It can be shown as `sigmoid (z) = 1/(1/e^-z)` *where z is the linear classification function that needed to be applied for*. Hence the value is bound between 0 and 1 such that, if the resultant value is greater than 0.5, the output is displayed as 1 else 0.
Code Implementation

#### 3. Cost Function:
Cost function is basically the error displayed by the model with respect to the predicted value. To achieved good model, the cost function error rate must be reduced to lowest i.e., attaining global minima. 

#### 4. Forward and Back Propagation:
The weights and bias are to be updated to reduce the error rate. Hence, we need to use forward and backward propagation in the model to  

#### 5. Gradient Descent 
Training the model such that the weights and bias must be achieved to optimal value where the error rate is low and close to global minima. To achieve this, the formula for calculation of updated weights is `W = (Ypred - Y)*X`. Where Ypred is the predicted output and X is the given features in the input matrix. Which can be calculated by applying sigmoid to the given function. And similarly, Bias can be calculated by `B = (Ypred - Y)`.

#### 6. Prediction:
I then called the model to predict the functions

#### 7. Visualization:
The following intermediatory outputs shows while training and testing the regressor classifier.
* Splitting data set for test and train
* Gradient descent. Reducing the error percent.
* Accuracy of the model.

#### 8. Conclusion:
> So I tried to implement regressor_2 to our test data with `learning_rate=0.01` with number of iterations of 1000, **I got an accuracy of 81.81%.**

### PART 2: Implementation of Neural Networks
***
   
* For Neural Networks upon implementing baseline mode with `2 hidden layers having 32 and 16 units`, I got the accuracy as `76.54%`.
* Adding l1 regularization increases the performance to `78.6`. Below depicted is the graph for accuracy model after adding l1 regularization.
            3.2.1. 
            3.2.2. Model accuracy with dropout
            3.2.3. 
            3.2.4. 




