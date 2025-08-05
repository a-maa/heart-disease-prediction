import numpy as np

class CustomRegression():
    """
    Custom implementation of Linear Regression

    Parameters:
    learning_rate (float): The learning rate of the model
    n_iterations (int): The number of iterations the model will run

    Returns:
    None

    Explanation:
    - Weights: Each feature is associated with a weight. The model learns the weights that minimize the error or learns the weights that maximize the accuracy.
               The weights basically mean the contribution of each feature to the correct prediction. The higher the weight, the more important the feature is.
    - Bias: It's a value that accounts for the error in the prediction, which is a constant value.

    Both of these terms are learned during the training process. and are initialized in the following function as None.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000, type="linear") -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.type = type

    """
    This function is the sigmoid function, which is used to transform the linear model into a probability.

    Parameters:
    z (np.ndarray): The linear model

    Returns:
    np.ndarray: The probability
    """
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))    

    """
    This function is the main fit function of this class. It's responsible for fitting the model and predicting the target based on regression type.

    Parameters:
    x_train (np.ndarray): The training data
    y_train (np.ndarray): The target data

    Returns:
    self (CustomLinearRegression): The trained model
    """
    def fit(self, x_train, y_train):
        match self.type:
            case "linear":
                self._fit_linear(x_train=x_train, y_train=y_train)
            case "logistic":
                self._fit_logistic(x_train=x_train, y_train=y_train)

    """
    This function is the main predict function of this class. It's responsible for predicting the target based on regression type.

    Parameters:
    x (np.ndarray): The data to be predicted

    Returns:
    np.ndarray: The predicted target
    """
    def predict(self, x):
        match self.type:
            case "linear":
                return self._predict_linear(x)
            case "logistic":
                return self._predict_logistic(x)

    def _fit_linear(self, x_train, y_train):
        num_samples, num_features = x_train.shape
        self.weights = np.random.rand(num_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            y_prediction = np.dot(x_train, self.weights) + self.bias

            gradient_loss_function_weights = (1 / num_samples) * np.dot(x_train.T, y_prediction - y_train)
            gradient_loss_function_bias = (1 / num_samples) * np.sum(y_prediction - y_train)

            self.weights = self.weights - self.learning_rate * gradient_loss_function_weights
            self.bias = self.bias - self.learning_rate * gradient_loss_function_bias

        return self 
    
    def _fit_logistic(self, x_train, y_train):
        num_samples, num_features = x_train.shape
        self.weights = np.random.rand(num_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            linear_model = np.dot(x_train, self.weights) + self.bias
            y_prediction = self.sigmoid(linear_model)

            gradient_loss_function_weights = (1 / num_samples) * np.dot(x_train.T, y_prediction - y_train)
            gradient_loss_function_bias = (1 / num_samples) * np.sum(y_prediction - y_train)

            self.weights = self.weights - self.learning_rate * gradient_loss_function_weights
            self.bias = self.bias - self.learning_rate * gradient_loss_function_bias

        return self

    def _predict_linear(self, x):
        return np.dot(x, self.weights) + self.bias
    
    def _predict_logistic(self, x):
        linear_model = np.dot(x, self.weights) + self.bias
        y_prediction = self.sigmoid(linear_model)

        return y_prediction
