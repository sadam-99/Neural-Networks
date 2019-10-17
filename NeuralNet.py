#####################################################################################################################
#   Assignment 2, Neural Network Programming
#   This is a starter code in Python 3.6 for a 2-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   train - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   h1 - number of neurons in the first hidden layer
#   h2 - number of neurons in the second hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   w01, delta01, X01 - weights, updates and outputs for connection from layer 0 (input) to layer 1 (first hidden)
#   w12, delata12, X12 - weights, updates and outputs for connection from layer 1 (first hidden) to layer 2 (second hidden)
#   w23, delta23, X23 - weights, updates and outputs for connection from layer 2 (second hidden) to layer 3 (output layer)
#
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing


class NeuralNet:
    def __init__(self, train,activation_func, header = True, h1 = 4, h2 = 2):
        np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers

        # raw_input = pd.read_csv(train)
        raw_input = train
        # If any attribute is missing replacing it with 0
        raw_input = raw_input.replace('?',0)
        
        # train_dataset = self.preprocess(raw_input)
        train_dataset = raw_input
        ncols = len(train_dataset.columns)
        nrows = len(train_dataset.index)
        self.X = train_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        self.y = train_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        # TODO: Remember to implement the preprocess method
        # Preprocessing the data
        self.X = self.preprocess(self.X)
        # converting the string class labels to integral values.
        self.y = self.category_encoding(self.y)
        self.activation_func = activation_func
        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))
    #
    # TODO: I have coded the sigmoid activation function, you need to do the same for tanh and ReLu
    #

    def __activation(self, x, activation_func="sigmoid"):
        if activation_func == "sigmoid":
            self.__sigmoid(self, x)
        if activation_func == "tanh":
            self.__tanh(self, x)
        if activation_func == "relu":
            self.__relu(self, x)

    #
    # TODO: Define the function for tanh, ReLu and their derivatives
    #

    def __activation_derivative(self, x, activation_func="sigmoid"):
        if activation_func == "sigmoid":
            self.__sigmoid_derivative(self, x)
        if activation_func == "tanh":
            self.__tanh_derivative(self, x)
        if activation_func == "relu":
            self.__relu_derivative(self, x)

    def __sigmoid(self, x):
        x_ar = np.array(x,dtype=np.float32)
        return 1 / (1 + np.exp(-x_ar))
    
    def __tanh(self, x):
        x_ar = np.array(x,dtype=np.float32)
        return (np.exp(x_ar)-np.exp(-x_ar))/(np.exp(x_ar)+np.exp(-x_ar))


    def __relu(self, x):
        return np.maximum(0,x)
    
    
    # derivative of sigmoid function, indicates confidence about existing weight

    def __sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def __tanh_derivative(self, x):
        relu_der = 1 - np.power(x,2)
        return relu_der
    
    def __relu_derivative(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x
        # if x>0:
        #     return 1
        # else:
        #     return 0

    #
    # TODO: Write code for pre-processing the dataset, which would include standardization, normalization,
    #   categorical to numerical, etc
    #
    
    def preprocess(self, X):
        # This function includes standardization, normalization
        
        # standardized_X = preprocessing.StandardScaler().fit(X)
        # standardized_X.mean_                                     
        # standardized_X.scale_                                       
        # standardized_X.transform(X) 
        scaled_X = preprocessing.scale(X)
        norm_X = preprocessing.normalize(scaled_X,norm='l2')
        return norm_X
    
    
    def category_encoding(self,y):
        # This function inclues categorical to numerical encoding
        classes=np.unique(y)
        for j in range(len(classes)):
            for k in range(len(y)):
                if y[k]==classes[j]:
                    y[k]=j
        # print(y)
        return y


    

    # Below is the training function

    def train(self, max_iterations = 1000, learning_rate = 0.05):
        for iteration in range(max_iterations):
            out = self.forward_pass(self.activation_func)
            error = 0.5 * np.power((out - self.y), 2)
            self.backward_pass(out, self.activation_func)
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 = self.w23 + update_layer2
            self.w12 =self.w12 + update_layer1
            self.w01 =self.w01  +  update_input

        print("After " + str(max_iterations) + " iterations, the total training error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers)")
        print(self.w01)
        print(self.w12)
        print(self.w23)

    def forward_pass(self, activation_func):
        # pass our inputs through our neural network
        in1 = np.dot(self.X, self.w01 )
        if activation_func == 'sigmoid':
            self.X12 = self.__sigmoid(in1)
        if activation_func == 'tanh':
            self.X12 = self.__tanh(in1)
        if activation_func == 'relu':
            self.X12 = self.__relu(in1)
        
        in2 = np.dot(self.X12, self.w12)
        if activation_func == 'sigmoid':
            self.X23 = self.__sigmoid(in2)
        if activation_func == 'tanh':
            self.X23 = self.__tanh(in2)
        if activation_func == 'relu':
            self.X23 = self.__relu(in2)
        in3 = np.dot(self.X23, self.w23)
        if activation_func == 'sigmoid':
            out = self.__sigmoid(in3)
        if activation_func == 'tanh':
            out = self.__tanh(in3)
        if activation_func == 'relu':
            out = self.__relu(in3)
        return out



    def backward_pass(self, out, activation_func):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation_func)
        self.compute_hidden_layer2_delta(activation_func)
        self.compute_hidden_layer1_delta(activation_func)

    # TODO: Implement other activation functions

    def compute_output_delta(self, out, activation_func="sigmoid"):
        if activation_func == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
        if activation_func == "tanh":
            delta_output = (self.y - out) * (self.__tanh_derivative(out))
        if activation_func == "relu":
            delta_output = (self.y - out) * (self.__relu_derivative(out))

        self.deltaOut = delta_output

    # TODO: Implement other activation functions

    def compute_hidden_layer2_delta(self, activation_func="sigmoid"):
        if activation_func == "sigmoid":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))
        if activation_func == "tanh":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__tanh_derivative(self.X23))
        if activation_func == "relu":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__relu_derivative(self.X23))

        self.delta23 = delta_hidden_layer2

    # TODO: Implement other activation functions

    def compute_hidden_layer1_delta(self, actactivation_funcivation="sigmoid"):
        if activation_func == "sigmoid":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
        if activation_func == "tanh":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__tanh_derivative(self.X12))
        if activation_func == "relu":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__relu_derivative(self.X12))

        self.delta12 = delta_hidden_layer1


    # TODO: Implement the predict function for applying the trained model on the  test dataset.
    # You can assume that the test dataset has the same format as the training dataset
    # You have to output the test error from this function

    def predict(self, test, header = True):
        # raw_input = pd.read_csv(test)
        raw_input = test
        # If any attribute is missing replacing it with 0
        raw_input = raw_input.replace('?',0)
        # dataset = self.preprocess(raw_input)
        ncols = len(raw_input.columns)
        nrows = len(raw_input.index)
        self.X = raw_input.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        self.y = raw_input.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        self.X = self.preprocess(self.X)
        self.y = self.category_encoding(self.y)
        output = self.forward_pass(self.activation_func)
        # predicted_class = output.argmax(axis=-1)
        test_err = 0.5 * np.sum(np.power((output - self.y), 2))
        return test_err


if __name__ == "__main__":
    # train_DF =pd.read_csv('train.csv')
    # test_DF =pd.read_csv('test.csv')
    # df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, names=['SepalL','SepalW','PetalL','PetalW','Class'])
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', header=None, names=['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class'])
    # df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data', header=None, names= ['Age of patient at time of operation', 'Patients year of operation','Number of positive axillary nodes detected' , 'Survival status'])
    # df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data', header=None)
    train_DF, test_DF = model_selection.train_test_split(df, test_size=0.2)
    # neural_network = NeuralNet(train_csv)
    
    
    activation_func= "sigmoid"
    learning_rate = 0.05
    max_iterations = 1000
    neural_network = NeuralNet(train_DF, activation_func)
    neural_network.train(max_iterations , learning_rate )
    testError = neural_network.predict(test_DF)
    print("The maximum number of iterations(Epochs) is =", max_iterations)
    print("The Activation Function is =", activation_func)
    print("The Learning Rate is =", learning_rate)
    print("------The Test Error is------- \n")
    print("The Test error is =", testError)

