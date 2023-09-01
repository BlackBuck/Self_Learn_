import random
import numpy
import scipy
class ann:
    input_nodes = 0 # the number of input nodes
    hidden_nodes = 0 # the number of hidden nodes
    output_nodes = 0 # the number of output nodes
    learning_rate = 0.0 # the learning rate
    
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate) -> None:
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        self.input = numpy.array([[0] for _ in range(self.input_nodes)])
        self.hidden = numpy.array([[0] for _ in range(self.hidden_nodes)])
        self.output = numpy.array([[0] for _ in range(self.output_nodes)])

        self.wih = numpy.random.normal(0.0, pow(self.hidden_nodes, -0.50), (self.hidden_nodes, self.input_nodes))
        self.who = numpy.random.normal(0.0, pow(self.output_nodes, -0.50), (self.output_nodes, self.hidden_nodes))

        self.input_bias = numpy.random.normal(0.0, pow(self.hidden_nodes, -0.50), (self.input_nodes, 1))
        self.hidden_bias = numpy.random.normal(0.0, pow(self.output_nodes, -0.50), (self.hidden_nodes, 1))

        self.activation_function = lambda x : scipy.special.expit(x)
    
 
    def train(self, output):
        # the errors in the output nodes (ej for the hidden layer)
        output = numpy.array([[elem] for elem in output], ndmin=2)
        err_output = numpy.subtract(output, self.output)

        #del w = -ej*sigmoid(sum(wjk . oj))*(1 - sigmoid(sum(wjk . oj)))*oj
        del_w = numpy.dot(self.who.copy(), self.hidden)
        for i in range(del_w.shape[0]):
            del_w[i][0] = self.activation_function(del_w[i][0])*(1 - self.activation_function(del_w[i][0]))
        
        #the following line was commented
        for i in range(len(del_w)):
            del_w[i][0] *= -1*err_output[i][0]*self.hidden[i][0]

        #del who are the changes in the weights b/w hidden and output
        
        del_who = self.who.copy().T
        for j in range(del_who.shape[0]):
            for k in range(del_who.shape[1]):
                del_who[j][k] = -1*err_output[k][0]*del_w[k][0]*self.hidden[j][0]*self.learning_rate
        
       
        err_hidden = numpy.dot(self.who.copy().T, err_output)
        
        #del_wih is the weight changes between the input and hidden nodes
        del_w2 = numpy.dot(self.wih.copy(), self.input)
        del_w2 = self.activation_function(del_w2)*(1 - self.activation_function(del_w2))
        
        #the following block was commented
        for i in range(len(del_w2)):
            del_w2[i][0] *= -1*err_hidden[i][0]*self.input[i][0]

        del_wih = self.wih.copy().T
        for j in range(del_wih.shape[0]):
            for k in range(del_wih.shape[1]):
                del_wih[j][k] = -1*err_hidden[k][0]*del_w2[k][0]*self.input[j][0]*self.learning_rate


        self.who = numpy.add(self.who, del_who.T)
        self.wih = numpy.add(self.wih, del_wih.T)
        print("Weight who : ", self.who)
        print("Weight wih : ", self.wih)
        


    def query(self, input):
        #convert the inputs into an array
        self.input = numpy.array(input, ndmin=2).T
        # the values for the hidden nodes
        self.hidden = numpy.dot(self.wih, self.input)
        self.hidden = self.activation_function(self.hidden)
        
        #calculating the output nodes
        self.output = numpy.dot(self.who, self.hidden)
        self.output = self.activation_function(self.output)
        return self.output



    