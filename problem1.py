import math
import numpy as np
from collections import Counter
#-------------------------------------------------------------------------
'''
    Problem 1: Decision Tree (with Descrete Attributes)
    In this problem, you will implement the decision tree method for classification problems.
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
'''
 
#--------------------------
def Terms_and_Conditions():
    ''' 
        By submiting this homework or changing this function, you agree with the following terms:
       (1) Not sharing your code/solution with any student before and after the homework due. For example, sending your code segment to another student, putting your solution online or lending your laptop (if your laptop contains your solution or your dropbox automatically sychronize your solution between your home computer and your laptop) to another student to work on this homework will violate this term.
       (2) Not using anyone's code in this homework, build your own solution. For example, using some code segments from another student or online resources due to any reason (like too busy recently) will violate this term. Changing other people's code as your solution (such as changing the variable names) will also violate this term.
       (3) When discussing with any other student about this homework, only discuss high-level ideas or using psudo-code. Don't discuss about the solution at the code level. For example, discussing with another student about the solution of a function (which needs 5 lines of code to solve), and then working on the solution "independently", however the code of the two solutions are exactly the same, or only with minor differences  (like changing variable names) will violate this term.
      All violations of (1),(2) or (3) will be handled in accordance with the WPI Academic Honesty Policy.  For more details, please visit: https://www.wpi.edu/about/policies/academic-integrity/dishonesty
      Historical Data: in one year, we ended up finding 25% of the students in the class violating this term in their homework submissions and we handled ALL of these violations according to the WPI Academic Honesty Policy. 
    '''
    #########################################
    ## CHANGE CODE HERE
    Read_and_Agree = True

    #########################################
    return Read_and_Agree
        
#-----------------------------------------------
class Node:
    '''
        Decision Tree Node (with discrete attributes)
        Inputs:
            X: the data instances in the node, a numpy matrix of shape p by n.
               Each element can be int/float/string.
               Here n is the number data instances in the node, p is the number of attributes.
            Y: the class labels, a numpy array of length n.
               Each element can be int/float/string.
            i: the index of the attribute being tested in the node, an integer scalar
            C: the dictionary of attribute values and children nodes.
               Each (key, value) pair represents an attribute value and its corresponding child node.
            isleaf: whether or not this node is a leaf node, a boolean scalar
            p: the label to be predicted on the node (i.e., most common label in the node).
    '''
    def __init__(self,X,Y, i=None,C=None, isleaf= False,p=None):
        self.X = X
        self.Y = Y
        self.i = i
        self.C= C
        self.isleaf = isleaf
        self.p = p

#-----------------------------------------------
class Tree(object):
    '''
        Decision Tree (with discrete attributes).
        We are using ID3(Iterative Dichotomiser 3) algorithm. So this decision tree is also called ID3.
    '''
    #--------------------------
    @staticmethod
    def entropy(Y):
        '''
            Compute the entropy of a list of values.
            Input:
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                e: the entropy of the list of values, a float scalar
            Hint: you could use collections.Counter.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        Y = Counter(Y)
        x = iter(Y.values())
        sum1 = sum(Y.values())
        e = 0
        while True:
            try:
                a = next(x)
                e += -(a/sum1)*np.log2(a/sum1)
            except StopIteration:
                break
















        #########################################
        return e



    #--------------------------
    @staticmethod
    def conditional_entropy(Y,X):
        '''
            Compute the conditional entropy of y given x.
            Input:
                Y: a list of values, a numpy array of int/float/string values.
                X: a list of values, a numpy array of int/float/string values.
            Output:
                ce: the conditional entropy of y given x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        ce = 0
        counter_x = Counter(X)
        x_key = iter(counter_x.keys())
        sum_x = sum(counter_x.values())
        while True:
            dict_1 = zip(X, Y)
            try:
                key = next(x_key)
                list_1 = [i[1] for i in dict_1 if key == i[0]]
                ce += (counter_x[key]/sum_x)*Tree.entropy(list_1)
            except StopIteration:
                break
        #########################################
        return ce



    #--------------------------
    @staticmethod
    def information_gain(Y,X):
        '''
            Compute the information gain of y after spliting over attribute x
            Input:
                X: a list of values, a numpy array of int/float/string values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                g: the information gain of y after spliting over x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        g = Tree.entropy(Y)-Tree.conditional_entropy(Y,X)



        #########################################
        return g


    #--------------------------
    @staticmethod
    def best_attribute(X,Y):
        '''
            Find the best attribute to split the node.
            Here we use information gain to evaluate the attributes.
            If there is a tie in the best attributes, select the one with the smallest index.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
            Output:
                i: the index of the attribute to split, an integer scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        g1 = 0
        for a in range(X.shape[0]):
            g = Tree.information_gain(list(Y),list(X[a]))
            if g > g1:
                g1 = g
                i = a









        #########################################
        return i


    #--------------------------
    @staticmethod
    def split(X,Y,i):
        '''
            Split the node based upon the i-th attribute.
            (1) split the matrix X based upon the values in i-th attribute
            (2) split the labels Y based upon the values in i-th attribute
            (3) build children nodes by assigning a submatrix of X and Y to each node
            (4) build the dictionary to combine each  value in the i-th attribute with a child node.

            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                i: the index of the attribute to split, an integer scalar
            Output:
                C: the dictionary of attribute values and children nodes.
                   Each (key, value) pair represents an attribute value and its corresponding child node.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        key = iter(Counter(X[i]).keys())
        C = {}
        while True:
            try:
                a = next(key)
                x = [b for b in np.where(X[i] == a)[0]] # if X[i] is a 1*n vector, np.where(X[i] == a) will return (array([0, 1]),)
                C[a] = Node(X[:,x],Y[x])
            except StopIteration:
                break















        #########################################
        return C

    #--------------------------
    @staticmethod
    def stop1(Y):
        '''
            Test condition 1 (stop splitting): whether or not all the instances have the same label.

            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                s: whether or not Conidtion 1 holds, a boolean scalar.
                True if all labels are the same. Otherwise, false.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        s = np.all(np.isin(Y,Y[0]))



        #########################################
        return s

    #--------------------------
    @staticmethod
    def stop2(X):
        '''
            Test condition 2 (stop splitting): whether or not all the instances have the same attributes.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
            Output:
                s: whether or not Conidtion 2 holds, a boolean scalar.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        for i in range(X.shape[0]):
            if Tree.stop1(X[i]):
                s = True
            else:
                s = False
                break












        #########################################
        return s


    #--------------------------
    @staticmethod
    def most_common(Y):
        '''
            Get the most-common label from the list Y.
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node.
            Output:
                y: the most common label, a scalar, can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        y = max(dict(Counter(Y)),key=lambda a:dict(Counter(Y))[a])





        #########################################
        return y



    #--------------------------
    @staticmethod
    def build_tree(t):
        '''
            Recursively build tree nodes.
            Input:
                t: a node of the decision tree, without the subtree built.
                t.X: the feature matrix, a numpy float matrix of shape n by p.
                   Each element can be int/float/string.
                    Here n is the number data instances, p is the number of attributes.
                t.Y: the class labels of the instances in the node, a numpy array of length n.
                t.C: the dictionary of attribute values and children nodes. 
                   Each (key, value) pair represents an attribute value and its corresponding child node.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        # if Condition 1 or 2 holds, stop recursion
        t.p = Tree.most_common(t.Y)
        if Tree.stop1(t.Y) == False and Tree.stop2(t.X) == False:
            t.i = Tree.best_attribute(t.X,t.Y)
            t.C = Tree.split(t.X,t.Y,t.i)
            for value in t.C.values():
                Tree.build_tree(value)
        else:
            t.isleaf=True










        # find the best attribute to split





        # recursively build subtree on each child node



        #########################################
    
    
    #--------------------------
    @staticmethod
    def train(X, Y):
        '''
            Given a training set, train a decision tree. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the training set, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                t: the root of the tree.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        t = Node(X,Y)
        Tree.build_tree(t)




        #########################################
        return t
    
    
    
    #--------------------------
    @staticmethod
    def inference(t,x):
        '''
            Given a decision tree and one data instance, infer the label of the instance recursively. 
            Input:
                t: the root of the tree.
                x: the attribute vector, a numpy vectr of shape p.
                   Each attribute value can be int/float/string.
            Output:
                y: the class label, a scalar, which can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        # predict label, if the current node is a leaf node
        if t.isleaf == False and x[t.i] in t.C.keys():
            y = Tree.inference(t.C[x[t.i]],x)
        else:
            y = t.p
        









        #########################################
        return y
    
    #--------------------------
    @staticmethod
    def predict(t,X):
        '''
            Given a decision tree and a dataset, predict the labels on the dataset. 
            Input:
                t: the root of the tree.
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the dataset, p is the number of attributes.
            Output:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        Y = []
        for i in range(X.shape[1]):
            Y.append(Tree.inference(t,X[:,i]))
        Y = np.asarray(Y)








        #########################################
        return Y



    #--------------------------
    @staticmethod
    def load_dataset(filename='data1.csv'):
        '''
            Load dataset 1 from the CSV file: 'data1.csv'. 
            The first row of the file is the header (including the names of the attributes)
            In the remaining rows, each row represents one data instance.
            The first column of the file is the label to be predicted.
            In remaining columns, each column represents an attribute.
            Input:
                filename: the filename of the dataset, a string.
            Output:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element is a string.
                   Here n is the number data instances in the dataset, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element is a string.
            Note: Here you can assume the data type is always str.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        data = np.genfromtxt(filename, dtype='str', delimiter=',')
        data = data[1::]
        Y = data[:, 0]
        X = data[:, 1:8].T

        #########################################
        return X,Y



