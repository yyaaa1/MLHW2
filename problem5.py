import math
import numpy as np
from problem2 import DT,Node
#-------------------------------------------------------------------------
'''
    Problem 5: Boosting (on continous attributes). 
               We will implement AdaBoost algorithm in this problem.
    You could test the correctness of your code by typing `nosetests -v test5.py` in the terminal.
'''

#-----------------------------------------------
class DS(DT):
    '''
        Decision Stump (with contineous attributes) for Boosting.
        Decision Stump is also called 1-level decision tree.
        Different from other decision trees, a decision stump can have at most one level of child nodes.
        In order to be used by boosting, here we assume that the data instances are weighted.
    '''
    #--------------------------
    @staticmethod
    def entropy(Y, D):
        '''
            Compute the entropy of the weighted instances.
            Input:
                Y: a list of labels of the instances, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
                e: the entropy of the weighted samples, a float scalar
            Hint: you could use np.unique(). 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        e = 0
        for i in np.unique(Y):
            a = 0
            for j in range(Y.shape[0]):
                if i == Y[j]:
                    a += D[j]
            if a != 0:
                e += -a * np.log2(a)
            else:
                e = 0









        #########################################
        return e
        #--------------------------
    @staticmethod
    def conditional_entropy(Y,X,D):
        '''
            Compute the conditional entropy of y given x on weighted instances
            Input:
                Y: a list of values, a numpy array of int/float/string values.
                X: a list of values, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
                ce: the weighted conditional entropy of y given x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        ce = 0
        for i in np.unique(X):
            c = np.where(X == i)
            a = np.sum(D[c])
            if a != 0:
                h = D[c]
                for j in range(len(D[c])):
                    h[j] = h[j] / a
                ce += a * DS.entropy(Y[np.where(X == i)], h)

        #########################################
        return ce 

    #--------------------------
    @staticmethod
    def information_gain(Y,X,D):
        '''
            Compute the information gain of y after spliting over attribute x
            Input:
                X: a list of values, a numpy array of int/float/string values.
                Y: a list of values, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
                g: the weighted information gain of y after spliting over x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        g = DS.entropy(Y,D)-DS.conditional_entropy(Y,X,D)


        #########################################
        return g


    #--------------------------
    @staticmethod
    def best_threshold(X,Y,D):
        '''
            Find the best threshold among all possible cutting points in the continous attribute of X. The data instances are weighted. 
            Input:
                X: a list of values, a numpy array of int/float values.
                Y: a list of values, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
                th: the best threhold, a float scalar. 
                g: the weighted information gain by using the best threhold, a float scalar. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        cp = DT.cutting_points(X,Y)
        if type(cp) == type(np.array([1])):
            g = -1
            th = float('-inf')
            for i in cp:
                a = (np.ma.masked_where(X > i, X)).mask
                if DS.information_gain(Y,a,D)>g:
                    g = DS.information_gain(Y,a,D)
                    th = i


        else:
            g = -1
            th = float('-inf')











        #########################################
        return th,g 
     
    #--------------------------
    def best_attribute(self,X,Y,D):
        '''
            Find the best attribute to split the node. The attributes have continous values (int/float). The data instances are weighted.
            Here we use information gain to evaluate the attributes. 
            If there is a tie in the best attributes, select the one with the smallest index.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                D: the weights of instances, a numpy float vector of length n
            Output:
                i: the index of the attribute to split, an integer scalar
                th: the threshold of the attribute to split, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        g = []
        th_1 = []
        for i in X:
            g.append(DS.best_threshold(i,Y,D)[1])
            th_1.append(DS.best_threshold(i,Y,D)[0])
        i = np.argmax(g)
        th = th_1[i]










        #########################################
        return i, th
             
    #--------------------------
    @staticmethod
    def most_common(Y,D):
        '''
            Get the most-common label from the list Y. The instances are weighted.
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node.
                D: the weights of instances, a numpy float vector of length n
            Output:
                y: the most common label, a scalar, can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        list_1 = []
        for i in np.unique(Y):
            d = 0
            for j in range(len(Y)):
                if i==Y[j]:
                    d += D[j]
            list_1.append(d)

        y = np.unique(Y)[np.argmax(list_1)]






        #########################################
        return y
 

    #--------------------------
    def build_tree(self, X,Y,D):
        '''
            build decision stump by overwritting the build_tree function in DT class.
            Instead of building tree nodes recursively in DT, here we only build at most one level of children nodes.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                D: the weights of instances, a numpy float vector of length n
            Return:
                t: the root node of the decision stump. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        t = Node(X,Y)
        t.p = DS.most_common(t.Y,D)



        # if Condition 1 or 2 holds, stop splitting
        if DT.stop1(t.Y) == False and DT.stop2(t.X) == False:
            t.i,t.th = DS().best_attribute(t.X,t.Y,D)
            t.C1, t.C2 = DT.split(t.X, t.Y, t.i, t.th)
            d1 = D[np.where(X[t.i]<t.th)]
            d2 = D[np.where(X[t.i] >= t.th)]
            t.C1.p = DS.most_common(t.C1.Y,d1)
            t.C2.p = DS.most_common(t.C2.Y,d2)
            t.C1.isleaf = True
            t.C2.isleaf = True

        else:
            t.isleaf = True








        # find the best attribute to split








        # configure each child node





        #########################################
        return t
    
 

#-----------------------------------------------
class AB(DS):
    '''
        AdaBoost algorithm (with contineous attributes).
    '''

    #--------------------------
    @staticmethod
    def weighted_error_rate(Y,Y_,D):
        '''
            Compute the weighted error rate of a decision on a dataset. 
            Input:
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                Y_: the predicted class labels, a numpy array of length n. Each element can be int/float/string.
                D: the weights of instances, a numpy float vector of length n
            Output:
                e: the weighted error rate of the decision stump
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        a = [i for i in range(len(Y)) if Y[i] != Y_[i]]
        e = np.sum(D[a])







        #########################################
        return e

    #--------------------------
    @staticmethod
    def compute_alpha(e):
        '''
            Compute the weight a decision stump based upon weighted error rate.
            Input:
                e: the weighted error rate of a decision stump
            Output:
                a: (alpha) the weight of the decision stump, a float scalar.
        '''
        #########################################
        ## INSERT YOUR CODE HERE##
        if e>=1:
            a = -101
        elif np.allclose(e,0,atol=1e-20):
            a = 101
        else:
            a = (1/2)*np.log((1-e)/e)










        #########################################
        return a

    #--------------------------
    @staticmethod
    def update_D(D,a,Y,Y_):
        '''
            update the weight the data instances 
            Input:
                D: the current weights of instances, a numpy float vector of length n
                a: (alpha) the weight of the decision stump, a float scalar.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                Y_: the predicted class labels by the decision stump, a numpy array of length n. Each element can be int/float/string.
            Output:
                D: the new weights of instances, a numpy float vector of length n
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        D2 = np.copy(D)
        corr = [i for i in range(len(Y)) if Y[i] == Y_[i]]
        in_corr = [i for i in range(len(Y)) if Y[i] != Y_[i]]
        D2[corr] = D[corr] * np.exp(-a)
        D2[in_corr] = D[in_corr] * np.exp(a)
        D = D2 / np.sum(D2)





        #########################################
        return D

    #--------------------------
    @staticmethod
    def step(X,Y,D):
        '''
            Compute one step of Boosting.  
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                D: the current weights of instances, a numpy float vector of length n
            Output:
                t:  the root node of a decision stump trained in this step
                a: (alpha) the weight of the decision stump, a float scalar.
                D: the new weights of instances, a numpy float vector of length n
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        t = DS().build_tree(X,Y,D)
        Y_ = DT.predict(t,X)
        e = AB.weighted_error_rate(Y,Y_,D)
        a = AB.compute_alpha(e)
        D = AB.update_D(D,a,Y,Y_)










        #########################################
        return t,a,D

    
    #--------------------------
    @staticmethod
    def inference(x,T,A):
        '''
            Given an adaboost ensemble of decision trees and one data instance, infer the label of the instance. 
            Input:
                x: the attribute vector of a data instance, a numpy vectr of shape p.
                   Each attribute value can be int/float
                T:  the root nodes of decision stumps, a list of length n_tree. 
                A: the weights of the decision stumps, a numpy float vector of length n_tree.
            Output:
                y: the class label, a scalar of int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        y_list = []
        b = 0
        for i in T:
            y_list.append(DT.inference(i,x))
        for i in set(y_list):
            a = 0
            for j in range(len(A)):
                if y_list[j] == i:
                    a += A[j]
            if a>b:
                b = a
                y = i













        #########################################
        return y
 

    #--------------------------
    @staticmethod
    def predict(X,T,A):
        '''
            Given an AdaBoost and a dataset, predict the labels on the dataset. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                T:  the root nodes of decision stumps, a list of length n_tree. 
                A: the weights of the decision stumps, a numpy float vector of length n_tree.
            Output:
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        Y = []
        for i in X.T:
            Y.append(AB.inference(i,T,A))
        Y = np.asarray(Y)









        #########################################
        return Y 
 

    #--------------------------
    @staticmethod
    def train(X,Y,n_tree=10):
        '''
            train adaboost.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                n_tree: the number of trees in the ensemble, an integer scalar
            Output:
                T:  the root nodes of decision stumps, a list of length n_tree. 
                A: the weights of the decision stumps, a numpy float vector of length n_tree.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        # initialize weight as 1/n
        A = []
        T = []
        D = np.ones(Y.shape[0])/Y.shape[0]
        # iteratively build decision stumps
        for i in range(n_tree):
            t,a,D= AB.step(X,Y,D)
            T.append(t)
            A.append(a)
        A = np.asarray(A)









        #########################################
        return T, A
   



 
