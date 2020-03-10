import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import random
from svm_util import *

def getConfusionMatrix(classes,y_te):
    """Takes the labels from the classification
    methods and the labels from the original data.
    Then finds the classification error and
    confusion matrix for the classification method.

    Parameters
    ----------
    classes : numpy array
        Array of shape (N,1) or (N,) with
        labels made from the classifiers.
    y_te : numpy array
        Array of shape (N,1) or (N,) with
        labels from the original data.

    Returns
    -------
    confusion_matrix : numpy array
        Confusion matrix of shape (2,2)
    error : float
        The classification error
        of the classifier.

    """
    #Finding where the classes array are equal
    #to the labels from the original data
    correct = classes[classes.flatten()==y_te.flatten()]
    false = classes[classes.flatten()!=y_te.flatten()]

    #Making the terms of the confusion matrix, term 1_1 is correct
    #classified ham e-mail, term 1_2 is spam classified as ham,
    #term 2_1 is ham classified as spam, and term 2_2 is
    #correct classified spam e-mail
    term11 = len(correct[correct==-np.ones_like(correct)])
    term12 = len(false[false==-np.ones_like(false)])
    term21 = len(false[false==np.ones_like(false)])
    term22 = len(correct[correct==np.ones_like(correct)])

    #Constructing the confusion matrix
    confusion_matrix = np.array([[term11,term12],[term21,term22]])

    #Claculating the classification error by the
    #wrongly classified data points
    error = (term12 + term21)/np.sum(confusion_matrix)

    return confusion_matrix, error

def SumOfLeastSquares(X_tr,X_te,y_tr,y_te):
    """Using the sum of least squares
    to classify the test data.

    Parameters
    ----------
    X_tr : numpy array
        Array of shape (N,d) where N is the number of
        training points and d is the number of features
        of the data.
    X_te : numpy array
        Array of shape (N,d) where N is the number of
        test points and d is the number of features
        of the data.
    y_tr : numpy array
        Array of length N, containing the
        labels for the training points
    y_te : numpy array
        Array of length N, containing the
        labels for the test points

    Returns
    -------
    confusion_matrix :
        The confusion matrix of the
        classification method
    error :
        The classification error
        of the method

    """
    #Adding a bias term to the feature vectors
    X_tr = np.c_[np.ones(len(X_tr)),X_tr]
    X_te = np.c_[np.ones(len(X_te)),X_te]

    #Computing the sample correlation matrix
    #of the training data and the test data
    Xtr_cor = X_tr.T@X_tr
    Xte_cor = X_te.T@X_te

    #Computing the weights given in eq 3.45 in the textbook
    weights = np.linalg.inv(Xtr_cor)@X_tr.T@y_tr

    #Classifying the test data based on the weights
    classes = X_te@weights

    #Setting negative values to -1, and positive (and zero) values to 1
    classes[classes>=0] = 1
    classes[classes<0] = -1

    #Calling function to find confusion
    #matrix and classification error
    confusion_matrix,error = getConfusionMatrix(classes,y_te)

    return confusion_matrix, error

def BayesClassifierWindowing(X_tr,X_te,y_tr,y_te,b=0.8):
    """Using the Bayes classifyer and Parzen
    windowing to classify the test data.

    Parameters
    ----------
    X_tr : numpy array
        Array of shape (N,d) where N is the number of
        training points and d is the number of features
        of the data.
    X_te : numpy array
        Array of shape (N,d) where N is the number of
        test points and d is the number of features
        of the data.
    y_tr : numpy array
        Array of length N, containing the
        labels for the training points
    y_te : numpy array
        Array of length N, containing the
        labels for the test points
    b : float
        The width of the Parzen windowing

    Returns
    -------
    confusion_matrix :
        The confusion matrix of the
        classification method
    error :
        The classification error
        of the method

    """
    #Separating the training data
    #into spam and ham data
    X_ham = X_tr[y_tr.flatten()==-1]
    X_spam = X_tr[y_tr.flatten()==1]

    #Finding the shape of the data,
    #then the prior probability
    #of spam and ham data
    N_ham,l = X_ham.shape
    N_spam,_ = X_spam.shape
    prior_ham = len(X_ham)/len(X_tr)
    prior_spam = len(X_spam)/len(X_tr)

    #Calculating the constant term
    #of the pdf estimation
    C_ham = 1/(N_ham*((2*np.pi)**(l/2))*(b**l))
    C_spam = 1/(N_spam*((2*np.pi)**(l/2))*(b**l))

    classes = []

    for data_point in X_te:
        #Initializing the probability of spam and ham
        p_spam = 0
        p_ham = 0
        for row in X_ham:
            diff = data_point - row
            #Using equation 2.110 from the textbook
            #to find the probability of ham
            p_ham += C_ham * np.exp(-(diff.T@diff)/2*b**2)
        for row in X_spam:
            diff = data_point - row
            #Using equation 2.110 from the textbook
            #to find the probability of spam
            p_spam += C_spam * np.exp(-(diff.T@diff)/2*b**2)
        #Multiplying the probabilities with the priors
        p_ham *= prior_ham
        p_spam *= prior_spam
        #Classifying the data based of Bayes
        #classification rule
        if p_ham > p_spam:
            classes.append(-1)
        else:
            classes.append(1)
        if len(classes) % 500 == 0:
            print(len(classes))

    classes = np.array(classes).flatten()

    confusion_matrix, error = getConfusionMatrix(classes,y_te)

    return confusion_matrix, error



def SVM_kernel(X,sigma):
    """Takes in a matrix X with shape (N,d)
    to find the euclidian distance between
    every point in the matrix

    Parameters
    ----------
    X : numpy array
        Array of shape (N,d) containing
        data N datapoints with d features
    sigma : float
        Width of the kernel

    Returns
    -------
    K : numpy array
        The kernel used in the SVM classification
        with the shape (N,N)

    """
    A = np.ones(np.shape(X.T))
    B = np.ones(np.shape(X))

    #Claculating the squared euclidian distance from
    #all points in X to every other point in X
    euc_dist_sqr = np.dot(X**2,A) - 2*np.dot(X,X.T) + np.dot(B,(X**2).T)

    K = np.exp(-0.5/(sigma**2)*euc_dist_sqr)

    return K

def calc_LH(i,j,y,alpha,C):
    """Calculating the lower and higher
    constraint of the lagrange multiplyer
    based on the multiplyer pair and the
    slack weight of the SVM


    Parameters
    ----------
    i : int
        Index of the first lagrange multiplyer
        in the pair we want to change
    j : int
        Index of the second lagrange multiplyer
        in the pair we want to change
    y : numpy array
        Array containing the labels of all training points
    alpha : numpy array
        Array containing the lagrange multiplyers
    C : int
        Weight of the slack variable in the SVM

    Returns
    -------
    L : float
        Lower constraint of the lagrange multiplyer
    H : float
        Higher constraint of the lagrange multiplyer

    """
    if y[i] != y[j]:
        L = max((0,alpha[j]-alpha[i]))
        H = min((C,C+alpha[j]-alpha[i]))
    else:
        L = max((0,alpha[i]+alpha[j]-C))
        H = min((C,alpha[i]+alpha[j]))

    return L,H

def alpha_update(alphaj_old,y_j,Ei,Ej,L,H,eta):
    """Short summary.

    Parameters
    ----------
    alphaj_old : float
        Old lagrange multiplyer for index j
    y_j : int
        Label of the data point at index j
    Ei : float
        Error between the SVM output
        and the label at index i
    Ej : float
        Error between the SVM output
        and the label at index j
    L : float
        Lower constraint for the
        lagrange multiplyer
    H : float
        Upper constraint for the
        lagrange multiplyer
    eta : float
        Dinstance from i to j squared

    Returns
    -------
    alpha : float
        Description of returned object.

    """
    alpha = alphaj_old - y_j*(Ei - Ej)/eta

    if alpha > H:
        alpha = H
    elif alpha < L:
        alpha = L

    return alpha

def b_update(b,i,j,Ei,Ej,alpha,K,C,y_tr,alphai_old,alphaj_old):
    """Calculates the bias term based on the previous bias
    term and updates in the other values.

    Parameters
    ----------
    b : float
        bias term for the SVM
    i : int
        Index for the first term
    j : int
        Index for the second term
    Ei : float
        Error between the SVM output
        and the label at index i
    Ej : float
        Error between the SVM output
        and the label at index j
    alpha : numpy array
        Array containing the
        lagrange multiplyers
    K : numpy array
        The kernel matrix used in the SVM
    C : float
        Weight for the slack term
    y_tr : numpy array
        The labels for the training data
    alphai_old : float
        Previous lagrange multiplyer
        at index i
    alphaj_old : float
        Previous lagrange multiplyer
        at index j

    Returns
    -------
    b : float
        New bias term for the SVM

    """
    #Using the formula to calculate the bias
    b1 = b - Ei - y_tr[i]*(alpha[i] - alphai_old)*K[i,i]\
    - y_tr[j]*(alpha[j] - alphaj_old)*K[i, j]
    b2 = b - Ej - y_tr[i]*(alpha[i] - alphai_old)*K[i,j]\
    - y_tr[j]*(alpha[j] - alphaj_old)*K[j, j]

    #Ensures the KKT conditions are satisfied
    if 0 < alpha[i] < C:
        b = b1
    elif 0 < alpha[j] < C:
        b = b2
    else:
        b = (b1 + b2)/2

    return b

def SupportVectorMachine(X_tr,X_te,y_tr,y_te,sigma=4.5,C=1.7,max_iter=10000,iter_conv=1000,tol=1e-6):
    """Uses sequential minimal optimalization (SMO)
    to find the best fot for the training data, and
    uses this to classify the test data

    Parameters
    ----------
    X_tr : numpy array
        Training data for the SVM
    X_te : numpy array
        Test data for the SVM
    y_tr : numpy array
        Labels for the training data
    y_te : numpy array
        Labels for the test data
    sigma : float
        Width of the SVM kernel
    C : float
        Weight of the slack term
    max_iter : int
        Max number of iterations before
        terminating the convergence
    iter_conv : int
        Number of iterations without change
        needed to have convergence
    tol : float
        The minimum value of the error before
        changing the values of the lagrange multiplyer

    Returns
    -------
    confusion_matrix : numpy array
        Confusion matrix for the
        classification method
    error : float
        Classification error for the
        classification method

    """
    y_tr = y_tr.flatten()
    N = len(y_tr)
    K = SVM_kernel(X_tr,sigma)

    #Initializing the lagrange
    #multiplyers and the bias term
    alpha = np.zeros(N,dtype=float)
    b = 0

    iter = 1
    unchanged_iter = 0

    while iter < max_iter and unchanged_iter < iter_conv:
        if iter%100 == 0:
            print(iter,unchanged_iter,alphas_changed)
        alphas_changed = 0

        #Iterating through all samples in the training set
        for i in range(N):
            #Finding the reeor between the transformation and the label
            Ei = np.dot(alpha*y_tr,K[:,i])+b-y_tr[i]
            #If the error is large enough and the lagrangian is
            #between 0 and C we go ahead with changing the lagrangian
            if ((Ei*y_tr[i] < -tol and alpha[i] < C) or (Ei*y_tr[i] > tol and alpha[i] > 0)):

                #Choosing a random j other than i
                j = i
                while j==i:
                    j = random.randint(0, N - 1)

                #Calculating the upper and lower
                #bound for the lagrange multiplyer
                L,H = calc_LH(i,j,y_tr,alpha,C)

                #If the lower and higher bound is equal
                #We continue to the next sample
                if L==H:
                    continue

                #Distance from point i to point j squared
                eta = 2*K[i, j] - K[i, i] - K[j, j]
                if eta>=0:
                    continue

                #Calculating the error term for the j-th lagrangian
                Ej = np.dot(alpha*y_tr,K[:,j])+b-y_tr[j]
                alphai_old, alphaj_old = alpha[i].copy(), alpha[j].copy()

                #Updating the j-th lagrangian
                alpha[j] = alpha_update(alpha[j],y_tr[j],Ei,Ej,L,H,eta)

                #If the change is too small, continue to next sample
                if abs(alpha[j]-alphaj_old)<1e-5:
                    continue

                #Updating the i-th lagrangian
                alpha[i] += y_tr[i]*y_tr[j]*(alphaj_old - alpha[j])

                #Update the bias term
                b = b_update(b,i,j,Ei,Ej,alpha,K,C,y_tr,alphai_old,alphaj_old)

                #Counter for the number of alpha pairs changed
                alphas_changed += 1

        #If the number of lagnangians changed is zero
        #we have one unchanged iteration
        if alphas_changed == 0:
            unchanged_iter += 1
        else:
            unchanged_iter = 0

        iter += 1

    #If the number of iterations reached it's maximum
    #we have not reached convergence
    if iter == max_iter:
        print("No convergence")

    #Uses the lagrange multiplyers to construct the
    #weights array
    weights = (alpha*y_tr).T@X_tr

    #Classifies the test points using the
    #weigths obtained and the bias term
    classes = (weights@X_te.T + b).T

    #Classifies all negative numbers to -1
    #and possitive numbers to 1
    classes[classes>=0] = 1
    classes[classes<0] = -1

    #Constructs the confusion matrix
    #and the classification error
    confusion_matrix, error = getConfusionMatrix(classes,y_te)

    return confusion_matrix, error


#Importing the training sets, and extracting the feature vectors and
#labels. Then transposing them to get the N features as rows.
X_tr = sio.loadmat('SpamData/Xtr_spam.mat')['Xtr_spam'].T
y_tr = sio.loadmat('SpamData/ytr_spam.mat')['ytr_spam'].T

#Importing the test sets, and extracting the feature vectors.
X_te = sio.loadmat('SpamData/Xte_spam.mat')['Xte_spam'].T
y_te = sio.loadmat('SpamData/yte_spam.mat')['yte_spam'].T

#Uses all three methods to find the
#classification error and the confusion matrix
least_squares = SumOfLeastSquares(X_tr,X_te,y_tr,y_te)
bayes = BayesClassifierWindowing(X_tr,X_te,y_tr,y_te)
SVM = SupportVectorMachine(X_tr,X_te,y_tr,y_te)

print(least_squares,bayes,SVM)
