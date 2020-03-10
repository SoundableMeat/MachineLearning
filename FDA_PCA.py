import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import random

def fda(moons,labels):
    """Uses the fisher discriminant ratio
    to classify the datapoints.

    Parameters
    ----------
    moons : numpy array
        Coordinates of the datapoints
    labels : numpy array
        Labels of the classes

    """
    #The next few lines are to mix and separate the data
    X = moons[labels.flatten()==0]
    Y = moons[labels.flatten()==1]

    rows = X.shape[0]
    idx = list(np.arange(rows))
    np.random.shuffle(idx)
    X = np.copy(X[idx,:])
    Y = np.copy(Y[idx,:])

    X_tr,Y_tr = X[:100],Y[:100]
    X_te,Y_te = X[100:],Y[100:]

    #The averages of the classes are calculated
    mu_x = (np.sum(X_tr,axis=0)/len(X_tr)).reshape(2,1)
    mu_y = (np.sum(Y_tr,axis=0)/len(Y_tr)).reshape(2,1)

    #We find the between and within class
    #scatter matrices
    S_B = (mu_x-mu_y)@(mu_x-mu_y).T
    S_W = np.zeros_like(S_B)
    for i in range(len(X_tr)):
        S_W += (X_tr[0]-mu_x)@(X_tr[0]-mu_x).T
        S_W += (Y_tr[0]-mu_y)@(Y_tr[0]-mu_y).T

    A = np.linalg.inv(S_W)@S_B

    #The eigenvalues and eigenvectors are found
    lam, v = np.linalg.eig(A)

    #We choose the eigenvector
    #corresponding to the largest eigenvalue
    if lam[0]>lam[1]:
        w = v[:,0]
    else:
        w = v[:,1]

    #The test data is classified
    test_x = X_te@w - w.T@(0.5*(mu_x+mu_y))
    test_y = Y_te@w - w.T@(0.5*(mu_x+mu_y))

    #We find the correct and incorrect
    #classified data points
    test_x[test_x<=0] = 0
    test_x[test_x>0] = 1
    test_y[test_y<=0] = 0
    test_y[test_y>0] = 1

    #Printing the classification error
    print(1-((len(test_x[test_x==0])+len(test_y[test_y==1]))\
    /(len(test_x)+len(test_y))))

    #Making a line for the weights
    #and the classification line
    x = np.linspace(-1.1,1.5,1000)
    classification_line = -(w[0]/w[1])*x -w.T@(0.5*(mu_x+mu_y))
    weights_direction = (w[0]/w[1])*x -w.T@(0.5*(mu_x+mu_y))

    #Calling the pca and getting the first
    #principal axis
    x1,pca_direction = pca(moons)

    fig = plt.figure()
    plt.scatter(X_te[test_x==0][:,0],X_te[test_x==0][:,1],c='b')
    plt.scatter(X_te[test_x==1][:,0],X_te[test_x==1][:,1],c='r')
    plt.scatter(Y_te[test_y==0][:,0],Y_te[test_y==0][:,1],c='b')
    plt.scatter(Y_te[test_y==1][:,0],Y_te[test_y==1][:,1],c='r')
    plt.plot(x,classification_line,c='g')
    plt.axis('equal')
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)

    fig = plt.figure()
    plt.scatter(X_te[test_x==0][:,0],X_te[test_x==0][:,1],c='b')
    plt.scatter(X_te[test_x==1][:,0],X_te[test_x==1][:,1],c='r')
    plt.scatter(Y_te[test_y==0][:,0],Y_te[test_y==0][:,1],c='b')
    plt.scatter(Y_te[test_y==1][:,0],Y_te[test_y==1][:,1],c='r')
    plt.plot(x,weights_direction,c='g')
    plt.plot(x1,pca_direction,c='y')
    plt.axis('equal')
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)

def pca(moons):
    """Finds the axis with the
    highest variance

    Parameters
    ----------
    moons : numpy array
        coordinates of all input points

    Returns
    -------
    x,y : numpy arrays
        line showing the direction of
        the first principal axis

    """
    C = np.cov(moons)

    lam, v = np.linalg.eig(C)

    if lam[0].real >= lam[1].real:
        w = v[0].real
    else:
        w = v[1].real

    x = np.linspace(-1.1,1.5,1000)
    y = (w[0]/w[1])*x

    return x,y


moons = sio.loadmat('NonLinearData/moons_dataset.mat')['X']
labels = sio.loadmat('NonLinearData/moons_dataset.mat')['y'].T
fda(moons,labels)
