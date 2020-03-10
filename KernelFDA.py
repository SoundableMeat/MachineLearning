import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import random

def kernel(X, Y, c=0.1):
    k = np.exp(-(X-Y)@(X-Y) / c)
    return k

def getConfusionMatrix(classes,test_labels):
    #Finding where the classes array are equal
    #to the labels from the original data
    correct = classes[classes.flatten()==test_labels.flatten()]
    false = classes[classes.flatten()!=test_labels.flatten()]

    #Making the terms of the confusion matrix, term 1_1 is correct
    #classified ham e-mail, term 1_2 is spam classified as ham,
    #term 2_1 is ham classified as spam, and term 2_2 is
    #correct classified spam e-mail
    term11 = len(correct[correct==np.zeros_like(correct)])
    term12 = len(false[false==np.zeros_like(false)])
    term21 = len(false[false==np.ones_like(false)])
    term22 = len(correct[correct==np.ones_like(correct)])

    #Constructing the confusion matrix
    confusion_matrix = np.array([[term11,term12],[term21,term22]])

    #Claculating the classification error by the
    #wrongly classified data points
    error = (term12 + term21)/np.sum(confusion_matrix)

    return confusion_matrix, error

def split_data(moons,labels,training_percentage=0.8):
    """Takes in the dataset, randomises the data and gives
    out the data as a training and validation set.

    Parameters
    ----------
    moons : numpy array
        x,y coordinates of the moons dataset
    labels : numpy array
        Labels for the datapoint, gives if point is class 0 or 1
    training_percentage : float
        Percentage of the data used as training data,
        the rest beeing validation data

    Returns
    -------
    training_data : numpy array
        Full set of training data
    training_labels : numpy array
        Labels for training data
    training_X : numpy array
        Training data of class 0
    training_Y : numpy array
        Training data of class 1
    test_data : numpy array
        Full set of validation data
    test_labels : numpy array
        Labels for validation data


    """
    data = np.c_[moons,labels]

    rows = data.shape[0]
    idx = list(np.arange(rows))
    np.random.shuffle(idx)
    data = np.copy(data[idx,:])

    bound = int(len(data)*training_percentage)

    training_data = data[:bound]
    training_data,training_labels = training_data[:,:2],training_data[:,2]

    test_data = data[bound:]
    test_data,test_labels = test_data[:,:2],test_data[:,2]

    training_X = training_data[training_labels==0]
    training_Y = training_data[training_labels==1]
    return training_data,training_labels,training_X,training_Y,test_data,test_labels

def kernelFDA(training,training_labels,X_tr,Y_tr,test,test_labels):
    """Impliments the kernel fda

    Parameters
    ----------
    training : numpy array
        Full set of training data
    training_labels : numpy array
        Labels for training data
    X_tr : numpy array
        Training data of class 0
    Y_tr : numpy array
        Training data of class 1
    test : numpy array
        Full set of validation data
    test_labels : numpy array
        Labels for validation data

    Returns
    -------
    confusion_matrix : numpy array
        Confusion matrix of datapoints
    error : float
        classification error of the method
    """
    N = len(training)
    N1,N2 = len(X_tr),len(Y_tr)

    #Finding the kernel of the datapoints.
    #This is the dot product of every
    #combination of datapoints from the input.
    #It is split into two kernels, one having all
    #inner products between class zero and all training
    #points, and the other between class one and all
    #training points
    K1 = np.zeros((N,N1))
    K2 = np.zeros((N,N2))
    for i in range(N):
        for j in range(N1):
            K1[i,j] = kernel(training[i], X_tr[j])
    for i in range(N):
        for j in range(N2):
            K2[i,j] = kernel(training[i], Y_tr[j])

    #Making the 1_N1 and 1_N2 matrices
    ones1, ones2 = np.ones((N1,N1))/N1,np.ones((N2,N2))/N2

    N_mat = K1@np.identity(N1)@K1.T - K1@ones1@K1.T\
     + K2@np.identity(N2)@K2.T - K2@ones2@K2.T

    #Making sure the N matrix is positive definite
    if np.amin(N_mat.diagonal()) <= 0:
        N_mat += np.identity(N)*(1e-5 + np.abs(np.amin(N_mat.diagonal())))

    #Making the M matrix by taking the mean of the kernels
    #in the axis containing the class spesific points.
    #This is done in a for loop and not using sum to
    #try to minimize numerical errors
    m1 = np.zeros(N)
    m2 = np.zeros(N)
    for i, point_i in enumerate(training):
        for point_j in X_tr:
            m1[i] += kernel(point_i, point_j)
        for j, point_j in enumerate(Y_tr):
            m2[i] += kernel(point_i, point_j)

    m1 = (m1* 1/N1).reshape(N,1)
    m2 = (m2* 1/N2).reshape(N,1)
    M = (m1 - m2)@(m1 - m2).T

    #Solving the eigen decomposition
    lam, v = np.linalg.eigh(np.linalg.inv(N_mat)@M)

    #Finding the maximum eigenvalue
    #and choosing the eigenvector
    #corresponding to it as the weights
    lam_max = np.amax(lam)
    w = v[:, lam_max == lam]

    #Projecting all test points into
    #the projection space to classify them
    proj = []
    for i in range(len(test)):
        point = test[i]
        tmp = 0
        for j in range(N):
            tmp += w[j]*kernel(point,training[j])
        proj.append(tmp)

    #Finding the mean of the two classes
    mean1 = np.sum(X_tr,axis=0)/N1
    mean2 = np.sum(Y_tr,axis=0)/N2

    #Projecting the mean into projection space
    tmp1,tmp2 = 0,0
    for j in range(N):
        tmp1 += w[j]*kernel(mean1,training[j])
        tmp2 += w[j]*kernel(mean2,training[j])
    mean_proj = [tmp1,tmp2]

    #Classifying the data based on which mean
    #they are closest to in projection space
    classes = []
    for point in proj:
        if abs(point-mean_proj[0]) < abs(point-mean_proj[1]):
            classes.append(0)
        else:
            classes.append(1)

    classes = np.array(classes)

    #Splitting the projected points for
    #visual representation
    proj = np.array(proj).flatten()
    proj1 = proj[classes.flatten()==0]
    proj2 = proj[classes.flatten()==1]
    y1 = np.zeros_like(proj1)
    y2 = np.zeros_like(proj2)

    #Finding the desition boundary
    avg_mean_proj = (mean_proj[0] + mean_proj[1])/2

    #Plotting the projected points
    #and the desition boundary
    fig = plt.figure()
    plt.plot(proj1,y1,'.',c='r')
    plt.plot(proj2,y2,'.',c='b')
    plt.axvline(avg_mean_proj,c='black')
    plt.yticks([])
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)

    confusion_matrix, error = getConfusionMatrix(classes, test_labels)

    return confusion_matrix, error


moons = sio.loadmat('NonLinearData/moons_dataset.mat')['X']
labels = sio.loadmat('NonLinearData/moons_dataset.mat')['y'].T

avg_con, avg_err = [], []

for i in range(1):
    training,training_labels,X_tr,Y_tr,test,test_labels = split_data(moons,labels)

    confusion_matrix, error = kernelFDA(training,training_labels,X_tr,Y_tr,test,test_labels)

    avg_con.append(confusion_matrix)
    avg_err.append(error)
    if i%10==0:
        print(i)

avg_con, avg_err = np.array(avg_con), np.array(avg_err)

print(np.sum(avg_con, axis=0)/len(avg_con))

print(np.sum(avg_err)/len(avg_err))
