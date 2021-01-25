# example of binary classification task

#si vous etes sous linux 
##installez sudo apt-get python3-pip
#mac os: demerde toi!

#s'il y a un paquet manquant du type sklearn : module not found

#pip3 install scikit-learn , par exemple
import numpy as np
from collections import Counter
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
# define dataset
X, y = make_blobs(n_samples=40, centers=2, random_state=1) #makeblob generates a dataset of "centers" classes, random state
for feature in y:
     if feature == 0 :
         print(y)
         indice=np.where(y==0)
         y[indice]=-1
     else:
         continue

# X = np.array([
#     [-2,4,-1],
#     [4,1,-1],
#     [1, 6, -1],
#     [2, 4, -1],
#     [6, 2, -1],
 
# ])

# y = np.array([-1,-1,1,1,1])
#defines the randomness of the ssamples, n_samples is n_samples, lisez la doc entre autres


# summarize observations by class label
counter = Counter(y)
print(counter)
# summarize first few examples
for i in range(len(X)):
	print(X[i], y[i])
# plot the dataset and color the by class label
for label, _ in counter.items():
# for label in y:
	row_ix = np.where(y == label)[0]
	plt.scatter(X[row_ix, 0], X[row_ix, 1])
plt.legend()
plt.show()


#lets perform stochastic gradient descent to learn the seperating hyperplane between both classes

def svm_gradient_decent_plot(X, Y):
    #Initialize our SVMs weight vector with zeros (3 values)
    w = np.zeros(len(X[0])) #zeros
    #The learning rate
    eta = 1
    #how many iterations to train for
    epochs = 10000
    #store misclassifications so we can plot how they change over time
    errors = []

    #training part, gradient descent part
    for epoch in range(1,epochs):
        print("Epoch no. " + str(epoch))
        error = 0
        for i, x in enumerate(X):
            #misclassification
            if (Y[i]*np.dot(X[i], w)) < 1:
                #misclassified update for ours weights
                w = w + eta * ( (X[i] * Y[i]) + (-2  *(1/epoch)* w) )
                error = 1
            else:
                #correct classification, update our weights
                w = w + eta * (-2  *(1/epoch)* w)
        errors.append(error)
        

    #lets plot the rate of classification errors during training for our SVM
    plt.plot(errors,'+')
    plt.ylim(0.5,1.5)
    plt.axes().set_yticklabels([])
    plt.xlabel('Epoch')
    plt.ylabel('Misclassified ')
    plt.show()
    
    return w




w = svm_gradient_decent_plot(X,y)
#they decrease over time! Our SVM is learning the optimal hyperplane

for d, sample in enumerate(X):
    # Plot the negative samples
    if d < 2:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
    # Plot the positive samples
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

# Add our test samples
plt.scatter(2,2, s=120, marker='o', linewidths=2, color='yellow')
plt.scatter(4,3, s=120, marker='o', linewidths=2, color='blue')

# # Print the hyperplane calculated by svm_sgd()
x2=[w[0],w[1],-w[1],w[0]]

x3=[w[0],w[1],w[1],-w[0]]
# plt.scatter(1,1)
x2x3 =np.array([x2,x3])
X,Y,U,V = zip(*x2x3)
ax = plt.gca()
ax.quiver(X,Y,U,V,scale=1, color='blue')
plt.show()
