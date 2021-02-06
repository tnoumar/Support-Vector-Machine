
import numpy as np
from collections import Counter
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

# define dataset
X, y = make_blobs(n_samples=80, centers=2, random_state=6) #makeblob generates a dataset of "centers" classes, random state

# classes en 1 et -1
for feature in y:
     if feature == 0 :
         indice=np.where(y==0)
         y[indice]=-1
     else:
         continue



# visualisation des classes
counter = Counter(y)
print(counter)


# representer le dataset et plot
for label, _ in counter.items(): # for label in y:

	row_ix = np.where(y == label)[0]
	plt.scatter(X[row_ix, 0], X[row_ix, 1], label=label)
plt.legend()
plt.title("Dataset d'apprentissage généré par sklearn.datasets")
plt.show()


#svm_predict
def svm_predict( X,w):
        approx = np.dot(X,w)  
        return np.sign(approx)

#methode du gradient
def svm_gradient_decent_plot(X, Y):
    w = np.zeros(len(X[0]))
    #The learning rate
    eta = 0.0001
    #how many iterations to train for
    epochs = 10000
    # store errors 
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
        

    # plot le taux d'erreur
    plt.plot(errors,'|')
    plt.ylim(0.5,1.5)
    plt.axes().set_yticklabels([])
    plt.xlabel('Epoch')
    plt.ylabel('Erreurs')
    plt.show()
    
    return w




w = svm_gradient_decent_plot(X,y)

for d, sample in enumerate(X):
    
    # Plot the negative points
    if y[d]<0:
        plt.scatter(sample[0], sample[1],c='red', linewidths=1)
    # Plot the positivecoordinates
    else:
        plt.scatter(sample[0], sample[1], c='blue', linewidths=1)

ax=plt.gca()
xlim=ax.get_xlim()
ylim=ax.get_ylim()
# Poids de l'hyperplan
x2=[w[0],w[1],-w[1],w[0]]
x3=[w[0],w[1],w[1],-w[0]]

# linspace 
xx = np.linspace(xlim[0], xlim[1])
Z=(-w[0]/w[1])*xx
plt.plot(xx,Z, color='blue')


#put some test data in the predict function and plot them with the correct
#symbol on the graph

X_test, y_test = make_blobs(n_samples=20, centers=10, random_state=1) 

predicted_classes=[]

for point in X_test:
    predicted_classes.append(svm_predict(point, w))

for d, sample in enumerate(X_test):
    
    # Plot the negative points
    if predicted_classes[d]<0:
        plt.scatter(sample[0], sample[1], facecolors='none', edgecolors='r', linewidths=2)
    # Plot the positivecoordinates
    else:
        plt.scatter(sample[0], sample[1], facecolors='none', edgecolors='b', linewidths=2)

# retracer l'hyperplan pour la nouvelle figure test
ax=plt.gca()
# acquisition des limites de la figure
xlim=ax.get_xlim()
ylim=ax.get_ylim()
#linspace
xx = np.linspace(xlim[0], xlim[1])
Z=(-w[0]/w[1])*xx #equation de l'hyperplan

class_1 = mpatches.Patch(color='red', label="-1")
class_2 = mpatches.Patch(color='blue', label='1')
plt.plot(xx,Z, color='blue')
plt.legend(handles=[class_1,class_2])
plt.title("Classification SVM (train,test)=(80,20) ")
plt.show()
