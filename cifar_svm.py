
import os
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn import svm
# Load the CIFAR10 dataset
from keras.datasets import cifar10
baseDir = os.path.dirname(os.path.abspath('__file__')) + '/'
classesName = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
xVal = xTrain[49000:, :].astype(np.float)
yVal = np.squeeze(yTrain[49000:, :])
xTrain = xTrain[:49000, :].astype(np.float)
yTrain = np.squeeze(yTrain[:49000, :])
yTest = np.squeeze(yTest)
xTest = xTest.astype(np.float)


# Show some CIFAR10 images
plt.subplot(221)
plt.imshow(xTrain[0].astype(np.int))
plt.axis('off')
plt.title(classesName[yTrain[0]])
plt.subplot(222)
plt.imshow(xTrain[1].astype(np.int))
plt.axis('off')
plt.title(classesName[yTrain[1]])
plt.subplot(223)
plt.imshow(xVal[0].astype(np.int))
plt.axis('off')
plt.title(classesName[yVal[0]])
plt.subplot(224)
plt.imshow(xTest[0].astype(np.int))
plt.axis('off')
plt.title('cat')
plt.show()


xTrain = np.reshape(xTrain, (xTrain.shape[0], -1)) # The -1 means that the corresponding dimension is calculated from the other given dimensions.
xVal = np.reshape(xVal, (xVal.shape[0], -1))
xTest = np.reshape(xTest, (xTest.shape[0], -1))


#Choosing a smaller dataset
xTrain=xTrain[:3000,:]
yTrain=yTrain[:3000]



def svm_linear(c):
    svc = svm.SVC(C = c)
    
    svc.fit(xTrain, yTrain) 
    # Find the prediction and accuracy on the training set.
    
    Yhat_svc_linear_train = svc.predict(xTrain)
    acc_train = np.mean(Yhat_svc_linear_train == yTrain)
    acc_train_svm_linear.append(acc_train)
    print('Précision des prédictions SVM : {:.2%}'.format(acc_train))
    

def knn_fnc(k):
    knn =  KNeighborsClassifier(n_neighbors=k)
    
    knn.fit(xTrain, yTrain) 
    # Find the prediction and accuracy on the training set.
    
    Yhat_knn_train = knn.predict(xTrain)
    acc_train = np.mean(Yhat_knn_train == yTrain)
    acc_train_knn.append(acc_train)
    print('Précision des prédictions KNN :{:.2%}'.format(acc_train))

c_svm_linear = [0.0001,0.001,0.01,0.1,1,10,100]
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
acc_train_svm_linear = []
acc_train_knn = []

for c in c_svm_linear:
    svm_linear(c)
for k in k_choices:
    knn_fnc(k)

plt.subplot(1,2,1)
plt.plot(c_svm_linear, acc_train_svm_linear,'.-',color='red', label='SVM')
plt.xlabel('c')
plt.ylabel('Précision en %')
plt.title("SVM: précision en fonction du facteur C")
plt.grid()
plt.xticks(c_svm_linear,c_svm_linear)
plt.legend()
plt.subplot(1,2,2)
plt.plot(k_choices, acc_train_knn,'.-',color='orange',label='KNN')
plt.xlabel('k')
plt.ylabel('Précision en %')
plt.legend()
plt.title("KNN: précision en fonction du facteur K")
plt.grid()
plt.show()


