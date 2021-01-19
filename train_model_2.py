import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
#import numpy as np


pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0 #normalize ,, scale
#X= np.array(X)
#y= np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=42)

model = svm.SVC()
print(X_train.shape)
nsamples, nx, ny, nz = X_train.shape
d2_train_dataset = X_train.reshape((nsamples,nx*ny*nz))

model.fit(d2_train_dataset,y_train)

nsamples, nx, ny, nz = X_test.shape
d2_test_dataset = X_test.reshape((nsamples,nx*ny*nz))

y_pred=model.predict(d2_test_dataset)

accuracy= model.score(d2_test_dataset ,y_test)
print('Model accuracy is: ', accuracy) #0.8913043478260869
