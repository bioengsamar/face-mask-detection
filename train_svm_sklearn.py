import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np


def load_data(x, y):
    X = pickle.load(x)
    y = pickle.load(y)
    X = X/255.0 #normalize ,, scale
    nsamples, nx, ny, nz = X.shape
    d2_dataset = X.reshape((nsamples,nx*ny*nz)) #convert from 4D to 2D
    return d2_dataset, y

def model(X_train, X_test, y_train, y_tes):
    global model
    model = svm.SVC()
    model.fit(X_train,y_train)
    accuracy= model.score(X_test ,y_test)
    return accuracy



if __name__ == '__main__':
    pickle1_in = open("X.pickle","rb")
    pickle_in = open("y.pickle","rb")
    X, y= load_data(pickle1_in, pickle_in)
    #split data into 90% train and 10% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=42)
    accuracy = model(X_train, X_test, y_train, y_test)
    print('Model accuracy is: ', accuracy) #0.8913043478260869
    import cv2


    image = cv2.imread('image/44.jpg',cv2.IMREAD_GRAYSCALE)
    roi = cv2.resize(image, (100, 100))
    feature_vector = np.reshape(roi, (100*100))
    print(model.predict(feature_vector.reshape(1,-1)))
    #print(feature_vector.reshape(-1,1))
