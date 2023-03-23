import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

from google.colab import drive

# File input
def read_in_data():
    drive.mount('/content/drive')
    with open('/content/drive/My Drive/Colab Notebooks/cosmic_sleuth/data/all_data.txt', 'r') as f:
      temp = np.genfromtxt(f,delimiter=' ')

    X = temp[:, :-3]
    y = temp[:, -3:]

    print(X.shape)

    return X, y

# Find all unique x,y,x coordinates, assign them a label
def enumerate_unique_labels(y):
    y_options = np.unique(y, axis=0)
    # Convert x,y,z coordinate into an enumerated label
    y_labels = np.array([np.where(y_options == yi)[0][0] for yi in y])
    return y_labels

def eval_classifier(clf, X_train, X_test, y_train, y_test):
    clf = clf.fit(X_train, y_train)

    print(clf.score(X_test, y_test))

def mlp():
    # Create the model
    model = Sequential()
    model.add(Dense(6, input_dim = 6, activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

def eval_mlp(model, X_train, X_test, y_train, y_test):
    # Convert target classes to categorical ones
    Y_train = to_categorical(y_train)
    Y_test = to_categorical(y_test)

    model.fit(X_train, Y_train, epochs=10, batch_size=45, verbose=1)
    # Test the model after training
    test_results = model.evaluate(X_test, Y_test, verbose=1)
    print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')

def main():
    X, y = read_in_data()
    y_labels = enumerate_unique_labels(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.1)

    # Evaluate many different classifiers
    print('Decision Tree Class.:', end='\t')
    eval_classifier(DecisionTreeClassifier(), X_train, X_test, y_train, y_test)
    print('1-NN Classifier:', end='\t')
    eval_classifier(KNeighborsClassifier(n_neighbors=1), X_train, X_test, y_train, y_test)
    print('3-NN Classifier:', end='\t')
    eval_classifier(KNeighborsClassifier(n_neighbors=3), X_train, X_test, y_train, y_test)
    print('10-NN Classifier:', end='\t')
    eval_classifier(KNeighborsClassifier(n_neighbors=10), X_train, X_test, y_train, y_test)
    print('GaussianNB Classifier:', end='\t')
    eval_classifier(GaussianNB(), X_train, X_test, y_train, y_test)
    print('LinearRegression:', end='\t')
    eval_classifier(LinearRegression(), X_train, X_test, y_train, y_test)
    print('LogisticRegression:', end='\t')
    eval_classifier(LogisticRegression(), X_train, X_test, y_train, y_test)
    print('Perceptron Class.:', end='\t')
    eval_classifier(Perceptron(), X_train, X_test, y_train, y_test)

    eval_mlp(mlp(), X_train, X_test, y_train, y_test)

main()
