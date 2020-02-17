from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def classifier(n_neighbor):
    iris_dataset = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train) # or X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors = n_neighbor)
    knn.fit(X_train, y_train)
    print(knn.predict(X_test)[0])

if __name__=='__main__':
    classifier(int(input('Enter n_neighbor: ')))