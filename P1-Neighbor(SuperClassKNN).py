#K-nearest neighbor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def classifier(n_neighbor):
    iris_dataset = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
    iris_dataframe=pd.DataFrame(X_train, columns=iris_dataset.feature_names)
    pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o',
                               hist_kwds={'bins':20}, s=60, alpha = 0.8)
    x = []
    y = []
    for i in range(n_neighbor):
        knn = KNeighborsClassifier(n_neighbors = i+1)
        knn.fit(X_train, y_train)
        x.append(knn.score(X_train, y_train))
        y.append(knn.score(X_test, y_test))
    I = [i+1 for i in range(n_neighbor)]
    plt.figure()
    plt.plot(I,x,'r',I,y,'k')
    plt.show()

if __name__=='__main__':
    classifier(int(input('Enter n_neighbor:')))