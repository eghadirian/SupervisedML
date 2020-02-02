#Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

def dectree(c):
    iris_dataset = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],
                                                        iris_dataset['target'], random_state=42)
    x = []
    y = []
    for i in range(c):
        reg = DecisionTreeClassifier(random_state=0, max_depth=i+1)
        reg.fit(X_train, y_train)
        x.append(reg.score(X_train, y_train))
        y.append(reg.score(X_test, y_test))
    I = [i+1 for i in range(c)]
    plt.figure()
    plt.plot(I, x, 'r', I, y, 'k')
    plt.show()
    print('Feature importance for depth of {}: {}'.format(c, reg.feature_importances_))

if __name__=='__main__':
    dectree(int(input('Enter max depth of the tree:')))