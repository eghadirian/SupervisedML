# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

def logreg(c):
    iris_dataset = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'][:100],
                                                        iris_dataset['target'][:100], random_state=0)
    x = []
    y = []
    for i in range(c):
        reg = LogisticRegression(C=i+1, penalty='l2') #L2 regularization, can be changed to L1
        reg.fit(X_train, y_train)
        x.append(reg.score(X_train, y_train))
        y.append(reg.score(X_test, y_test))
    I = [i+1 for i in range(c)]
    plt.figure()
    plt.plot(I,x,'r',I,y,'k')
    plt.show()

if __name__=='__main__':
    logreg(int(input('Enter C:')))