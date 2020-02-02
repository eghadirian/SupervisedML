# Gradient Bossted Classification Tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def gradboost(n_trees):
    iris_dataset = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
    x = []
    y = []
    for i in range(n_trees):
        knn = GradientBoostingClassifier(random_state=0, n_estimators=i+1, max_depth=1)
        knn.fit(X_train, y_train)
        x.append(knn.score(X_train, y_train))
        y.append(knn.score(X_test, y_test))
    I = [i+1 for i in range(n_trees)]
    plt.figure()
    plt.plot(I,x,'r',I,y,'k')
    plt.show()
    print(knn.decision_function(X_test)[0][0])
    print(knn.predict_proba(X_test)[0][0])

if __name__=='__main__':
    gradboost(int(input('Enter number of trees:')))
