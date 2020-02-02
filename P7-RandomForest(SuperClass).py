#Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def randomforest(n):
    X, y = make_moons(n_samples=100, noise=0.25,random_state=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    x = []
    y = []
    for i in range(n):
        forest = RandomForestClassifier(n_estimators=i+1, random_state=2).fit(X_train, y_train)
        x.append(forest.score(X_train, y_train))
        y.append(forest.score(X_test, y_test))
    I = [i+1 for i in range(n)]
    plt.figure()
    plt.plot(I, x, 'r', I, y, 'k')
    plt.show()

if __name__=='__main__':
    randomforest(int(input('Enter number of trees:')))