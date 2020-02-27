import numpy as np
from pandas import read_csv

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.lda import LDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

def main():

    data = read_csv('e.csv')
    del data['time']

    Y = np.array(data['spam'])

    Z = np.array(data['winner'])
    print Z
    for i in range(50):
        if Z[i] == "yes":
            print i+1

    del data['spam']
    del data['number']
    del data['winner']
    X = np.array(data)

    # SVC
    clf = SVC()
    clf.fit(X, Y)
    cnt = 0
    for i in range(50):
        res = clf.predict([X[i]])
        if int(res) != Y[i]:
            cnt = cnt + 1
            print i,
    print "Support Vector Classification - ", cnt

    # KNeighborsClassifier
    neigh1 = KNeighborsClassifier(n_neighbors=1)
    neigh1.fit(X, Y)
    cnt = 0
    for i in range(50):
        res = neigh1.predict([X[i]])
        if int(res) != Y[i]:
            cnt = cnt + 1
            print i,
    print "k-nearest neighbors (k = 1) - ", cnt

    neigh5 = KNeighborsClassifier(n_neighbors=2)
    neigh5.fit(X, Y)
    cnt = 0
    for i in range(50):
        res = neigh5.predict([X[i]])
        if res != Y[i]:
            cnt = cnt + 1
            print i,
    print "k-nearest neighbors (k = 2) - ", cnt

    neigh3 = KNeighborsClassifier(n_neighbors=3)
    neigh3.fit(X, Y)
    cnt = 0
    for i in range(50):
        res = neigh3.predict([X[i]])
        if res != Y[i]:
            cnt = cnt + 1
            print i,
    print "k-nearest neighbors (k = 3) - ", cnt

    neigh5 = KNeighborsClassifier(n_neighbors=5)
    neigh5.fit(X, Y)
    cnt = 0
    for i in range(50):
        res = neigh5.predict([X[i]])
        if res != Y[i]:
            cnt = cnt + 1
            print i,
    print "k-nearest neighbors (k = 5) - ", cnt

    neigh8 = KNeighborsClassifier(n_neighbors=8)
    neigh8.fit(X, Y)
    cnt = 0
    for i in range(50):
        res = neigh8.predict([X[i]])
        if res != Y[i]:
            cnt = cnt + 1
            print i,
    print "k-nearest neighbors (k = 8) - ", cnt

    neigh10 = KNeighborsClassifier(n_neighbors=10)
    neigh10.fit(X, Y)
    cnt = 0
    for i in range(50):
        res = neigh10.predict([X[i]])
        if res != Y[i]:
            cnt = cnt + 1
            print i,
    print "k-nearest neighbors (k = 10) - ", cnt

    # LinearRegression
    lin = LinearRegression()
    lin.fit(X, Y)
    cnt = 0
    for i in range(50):
        res = lin.predict([X[i]])
        if int(res) != Y[i]:
            cnt = cnt + 1
            print i,
    print "Linear Regression - ", cnt

    # Random forest classifier
    tr = RandomForestClassifier()
    tr.fit(X, Y)
    cnt = 0
    for i in range(50):
        res = tr.predict([X[i]])
        if int(res) != Y[i]:
            cnt = cnt + 1
            print i,
    print "Random forest classifier - ", cnt

    # LDA
    lda = LDA()
    lda.fit(X, Y)
    cnt = 0
    for i in range(50):
        res = lda.predict([X[i]])
        if int(res) != Y[i]:
            cnt = cnt + 1
            print i,
    print "LDA - ", cnt

    # DecisionTreeClassifier
    dts = DecisionTreeClassifier()
    dts.fit(X, Y)
    cnt = 0;
    for i in range(50):
        res = dts.predict([X[i]])
        if int(res) != Y[i]:
            cnt = cnt + 1
            print i,
    print "DecisionTreeClassifier - ", cnt

    # AdaBoost
    AD = AdaBoostClassifier()
    AD.fit(X, Y)
    cnt = 0;
    for i in range(50):
        res = AD.predict([X[i]])
        if int(res) != Y[i]:
            cnt = cnt + 1
            print i,
    print "AdaBoostClassifier - ", cnt

main()