import math
import os
from csv import reader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset


def func(clas):
    if clas == "Iris-setosa":
        return 0
    if clas == "Iris-versicolor":
        return 1
    else:
        return 2


def transform(df):
    l = len(df)
    for i in range(0, l):
        df[i][4] = func(df[i][4])
    return df


def mean(df, c):
    s = 0.0
    for i in range(len(df)):
        s = s+df[i][c]
    s = s/len(df)
    return s


def var(df, c):
    s = 0
    for i in range(len(df)):
        s = s + (df[i][c]*df[i][c])
    s = s/len(df)
    return s
def normalize(df):
    c = len(df[0])
    for i in range(c-1):
        m = mean(df, i)
        v = var(df, i)
        v = v - (m*m)
        v = math.sqrt(v)
        for j in range(len(df)):
            df[j][i] -= m
        for j in range(len(df)):
            df[j][i] = df[j][i]/v
    return df


def back_elem(x_train, y_train, x_test, y_test, hl, lr, acc_org):
    tv = 0.5
    mv = acc_org
    while x_train.shape[1] > 1 and mv > tv:
        am = []
        for cn in x_train.columns:
            xtm = x_train.drop(cn, axis=1)
            x_test_m = x_test.drop(cn, axis=1)
            bclf = MLPClassifier(batch_size=bs, solver='lbfgs', learning_rate_init=lr, hidden_layer_sizes=hl)
            bclf.fit(xtm, y_train)
            pred = bclf.predict(x_test_m)
            acc = accuracy_score(y_test, pred)
            am.append(acc)
        mv = max(am)
        mi = am.index(mv)
        if mv > tv:
            x_train = x_train.drop(x_train.columns[mi], axis=1)
            x_test = x_test.drop(x_test.columns[mi], axis=1)

    print('The Best Features Are: ', x_train.columns)
    print('Accuracy with above Features: ', mv)




if __name__ == '__main__':
    os.rename('iris.data', 'iris.csv')
    df = load_csv('iris.csv')
    df = transform(df)
    for i in range(len(df)):
        for j in range(len(df[0])-1):
            df[i][j] = float(df[i][j])
    print('###########normalization of data started##########')
    df = normalize(df)
    df = np.array(df)
    x = df[:, 0:4]
    y = df[:, 4:5]
    print('###############Linear SVM classifier##############')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    clf = svm.SVC(kernel='linear', C=0.5)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('SVM accuracy for linear kernel: ', accuracy_score(y_test, y_pred))
    print('\n')
    print('###############Quadratic SVM classifier###########')
    clf = svm.SVC(kernel='poly', degree=2, C=0.5)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('SVM accuracy for quadratic kernel: ', accuracy_score(y_test, y_pred))
    print('\n')
    print('###################Radial SVM classifier##########')
    clf = svm.SVC(kernel='rbf', degree=2, C=0.5)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('SVM accuracy for radial kernel: ', accuracy_score(y_test, y_pred))
    print('\n')
    lr = 0.001
    bs = 32
    print('##############MLP classifier###################')
    mlp_clf1 = MLPClassifier(batch_size=bs, solver='lbfgs', learning_rate_init=lr, hidden_layer_sizes=(16, ))
    mlp_clf1.fit(x_train, y_train)
    y_pred = mlp_clf1.predict(x_test)
    comp1 = accuracy_score(y_test, y_pred)
    print("The mlp accuracy for batch size 32, with 1 hidden layer and 16 nodes: ", comp1)
    mlp_clf2 = MLPClassifier(batch_size=bs, solver='lbfgs', learning_rate_init=lr, hidden_layer_sizes=(256, 16))
    mlp_clf2.fit(x_train, y_train)
    y_pred = mlp_clf2.predict(x_test)
    comp2 = accuracy_score(y_test, y_pred)
    print("The mlp accuracy for batch size 32, with 2 hidden layers - 256 and 16 respectively: ", comp2)
    lr = 0.1
    q = []
    best = ()
    if comp1 > comp2:
        best = (16,)
    else:
        best = (256, 16)

    print('#############learning rate vs Accuracy###############')
    for i in range(5):
        mlp_clf = MLPClassifier(batch_size=bs, solver='lbfgs', learning_rate_init=lr, hidden_layer_sizes=best)
        mlp_clf.fit(x_train, y_train)
        y_pred = mlp_clf.predict(x_test)
        q.append(accuracy_score(y_test, y_pred))
        lr = lr/10
    print('learning rate    -     accuracy')
    kkjh = 0.1
    for i in range(5):
        print(kkjh, '             -    ', q[i])
        kkjh = kkjh/10
    print('\n')


    # x-coordinates of left sides of bars
    left = [0.1, 0.01, 0.001, 0.0001, 0.00001]





    # plotting a bar chart
    plt.plot(left, q, color='green', linestyle='dashed', linewidth=3,
             marker='o', markerfacecolor='blue', markersize=12)

    # naming the x-axis
    plt.xlabel('Learning Rate')
    # naming the y-axis
    plt.ylabel('Accuracy')
    # plot title
    plt.title('Learning Rate vs Accuracy')

    # function to show the plot
    plt.savefig('LearningRate_vs_Accuracy.png')

    qw = pd.read_csv("iris.csv", header=None, delimiter=',')
    qw.columns = ['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm', 'class']
    r = qw.iloc[:, 0:4]
    t = qw.iloc[:, -1]
    r_train, r_test, t_train, t_test = train_test_split(r, t, test_size=0.2, random_state=1)
    #bfs = SequentialFeatureSelector(RandomForestClassifier(n_jobs=-1), k_features=(1, 4), forward=False, floating=False, verbose=2, scoring='accuracy', cv=5).fit(r_train, t_train)
    print('\n############################The Best Features#########################')
    if comp1 > comp2:
        comp2 = comp1
    back_elem(r_train, t_train, r_test, t_test, best, 0.001, comp2)
    # print(bfs.k_feature_names_)
    clf1 = svm.SVC(kernel='poly', degree=2)
    clf2 = svm.SVC(kernel='rbf', degree=2)
    clf3 = MLPClassifier(batch_size=bs, solver='lbfgs', learning_rate_init=lr, hidden_layer_sizes=(16, ))

    eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], weights=[1, 1, 1])
    eclf.fit(x_train, y_train)
    y_pred = eclf.predict(x_test)
    print('\n##########################Maximum Vote Classifier#################################')
    print('Accuracy for Max vote classifier - ', accuracy_score(y_test, y_pred))




