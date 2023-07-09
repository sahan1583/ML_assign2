import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from csv import reader
from k_cluster import find_points


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


def autolabel(rects):
    """
    Attach a text label above each bar, displaying its height.
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate('%.3f' % height,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
if __name__ == '__main__':
    os.rename('iris.data', 'iris.csv')
    df = load_csv('iris.csv')
    df = transform(df)
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)
    pca = PCA(n_components=4)
    ds = [[i[j] for j in range(len(i)-1)] for i in df]
    pca.fit(ds)
    var = pca.explained_variance_ratio_[:]
    labels = ['PC' + str(i + 1) for i in range(len(var))]

    fig, ax = plt.subplots(figsize=(15, 7))
    plot1 = ax.bar(labels, var)

    ax.plot(labels, var)
    ax.set_title('Proportion of Variance Explained VS Principal Component')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Proportion of Variance Explained')
    autolabel(plot1)

    cs = [i for i in var]
    nc = -1
    for i in range(1, len(cs)):
        cs[i] += cs[i - 1]
        if cs[i] >= 0.95 and nc == -1:
            nc = i + 1
    plt.savefig('Proportion_vs_pca.png')
    fig, ax = plt.subplots(figsize=(15, 7))
    plot2 = ax.bar(labels, cs)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Variance Ratio cumulative sum')
    ax.set_xlabel('number principal components')
    ax.set_title('Variance Ratio cumulative sum VS number principal components')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)

    ax.axvline('PC' + str(nc), c='red')
    ax.axhline(0.95, c='green')
    ax.text('PC5', 0.95, '0.95', fontsize=15, va='center', ha='center', backgroundcolor='w')
    autolabel(plot2)
    print('############################plot for proportion vs PCA plotted############################')
    print('############################plot for varience cumilative sum vs number of pca plotted############')
    print('Number of Components selected:{}'.format(nc))
    print('Variance captured: {}'.format(cs[nc - 1]))
    pca = PCA(n_components=nc)
    dataset = pca.fit_transform(ds).tolist()
    for i in range(len(dataset)):
        if(df[i][4] == 0.):
            dataset[i].append(0)
        elif(df[i][4] == 0.5):
            dataset[i].append(1)
        else:
            dataset[i].append(2)

    plt.savefig('Var_CumSum_vs_NumbPca.png')
    w = []
    for i in range(2, 9):
        w.append(find_points(dataset, i))
    left = [2, 3, 4, 5, 6, 7, 8]

    # labels for bars
    tick_label = ['two', 'three', 'four', 'five', 'six', 'seven', 'eight']
    fig, ax = plt.subplots(figsize=(15, 7))
    plot3 = ax.bar(left, w, tick_label=tick_label, width=0.8, color=['green'])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Values of K')
    ax.set_ylabel('Accuracy')
    ax.set_title('K Vs Accuracy')

    autolabel(plot3)
    print('#################Plot of K and Accuracy is plotted################')

    plt.savefig('K_vs_Accuracy.png')
    print('#################Printing k vs Accuracy#################')
    print('k    -    Accuracy')
    for i in range(2, 9):
        print(i, '   -   ', w[i-2])
    print('SO FOR K VALUE AS 4 WE HAVE HIGHEST ACCURACY')












