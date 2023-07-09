import math
from nmi import find_nmi
import numpy as np
def check(a, b):
    # print("hi")
    for i in range(len(a)):
        if (a[i][0] - b[i][0] > (0.0001)) or (a[i][1] - b[i][1] > (0.0001)):
            return False
    return True


def dist(a, b, c, d):
    return math.sqrt(((a-c)*(a-c))+((b-d)*(b-d)))


def find_centroid(ds, b):
    l = len(b)
    sx = 0
    sy = 0
    for i in range(l):
        sx = sx+ds[b[i]][0]
        sy = sy+ds[b[i]][1]
    sx = sx/l
    sy = sy/l
    return sx, sy


def find_points(ds, k):
    ar = []
    m = int(149/k)
    n = m
    for i in range(k):
        ar.append(m)
        m = m+n

    a = [[0.0]*2 for _ in range(k)]
    for i in range(k):
        a[i][0] = ds[ar[i]][0]
        a[i][1] = ds[ar[i]][1]
    b = [[] for _ in range(k)]
    for i in range(150):
        c = 100
        ind = 0
        for j in range(k):
            d = dist(ds[i][0], ds[i][1], a[j][0], a[j][1])
            if c > d:
                c = d
                ind = j
        b[ind].append(i)     #not sure
    nar = [[0.0]*2 for _ in range(k)]
    qw = 0
    for i in range(len(b)):
        nar[qw][0], nar[qw][1] = find_centroid(ds, b[i])
        qw = qw+1
    while 1:
        if check(nar, a):
            break
        for i in range(k):
            b[i].clear()
        a = nar
        for i in range(150):
            c = 100
            ind = 0
            for j in range(k):
                d = dist(ds[i][0], ds[i][1], a[j][0], a[j][1])
                if c > d:
                    c = d
                    ind = j
            b[ind].append(i)  # not sure
        qw = 0
        for i in b:
            nar[qw][0], nar[qw][1] = find_centroid(ds, i)
            qw = qw + 1
    sumi = 0
    for i in b:
        sumi += len(i)
    for i in range(k):
        for j in range(len(b[i])):
            if b[i][j] <= 49:
                b[i][j] = 0
            elif b[i][j] <= 99:
                b[i][j] = 1
            else:
                b[i][j] = 2
    return find_nmi(b, k)







