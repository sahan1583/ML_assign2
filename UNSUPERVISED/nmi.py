import math


def find_h_c(s, a, k):
    f = 0.0
    for i in range(k):
        f += (a[i]/s)*(math.log(a[i]/s, 2))
    return f*-1


def find_h_y(a):
    z = 0
    o = 0
    t = 0
    for i in range(len(a)):
        if a[i] == 0:
            z += 1
        elif a[i] == 1:
            o += 1
        else:
            t += 1
    l = len(a)
    sum = 0
    if z != 0:
       sum += (-z*math.log(z/l, 2))
    if o != 0:
        sum += (-o*math.log(o/l, 2))
    if t != 0:
        sum += (-t*math.log(t/l, 2))
    return sum/150


def find_nmi(b, k):
    a = []
    s = 0
    for i in range(k):
        a.append(len(b[i]))
        s += len(b[i])
    hc = find_h_c(s, a, k)
    a = []
    s = 0
    hy = -(math.log(1/3, 2))
    den = hc+hy
    i_c_y = hy
    for i in range(k):
        i_c_y -= find_h_y(b[i])
    return abs((2*i_c_y)/(hc+hy))
