import numpy as np 


def sinefit(data):
    N = len(data)
    n = N*N
    fit = np.zeros(N)
    time = np.array([i for i in range(1, N+1)])
    corr = np.corrcoef(data, fit)[1][0]
    counter = 1
    while corr != 1:
        corr = np.corrcoef(data, fit)[1][0]
        count = counter
        sin = np.sin(count*time)
        coef = np.dot(sin, data)
        fit += coef * sin
        counter += 1
    return fit, counter


def fourier(data):
    N = len(data)
    time = np.arange(N)
    Z = np.zeros(N)
    ampS = []
    ampC = []
    freq = []
    for i in range(N):
        f = i*2*np.pi/N
        s1 = np.sin(f*time)
        c1 = np.cos(f*time)
        a1 = np.dot(s1, data)
        b1 = np.dot(c1, data)
        Z += (a1*s1 + b1*c1)
        ampS.append(a1)
        ampC.append(b1)
        freq.append(f)
    Z = Z/N
    return Z, ampS, ampC, freq

def forcast(time, a1, b1, f1):
    N = time.shape[0]
    Z = np.zeros(N)
    for i in range(N):
        s1 = np.sin(f1[i]*2*np.pi*time/N)
        c1 = np.cos(f1[i]*2*np.pi*time/N)
        a11 = a1[i]
        b11 = b1[i]
        Z += (a11*s1 + b11*c1)
    Z = Z/N
    return Z

def fourier_matrix(data):
    N = len(data)
    time = np.arange(1, N+1)
    Z = []
    ampS = []
    ampC = []
    freq = []
    for i in range(N):
        f = i
        s1 = np.sin(f*2*np.pi*time/N)
        c1 = np.cos(f*2*np.pi*time/N)
        a1 = np.dot(s1, data)
        b1 = np.dot(c1, data)
        C = a1*s1/N + b1*c1/N
        Z.append(C)
    return np.array(Z)

