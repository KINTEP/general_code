import numpy as np
from scipy.integrate import simps

def detrend(sequence):
    "This takes a sequence and outputs the difference between two successive elements"
    diffs = [0]
    for i in range(1, len(sequence)):
        prev = i - 1
        now = i
        diffs.append(sequence[prev] - sequence[now])
    return diffs

def detrend2(sequence):
    "This takes a sequence and outputs the difference between two successive elements"
    diffs = [0]
    for i in range(1, len(sequence)):
        prev = i - 1
        now = i
        diffs.append(np.exp(sequence[prev] - sequence[now]))
    return diffs

def normalize(sequence):
	seq = np.array(sequence) + 1
	max1 = np.max(seq)
	min1 = np.min(seq)
	data = []
	for val in seq:
		norm = (val - min1)/(max1 - min1)
		data.append(norm)
	return np.array(data)


def lags(n, data):
    "This function takes in sequence and outputs the lag periods as a matrix"
    all_data = []
    N = len(data)
    start = n-1
    for i in range(n):
        ls = data[start:N-i]
        all_data.append(ls)
        start -= 1
    return all_data

def extractOHLC(open1, high, low, close):
    "This function extracts open, high, low, close of a financial time series on the same level and outputs as a list"
    N = len(close)
    list_data = []
    for i in range(N):
        list_data.append([open1[i], high[i], low[i], close[i]])
    return list_data


def ratios(sequence):
    "This takes a sequence and outputs the difference between two successive elements"
    diffs = [0]
    for i in range(1, len(sequence)):
        prev = i - 1
        now = i
        diffs.append(((sequence[prev])+1)/((sequence[now]+1)))
    return diffs

def volume(data):
    norm = normalize(data)
    y = np.array(norm)*np.array(norm)
    y1 = np.pi * y
    return simps(y1)

def divide(seq, rate):
    N = int(len(seq)/rate)
    start = 0
    end = rate
    data = []
    for i in range(N):
        ls = seq[start:end]
        data.append(ls)
        start = end
        end = start + rate
    return data

def frequency_distribution(data, n):
    count, rang = np.histogram(data, bins = n)
    sum1 = np.sum(count)
    return count, rang[1:]

def standardize(list1):
    std = np.std(list1)
    mean = np.mean(list1)
    standad = []
    for v in list1:
        z = (v - mean)/std
        standad.append(z)
    return standad

def first_difference(data):
    diff = []
    for i in range(1, len(data)):
        diff.append(data[i] - data[i-1])
    return np.array(diff)

def higher_derivatives(data, n):
    differences = []
    data1 = normalize(data)
    for i in range(n):
        diff1 = first_difference(data=data1)
        differences.append(diff1)
        data1 = normalize(diff1)
    return differences

def fit_curve(data, loop, init=0):
    freq1 = []
    N = len(data)
    time = np.arange(N)
    res1 = init[:]
    max_corrs = []

    for _ in range(loop):
        r1 = np.random.randn()
        cosa1 = np.cos(r1*time)[1:]
        new_res = res1 + cosa1
        corr1 = correlate(x=new_res,y=data[1:])
        corr2 = correlate(x=res1, y=data[1:])
        if corr1 > corr2:
            res1 += cosa1
            max_corrs.append(corr1)
            freq1.append(r1)
    return max_corrs, res1, freq1


def derivative():
    from derivative import dxdt

    N = len(determinants)
    t = np.linspace(1,N,N)
    x = determinants

    # 1. Finite differences with central differencing using 3 points.
    result1 = dxdt(x, t, kind="finite_difference", k=1)

    # 2. Savitzky-Golay using cubic polynomials to fit in a centered window of length 1
    result2 = dxdt(x, t, kind="savitzky_golay", left=.5, right=.5, order=5)

    # 3. Spectral derivative
    result3 = dxdt(x, t, kind="spectral")

    # 4. Spline derivative with smoothing set to 0.01
    result4 = dxdt(x, t, kind="spline", s=1e-2)

    # 5. Total variational derivative with regularization set to 0.01
    #result5 = dxdt(x, t, kind="trend_filtered", order=0, alpha=1e-2)

def rejection_sampling(r=1):
    while True:
        x = np.random.randn()*2 - 1
        y = np.random.randn()*2 - 1
        if x**2 + y**2 < r:
            return x+y*1j

def random_polar():
    theta = np.random.randn()*2*np.pi
    r = np.random.randn()
    return r*np.cos(theta)+r*np.sin(theta)*1j

def dist_from_triang():
    theta = np.random.randn()*2*np.pi 
    r = np.random.randn() + np.random.randn()
    if r >= 1:
        r = 2 - r
    return r*np.cos(theta) + r*np.sin(theta)*1j

def zeta(x, st=1, n=100000):
    z = 0
    zer = []
    for i in range(st, n):
        den = i**x
        ans = 1/den
        z += ans
        zer.append(ans)
    return z, zer


class Parameterize:
    def __init__(self, data):
        self.data = data
    
    def step1(self, data):
        res = []
        if len(data) < 1:
            res.append(0)
        if len(data) == 1:
            res.append(data[0])
        for i in range(1, len(data)):
            res.append(data[i] - data[i-1])
        return res
    
    def params(self):
        final = []
        N = len(self.data)
        ray1 = self.data
        for i in range(N-1):
            ans1 = self.step1(ray1)
            ans2 = ans1[0]
            final.append(ans2)
            ray1 = ans1
        return [self.data[0]] + final

    def rec(self, lst, up):
        res = [up]
        for val in lst:
            ans = res[-1]+val
            res.append(ans)
        return res

    def recover(self, ups):
        results = [[ups[-1]]]
        N = len(ups) - 1
        for i in range(N):
            ans = self.rec(lst=results[-1], up = ups[N-(i+1)])
            results.append(ans)
        return results[-1]