import numpy as np 
from general import frequency_distribution, normalize
from other_authors import kernel

class Compare:
    def __init__(self, seq1, seq2):
        self.seq1 = seq1
        self.seq2 = seq2
        #self.min_max_scaler = MinMaxScaler()
        
    def matrix(self, m1):
        return np.abs(m1[:, None] - m1)
    
    def dot(self):
        mat1 = self.matrix(self.seq1)
        mat2 = self.matrix(self.seq2)
        mat1 =  mat1.flatten()
        mat2 = mat2.flatten()
        return np.corrcoef(mat1, mat2)[0][1]
    
    def norm_dot(self):
        mat1 = self.matrix(self.seq1)
        mat2 = self.matrix(self.seq2)
        mat1 =  normalize(mat1.flatten())
        mat2 = normalize(mat2.flatten())
        return np.corrcoef(mat1, mat2)[0][1]
    
    def distance(self):
        mat1 = self.matrix(self.seq1)
        mat2 = self.matrix(self.seq2)
        mat1 =  mat1.flatten()
        mat2 = mat2.flatten()
        return np.abs(np.sum(mat1 - mat2))


def gradient_at_b(Y, m, x, c, l=0.01):
    result = l*(Y - m*x - c)
    return result

def gradient_at_m(Y, m, x, c, l=0.01):
    result = l*(Y - m*x - c)*x
    return result

def gradient_descent(target, inputs, l=0.01, num=1000):
    b1 = np.random.randn()
    m1 = np.random.randn()
    N = len(inputs)
    for i in range(num):
        p1 = b1*np.array(inputs) + b1
        y1 = np.array(target)
        x1 = np.array(inputs)
        b2 = 2/N * np.sum(gradient_at_b(Y=y1, m=m1, x=x1, c=b1, l=0.01))
        m2 = 2/N * np.sum(gradient_at_m(Y=y1, m=m1, x=x1, c=b1, l=0.01))
    
        b1 += b2
        m1 += m2
    return m1, b1


def Energy(data, n=100):
    N = len(data)
    min1 = min(data)
    max1 = max(data)
    rand1 = []
    for i in range(n):
        ref1 = np.random.uniform(low = min1, high = max1, size = N)
        rand1.append(ref1)
    ken1 = -1*kernel(np.array(data), np.array(data))
    dist1, _ = frequency_distribution(data = normalize(ken1.flatten()), n=N)
    dist1 = 1 + dist1
    
    ref_dists = []
    for i in range(len(rand1)):
        ken = -1*kernel(np.array(rand1[i]), np.array(rand1[i]))
        dd, _ = frequency_distribution(data = normalize(ken.flatten()), n=N)
        ref_dists.append(1+np.array(dd))
        
    ref_dists = np.array(ref_dists)
    
    E11 = np.sum(dist1/ref_dists)
    log = -1 * np.log(E11)
    return log