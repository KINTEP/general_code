#Discrete fourier transform
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def dft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

#Gaussian


n = 50
x = np.linspace(0, 10, n)

# Define the gaussian with mu = sin(x) and negligible covariance matrix
norm = stats.multivariate_normal(mean=np.sin(x), cov=np.eye(n) * 1e-6)
plt.figure(figsize=(16, 6))

# Taking a sample from the distribution and plotting it.
plt.plot(x, norm.rvs())

#Part 2
norm = stats.multivariate_normal(mean=np.zeros(n), cov=np.eye(n))
plt.figure(figsize=(16, 6))

# Taking 3 sample from the distribution and plotting it.
[plt.plot(x, norm.rvs()) for _ in range(3)]


#part 3
def kernel(m1, m2, l=1):
    return np.exp(- 1 / (2 * l**2) * (m1[:, None] - m2)**2)

n = 50
x = np.linspace(0, 10, n)
cov = kernel(x, x, 0.44)

norm = stats.multivariate_normal(mean=np.zeros(n), cov=cov)
plt.figure(figsize=(16, 6))
[plt.plot(x, norm.rvs()) for _ in range(3)]


def random_walk(N):
    """
    Simulates a discrete random walk
    :param int N : the number of steps to take
    """
    # event space: set of possible increments
    increments = np.array([1, -1])
    # the probability to generate 1
    p=0.5

    # the epsilon values
    random_increments = np.random.choice(increments, N, p)
    # calculate the random walk
    random_walk = np.cumsum(random_increments)

    # return the entire walk and the increments
    return random_walk, random_increments

# generate a random walk
N = 500
X, epsilon = random_walk(N)

# normalize the random walk using the Central Limit Theorem
X = X * np.sqrt(1./N)


def brownian_motion(N, T, h):
    """
    Simulates a Brownian motion
    :param int N : the number of discrete steps
    :param int T: the number of continuous time steps
    :param float h: the variance of the increments
    """
    dt = 1. * T/N  # the normalizing constant
    random_increments = np.random.normal(0.0, 1.0 * h, N)*np.sqrt(dt)  # the epsilon values
    brownian_motion = np.cumsum(random_increments)  # calculate the brownian motion
    brownian_motion = np.insert(brownian_motion, 0, 0.0) # insert the initial condition

    return brownian_motion, random_increments

N = 50 # the number of discrete steps
T = 1 # the number of continuous time steps
h = 1 # the variance of the increments
dt = 1.0 * T/N  # total number of time steps

# generate a brownian motion
X, epsilon = brownian_motion(N, T ,h)


def geometric_brownian_motion(G0, mu, sigma, N, T):
    """Simulates a Geometric Brownian Motion.

    :param float G0: initial value
    :param float mu: drift coefficient
    :param float sigma: diffusion coefficient
    :param int N: number of discrete steps
    :param int T: number of continuous time steps
    :return list: the geometric Brownian Motion
    """
    # the normalizing constant
    dt = 1. * T/N
    # standard brownian motion
    W, _ = brownian_motion(N, T ,1.0)
    # generate the time steps
    time_steps = np.linspace(0.0, N*dt, N+1)

    # calculate the geometric brownian motion
    G = G0 * np.exp(mu * time_steps + sigma * W)
    # replace the initial value
    G[0] = G0

    return G


  def gbm_mean(G0, mu, sigma, N, T):
    """Simulates the mean of the Geometric Brownian Motion, which is:
        E(t) = G0*e^{(mu + sigma^{2}/2)*t}
    :param float G0: initial value
    :param float mu: drift coefficient
    :param float sigma: diffusion coefficient
    :param int N: number of discrete steps
    :param int T: number of continuous time steps
    """
    # generate the time steps
    t = np.linspace(0.0, T, N+1)
    # calculate the mean
    E = G0 * np.exp((mu + 0.5*sigma**2)*t)

    return E

def gbm_var(G0, mu, sigma, N, T):
    """Simulates the variance of the Geometric Brownian Motion, which is:
        Var(t) = G0^2 * e^{(2*mu + sigma^{2})*t} * (e^{sigma^{2}*t} - 1)
    :param float G0: initial value
    :param float mu: drift coefficient
    :param float sigma: diffusion coefficient
    :param int N: number of discrete steps
    :param int T: number of continuous time steps
    """
    # generate the time steps
    t = np.linspace(0.0, T, N+1)
    # calculate the variance
    V = G0**2 * np.exp(t * (2*mu + sigma**2)) * (np.exp(t * sigma**2) - 1)

    return V


def calculate_integral(f, a, b, n):
    '''Calculates the integral based on the composite trapezoidal rule
    relying on the Riemann Sums.

    :param function f: the integrand function
    :param int a: lower bound of the integral
    :param int b: upper bound of theintergal
    :param int n: number of trapezoids of equal width
    :return float: the integral of the function f between a and b
    '''
    w = (b - a)/n
    result = 0.5*f(a) + sum([f(a + i*w) for i in range(1, n)]) + 0.5*f(b)
    result *= w
    return result


def get_gradient_at_b(x, y, b, m):
  N = len(x)
  diff = 0
  for i in range(N):
    x_val = x[i]
    y_val = y[i]
    diff += (y_val - ((m * x_val) + b))
  b_gradient = -(2/N) * diff  
  return b_gradient

def get_gradient_at_m(x, y, b, m):
  N = len(x)
  diff = 0
  for i in range(N):
      x_val = x[i]
      y_val = y[i]
      diff += x_val * (y_val - ((m * x_val) + b))
  m_gradient = -(2/N) * diff  
  return m_gradient

#Your step_gradient function here
def step_gradient(b_current, m_current, x, y, learning_rate):
    b_gradient = get_gradient_at_b(x, y, b_current, m_current)
    m_gradient = get_gradient_at_m(x, y, b_current, m_current)
    b = b_current - (learning_rate * b_gradient)
    m = m_current - (learning_rate * m_gradient)
    return [b, m]
  
#Your gradient_descent function here:
def gradient_descent(x,y,learning_rate,num_iterations):
  b = 0
  m = 0
  for i in range(num_iterations):
    b,m = step_gradient(b, m, x, y, learning_rate)
  return b,m

months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
revenue = [52, 74, 79, 95, 115, 110, 129, 126, 147, 146, 156, 184]

#Uncomment the line below to run your gradient_descent function
b, m = gradient_descent(months, revenue, 0.01, 1000)