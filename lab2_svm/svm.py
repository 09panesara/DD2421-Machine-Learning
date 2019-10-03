import numpy as np, random , math
from scipy . optimize import minimize
import matplotlib . pyplot as plt

ts = None # target values
xs = None # input values corresponding to target values
N = None # size of training data
K = None # precomputed Kernel function
P = None # precomputed ti tj K[i,j] matx
non_zero_as, non_zero_xs, non_zero_ts = None, None, None # support vectors
b = None

def compute_objective_matx(t, x):
    ''' Precomputes P
        P[i,j] = t[i][t[j] K(x[i]x[j])
    '''
    global P, K
    n = len(t)
    K = np.asarray([[kernel(x[i], x[j]) for j in range(n)] for i in range(n)])
    P = np.asarray([[t[i] * t[j] * K[i, j] for j in range(n)] for i in range(n)])

def generate_data(plot=False):
    global xs, ts
    np.random.seed(100) # generate same data each time
    classA = np.concatenate((np.random.randn(10, 2) * 0.2 +[1.5, 0.5], np.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))
    classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]
    if plot:
        plot_data(classA, classB)
    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))
    N = inputs.shape[0]  # Number of rows (samples)
    # randomly reorder samples
    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]
    xs = inputs
    N = len(xs)
    ts = targets

def plot_data(classA, classB):
    plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
    plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
    plt.axis('equal')  # Force same scale on both axes plt.savefig(’svmplot.pdf’) # Save a copy in a file plt .show() # Show the plot on the screen
    plt.savefig('plots/svmplot.pdf')
    plt.show()

def _euclidean_squared(x, y):
    return np.sum([(x[i]-y[i])**2 for i in range(len(x))])

def kernel(x, y, kernel_type='linear', p=2, sigma=0.01):
    if kernel_type == 'linear':
        return np.dot(x, y)
    elif kernel_type == 'polynomial':
        return (np.dot(x, y) + 1)**p
    elif kernel_type == 'rbf': # radial basis function
        return math.e**(-_euclidean_squared(x,y)/(2*sigma**2))
    else:
        raise Exception(f'Cannot recognise kernel type {kernel_type}')

def objective(alpha):
    ''' Computes objective function using kernel trick (see eqn 4 in lab notes)
    '''
    n = len(alpha)
    alpha_mat = np.array([a_i*alpha for a_i in alpha])
    # TODO remove for testing purposes only
    correct_sum = np.sum([[alpha[i] * alpha[j] * P[i,j] for j in range(n)] for i in range(n)])
    assert correct_sum == np.sum(np.dot(P, alpha_mat)) - np.sum(alpha)
    # END OF REMOVE
    minimize_fn = 1 / 2 * np.sum(np.dot(P, alpha_mat)) - np.sum(alpha)
    return minimize_fn

def zerofun(alpha):
    # TODO does this receive alpha or t vector?
    return np.sum(np.dot(alpha, ts))

def compute_b(s, target_s, alpha):
    ''' computes threshold value b
        s = support vector
        x =
        target_s = target class of support vector (+/-1)
        alpha =
        t = target classes of x?
    '''
    global ts
    alpha = np.array(alpha)
    alpha_ts = alpha * ts # compute a_i * t_i vector [a_1 * t_1, a_2 * t_2, ...]
    return np.sum(np.dot(alpha_ts, kernel(s, xs).T)) - target_s

def extract_non_zero_alphas(alpha, x):
    ''' extract alpha values which are non-zero and returns the corresponding alphas, data points and target values '''
    non_zero_indices = [i for i, a in enumerate(alpha) if a > 10**-5]
    a_s = alpha[non_zero_indices]
    x_s = x[non_zero_indices]
    t_s = ts[non_zero_indices]
    return a_s, x_s, t_s

def indicator_function(s, target_s, alpha):
    ''' computes sum over non-zero values
        s = new datapoint to be classified
    '''
    alpha_ts = non_zero_as * non_zero_as
    return np.sum(np.dot(alpha_ts, kernel(s, non_zero_xs).T)) - b

def svm_classifier():
    global non_zero_as, non_zero_xs, non_zero_ts, b
    # TODO
    # b = compute_b(s, target_s, alpha)
    # non_zero_as, non_zero_xs, non_zero_ts = extract_non_zero_alphas(alpha, xs)
    generate_data()
    C = 0.1 # TODO change value. Lower = rely more on slack = better for noisier datasets
    ret = minimize(objective,  np.zeros(N), bounds=[(0, C) for x in xs], constraints=zerofun)
    alpha = ret['x']

if __name__ == '__main__':
    generate_data()