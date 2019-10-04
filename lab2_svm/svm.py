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
classA, classB = None, None

#TODO find bug for wrong decision boundary
# with simple data points and control calculations made by hand compute_objective_matx(), objective() and kernel() seem to be ok

def compute_objective_matx(t, x):
    ''' Precomputes P
        P[i,j] = t[i][t[j] K(x[i]x[j])
    '''
    global P, K
    n = len(t)
    K = np.asarray([[kernel(x[i], x[j]) for j in range(n)] for i in range(n)])
    P = np.asarray([[t[i] * t[j] * K[i, j] for j in range(n)] for i in range(n)])

def generate_data():
    global xs, ts, classA, classB, N
    # TODO remove comments after debugging is finished
    #np.random.seed(100) # generate same data each time
    #classA = np.concatenate((np.random.randn(10, 2) * 0.2 +[1.5, 0.5], np.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))
    #classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]
    # BEGIN REMOVE
    classA = np.array([(-4, 0)])
    classB = np.array([(4, 0)])
    # END REMOVE
    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))
    N = inputs.shape[0]  # Number of rows (samples)
    # randomly reorder samples
    #TODO remove comments after debugging is finished
    #permute = list(range(N))
    #random.shuffle(permute)
    #inputs = inputs[permute, :]
    #targets = targets[permute]
    xs = inputs
    N = len(xs)
    ts = targets
    compute_objective_matx(ts, xs)

def plot_data(classA, classB):
    plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
    plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
    plt.axis('equal')  # Force same scale on both axes plt.savefig(’svmplot.pdf’) # Save a copy in a file plt .show() # Show the plot on the screen

    # plot decision boundary
    xgrid = np.linspace(-5, 5)
    ygrid = np.linspace(-4, 4)

    grid = np.array([[indicator_function(np.array(x, y)) for x in xgrid] for y in ygrid])
    plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

    plt.savefig('plots/svmplot-test.pdf')
    plt.show()

def _euclidean_squared(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.sum((x-y)**2)

def kernel(x, y, kernel_type='linear', p=2, sigma=0.01):
    #print(x, y)
    if kernel_type == 'linear':
        #print(np.dot(x, y))
        return np.dot(x, y)
    elif kernel_type == 'polynomial':
        return (np.dot(x, y) + 1)**p
    elif kernel_type == 'rbf': # radial basis function
        return math.e**(-_euclidean_squared(x, y)/(2*sigma**2))
    else:
        raise Exception(f'Cannot recognise kernel type {kernel_type}')

def objective(alpha):
    ''' Computes objective function using kernel trick (see eqn 4 in lab notes)
    '''
    n = len(alpha)
    #alpha_mat = np.array([a_i*alpha for a_i in alpha])
    # TODO remove for testing purposes only
    #print(n, alpha, P)
    part_1 = np.sum([[alpha[i] * alpha[j] * P[i,j] for j in range(n)] for i in range(n)])
    part_2 = np.sum(alpha)
    correct_sum = part_1 - part_2

    # target_sum = np.sum(np.dot(P, alpha_mat)) - np.sum(alpha)
    # if correct_sum != target_sum and abs(correct_sum) > 10**-6 and abs(target_sum) > 10**-6:
    #     print('correct sum', correct_sum)
    #     print('target sum', target_sum)
    #     assert correct_sum == np.sum(np.dot(P, alpha_mat)) - np.sum(alpha)
    # # END OF REMOVE
    # minimize_fn = 1 / 2 * np.sum(np.dot(P, alpha_mat)) - np.sum(alpha)
    minimize_fn = 0.5 * correct_sum
    return minimize_fn

def zerofun(alpha):
    return np.sum(np.dot(alpha, ts))

def compute_b(alpha, t, x):
    ''' computes threshold value b
        s = support vector
        x =
        target_s = target class of support vector (+/-1)
        alpha =
        t = target classes of x?
    '''
    s = x[0]
    target_s = ts[0]
    alpha = np.array(alpha)
    alpha_ts = alpha * t # compute a_i * t_i vector [a_1 * t_1, a_2 * t_2, ...]
    return np.sum([np.dot(alpha_ts, kernel(s, xi)) for xi in x]) - target_s

def extract_non_zero_alphas(alpha, x):
    ''' extract alpha values which are non-zero and returns the corresponding alphas, data points and target values '''
    non_zero_indices = [i for i, a in enumerate(alpha) if a > 10**-5]
    a_s = alpha[non_zero_indices]
    x_s = x[non_zero_indices]
    t_s = ts[non_zero_indices]
    return a_s, x_s, t_s

def indicator_function(s):
    ''' computes sum over non-zero values
        s = new datapoint to be classified
    '''
    alpha_ts = non_zero_as * non_zero_as
    return np.sum(np.dot(alpha_ts, kernel(s, non_zero_xs))) - b

def svm_classifier(plot=False):
    global non_zero_as, non_zero_xs, non_zero_ts, b
    # TODO
    generate_data()
    C = 0.1 # TODO change value. Lower = rely more on slack = better for noisier datasets
    constraint = {'type': 'eq', 'fun': zerofun}
    ret = minimize(objective,  np.zeros(N), bounds=[(0, None) for x in xs], constraints=constraint)
    if ret['success']:
        alpha = ret['x']
        non_zero_as, non_zero_xs, non_zero_ts = extract_non_zero_alphas(alpha, xs)
        b = compute_b(non_zero_as, non_zero_ts, non_zero_xs)
        print('b', b)
        print('found alpha', alpha)

        if plot:
            plot_data(classA, classB)
    else:
        print('Unable to separate data')

if __name__ == '__main__':
    svm_classifier(True)