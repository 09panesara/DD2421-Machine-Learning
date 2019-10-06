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

kernel_type = 'linear'
kernel_p = 2
kernel_sigma = 0.01
slack = None

def compute_objective_matx(t, x):
    ''' Precomputes P
        P[i,j] = t[i][t[j] K(x[i]x[j])
    '''
    global P, K
    n = len(t)
    K = np.asarray([[kernel(x[i], x[j]) for j in range(n)] for i in range(n)])
    P = np.asarray([[t[i] * t[j] * K[i, j] for j in range(n)] for i in range(n)])

def generate_data(cluster_size, cluster_centers):
    global xs, ts, classA, classB, N
    np.random.seed(100) # generate same data each time
    classA_size = int(cluster_size/2)
    classA = np.concatenate((np.random.randn(classA_size, 2) * 0.2 + cluster_centers[0], np.random.randn(classA_size, 2) * 0.2 + cluster_centers[1]))
    classB = np.random.randn(cluster_size, 2) * 0.2 + cluster_centers[2]
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
    compute_objective_matx(ts, xs)

def plot_data(classA, classB, cluster_center_i, cluster_size):
    plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
    plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
    plt.axis('equal')  # Force same scale on both axes plt.savefig(’svmplot.pdf’) # Save a copy in a file plt .show() # Show the plot on the screen

    # plot decision boundary
    xgrid = np.linspace(-5, 5)
    ygrid = np.linspace(-4, 4)
    grid = np.array([[indicator_function([x,y]) for x in xgrid] for y in ygrid])
    plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
    title = f'SVM Classifer with {kernel_type} kernel'
    plot_name = f'kernel={kernel_type}'
    if kernel_type == 'rbf':
        plot_name += f'_sigma={kernel_sigma}'
        title += f' with sigma = {kernel_sigma}'
    elif kernel_type == 'polynomial':
        title += f' with p = {kernel_p}'
        plot_name += f'_p={kernel_p}'

    if slack is not None:
        title += f' with slack {slack}'
        plot_name += f'_slack={slack}'
    plot_name += f'_cluster-center={cluster_center_i}_cluster-size={cluster_size}.png'
    plt.title(title)
    print('Saving plot to', plot_name)
    plt.savefig('plots/' + plot_name)
    plt.clf()
    # plt.show()

def _euclidean_squared(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.sum((x-y)**2)

def kernel(a, b):
    global kernel_type, kernel_p, kernel_sigma
    if kernel_type == 'linear':
        #print(np.dot(x, y))
        return np.dot(a, b)
    elif kernel_type == 'polynomial':
        return (np.dot(a, b) + 1)**kernel_p
    elif kernel_type == 'rbf': # radial basis function
        return math.e**(-_euclidean_squared(a, b)/(2*kernel_sigma**2))
    else:
        raise Exception(f'Cannot recognise kernel type {kernel_type}')



def objective(alphas):
    # DOES AS EXPECTED GIVEN PARAMETERS
    ''' Computes objective function using kernel trick (see eqn 4 in lab notes)
    '''
    n = len(alphas)
    part_1 = 0.5 * np.sum([[alphas[i] * alphas[j] * P[i,j] for j in range(n)] for i in range(n)])
    part_2 = np.sum(alphas)

    # target_sum = np.sum(np.dot(P, alpha_mat)) - np.sum(alpha)
    # if correct_sum != target_sum and abs(correct_sum) > 10**-6 and abs(target_sum) > 10**-6:
    #     print('correct sum', correct_sum)
    #     print('target sum', target_sum)
    #     assert correct_sum == np.sum(np.dot(P, alpha_mat)) - np.sum(alpha)
    # minimize_fn = 1 / 2 * np.sum(np.dot(P, alpha_mat)) - np.sum(alpha)
    minimize_fn = part_1 - part_2
    return minimize_fn

def zerofun(alphas):
    # DOES AS EXPECTED
    return np.sum(np.dot(alphas, ts))

def compute_b(alpha, t, x):
    # DOES AS EXPECTED GIVEN PARAMETERS
    ''' computes threshold value b
        s = support vector
        x = set of points
        target_s = target class of support vector (+/-1)
        alpha =
        t = target classes of x?
    '''

    s = x[0]
    t_s = t[0]
    alpha_ts = np.array(alpha) * t # compute a_i * t_i vector [a_1 * t_1, a_2 * t_2, ...]

    kernel_vals = [kernel(s, xi) for xi in x]
    b = np.sum(alpha_ts * kernel_vals) - t_s
    return b

def extract_non_zero_alphas(alphas):
    # DOES AS EXPECTED
    ''' extract alpha values which are non-zero and returns the corresponding alphas, data points and target values '''
    non_zero_indices = [i for i, a in enumerate(alphas) if a > 10**-5]
    a_s = alphas[non_zero_indices]
    x_s = xs[non_zero_indices]
    t_s = ts[non_zero_indices]
    return a_s, x_s, t_s

def indicator_function(s):
    # DOES AS EXPECTED given parameter
    ''' computes sum over non-zero values
        s = new datapoint to be classified
    '''
    alpha_ts = non_zero_as * non_zero_ts
    kernel_vals = np.array([kernel(s, x) for x in non_zero_xs])
    target = np.sum(np.dot(kernel_vals, alpha_ts)) - b
    return target

def svm_classifier(cluster_size, cluster_centers, cluster_centers_i, C=None, plot_name='svmplot-test.pdf', plot=True):
    global non_zero_as, non_zero_xs, non_zero_ts, b
    # TODO
    generate_data(cluster_size, cluster_centers)
    constraint = {'type': 'eq', 'fun': zerofun}
    ret = minimize(objective,  np.zeros(N), bounds=[(0, C) for x in xs], constraints=constraint)
    if ret['success']: # we minimised the objective function
        alphas = ret['x'] # get alpha values
        non_zero_as, non_zero_xs, non_zero_ts = extract_non_zero_alphas(alphas)
        # set global variable b to be used in indicator function
        if len(non_zero_xs) > 0:
            b = compute_b(non_zero_as, non_zero_ts, non_zero_xs)
            if plot:
                plot_data(classA, classB, cluster_centers_i, cluster_size)
        else:
            print('no non-zero alphas found for ', plot_name)
    else:
        print('Unable to separate data for ', plot_name)

if __name__ == '__main__':
    # kernels = ['linear', 'polynomial', 'rbf']
    kernels = ['linear', 'polynomial']
    kernel_parameters = {'polynomial': [2, 3, 4, 5], 'rbf': [0.1, 0.15]}
    cluster_centers = [[[-1.5, 0.5], [1.5, 0.5], [0.0, -0.5]], [[1.0, 1.0], [4.0, 2.0], [2.0, -2.0]], [[0.0, 3.5], [0.5, -1.0], [0.5, 1.5]]]
    cluster_sizes = [20, 30, 40]

    # kernel_type, kernel_p, kernel_sigma

    for i, cluster_size in enumerate(cluster_sizes):
        for j, cluster_center in enumerate(cluster_centers):
            for used_kernel in kernels:
                kernel_type = used_kernel
                plot_name = used_kernel + '_cluster-size-' + str(i) + '_cluster-center-' + str(j)
                if 'linear' == used_kernel:
                    svm_classifier(cluster_size, cluster_center, j)
                else:  # non-linear kernels, therefore we need to take the parameters into account as well
                    params = kernel_parameters.get(used_kernel)
                    for parameter in params:
                        if used_kernel == 'rbf':
                            kernel_sigma = parameter
                        else:
                            kernel_p = parameter
                        svm_classifier(cluster_size, cluster_center, j)
