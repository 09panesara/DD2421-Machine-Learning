import monkdata as m
from dtree import entropy, averageGain, buildTree, check, allPruned
from drawtree_qt5 import drawTree
import random
import numpy as np
import matplotlib.pyplot as plt


monk1 = None
monk2 = None
monk3 = None
monk1test = None
monk2test = None
monk3test = None

def calc_entropy():
    entropy1 = entropy(monk1)
    entropy2 = entropy(monk2)
    entropy3 = entropy(monk3)
    print('entropy of monk1', entropy1)
    print('entropy of monk2', entropy2)
    print('entropy of monk3', entropy3)

def calc_info_gain():
    datasets = {'monk1': monk1,'monk2':  monk2, 'monk3': monk3}
    for dataset in datasets:
        print('Dataset ', dataset)
        monk_dataset_i = datasets[dataset]
        for i in range(6):
            print(f'Information gain for attribute {i}: {round(averageGain(monk_dataset_i, m.attributes[i]), 5)}')

def decision_tree_id3():
    # predefined decision tree function
    datasets = {'monk1': [monk1, monk1test], 'monk2': [monk2, monk2test], 'monk3': [monk3, monk3test]}

    for dataset in datasets:
        if dataset != 'monk2':
            continue
        print(dataset)
        train_data = datasets[dataset][0]
        test_data = datasets[dataset][1]
        t = buildTree(train_data, m.attributes)
        print(drawTree(t))
        print('Error in train', 1-check(t, train_data))
        print('Error in test', (1-check(t, test_data)))



# def decision_tree():
#     # our decision tree function
#     datasets = {'monk1': monk1, 'monk2': monk2, 'monk3': monk3}
#     for dataset in datasets:

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]




def eval_pruning():
    fraction_params = np.arange(3, 9, 1) / 10
    no_runs = 100
    def pruning(dataset, test):
        print('fraction params', fraction_params)
        test_errors = np.zeros((6, no_runs))
        for i, f in enumerate(fraction_params):
            for j in range(no_runs):
                monk_train, monk_val = partition(dataset, f)
                is_smaller_err = True
                tree = buildTree(monk_train, m.attributes)
                while is_smaller_err:
                    curr_err = 1 - check(tree, monk_val)
                    pruned_trees = allPruned(tree)
                    pruned_errors = np.array([1 - check(pruned_tree, monk_val) for pruned_tree in pruned_trees])
                    is_smaller_err = len(np.where(pruned_errors < curr_err)[0]) > 0
                    if not is_smaller_err or len(pruned_trees) == 0:
                        break
                    tree = pruned_trees[np.argmin(pruned_errors)]
                test_errors[i, j] = (1 - check(tree, test))
        avg_test_error = np.mean(test_errors, 1)
        std_test_error = np.std(test_errors, 1)
        return avg_test_error, std_test_error

    monk1_avg_err, monk1_std = pruning(monk1, monk1test)
    monk3_avg_err, monk3_std = pruning(monk3, monk3test)
    plt.errorbar(fraction_params, monk1_avg_err, yerr=monk1_std)
    plt.errorbar(fraction_params, monk3_avg_err, yerr=monk3_std)
    plt.xlabel('Fraction of data set to use for training')
    plt.ylabel('Error on test data set')
    plt.title(f'Average error after pruning against train-test split over {no_runs} runs')
    plt.legend(['Monk1', 'Monk3'])
    plt.savefig('plots/pruning_fraction_split.png')
    plt.show()



if __name__ == '__main__':
    monk1 = m.monk1
    monk2 = m.monk2
    monk3 = m.monk3
    monk1test = m.monk1test
    # monk2test = m.monk2test
    monk3test = m.monk3test
    # decision_tree_id3()
    eval_pruning()