import numpy as np


def loadDataset(filename):
    fr = open(filename)
    dataMat = []
    for line in fr.readlines():
        listOfLine = line.strip().split('\t')
        dataArray = [float(element) for element in listOfLine]
        dataMat.append(dataArray)
    dataMat = np.array(dataMat)
    return dataMat


def biSplitDataset(dataset, featureIndex, value):
    """
    split the dataset on a given feature based on the value
    into two subsets
    :param dataset: input data
    :param featureIndex: the index of feature/attribute to split
    :param value: the value of this feature
    :return: two subsets after splitting
    """
    mat0 = dataset[dataset[:, featureIndex] <= value]
    mat1 = dataset[dataset[:, featureIndex] > value]
    return mat0, mat1


def valueLeaf(dataset):
    """
    the value of the leave node, which is the average y value
    of the data falling into this leaf.
    :param dataset: input data
    :return: mean value of y
    """
    return np.mean(dataset[:, -1])


def valueErr(dataset):
    """
    The error function to qualify the error of dataset after
    splitting. We select the sum-of-square error(SSE) of sets.
    :param dataset:
    :return: error
    """
    n = len(dataset)
    return np.var(dataset[:, -1]) * n


def chooseBestSplit(dataset, thrshold=1, leastNum=4):
    """
    Choose the best value in the best feature for splitting.
    Iterately we traversal each value in each feature, computing
    the error based on this splitting error and find the value and
    the feature based on which the dataset has smallest SSE error.
    :param dataset: input data
    :param thrshold: the threshold of error reduction
    :param leastNum: least number of data in a subset
    :return: the best splitting feature index and best value
    """
    length, n = np.shape(dataset)
    numFeat = n - 1
    if len(np.unique(dataset[:, -1])) == 1:  # if all of entries in the dataset hold the same y value
        return None, valueLeaf(dataset)
    bestErr = np.inf  # initialize the best splitting error
    initErr = valueErr(dataset)  # the error on the whole dataset
    bestFeature = 0
    bestValue = 0
    for feat in range(numFeat):
        for value in dataset[:, feat]:
            mat0, mat1 = biSplitDataset(dataset, feat, value)
            # ignore when the number of data in either subset is too small
            if (len(mat0) < leastNum) or (len(mat1) < leastNum):
                continue
            newErr = valueErr(mat0) + valueErr(mat1)
            if newErr < bestErr:
                bestErr = newErr
                bestFeature = feat
                bestValue = value
    if initErr - bestErr < thrshold:  # ignore if the reduction of error is not significant
        return None, valueLeaf(dataset)
    mat0, mat1 = biSplitDataset(dataset, bestFeature, bestValue)
    if (len(mat0) < leastNum) or (len(mat1) < leastNum):
        return None, valueLeaf(dataset)
    return bestFeature, bestValue


def createTree(dataset, thrshold=1, leastNum=4):
    """
    create the CART tree recursively and store the node information
    into a dictionary structure. Each node has four keywords:
    'spIdx' denotes the split feature index;
    'spValue' denotes the split value;
    and 'left' and 'right' denote both of leaves nodes.
    :return: a complete tree
    """
    feat, valueSplit = chooseBestSplit(dataset, thrshold, leastNum)
    if feat == None:
        return valueSplit
    retTree = {}
    retTree['spIdx'] = feat
    retTree['spValue'] = valueSplit
    leftSet, rightSet = biSplitDataset(dataset, feat, valueSplit)
    retTree['left'] = createTree(leftSet, thrshold, leastNum)
    retTree['right'] = createTree(rightSet, thrshold, leastNum)
    return retTree


def isTree(obj):  #  judge weather the variable is a tree
    if isinstance(obj, dict):
        return True
    else:
        return False


def collapse(tree):
    """
    collapse the tree nodes if there isn't a subset of testset
    input into the tree. Then we combine the left and right leave
    together and replace the value in the node with their average.
    :param tree:
    :return: the final value in the root
    """
    if isTree(tree['left']):
        tree['left'] = collapse(tree['left'])
    if isTree(tree['right']):
        tree['right'] = collapse(tree['right'])
    else:
        return (tree['left'] + tree['right']) / 2


def pruning(tree, testset):
    """
    prune the tree using testdata recursively.
    :param tree: the input tree constructed before
    :param testset: test dataset
    :return: the tree after pruning
    """
    if len(testset) == 0:  # if the subset of testset is null, collapse the nodes below
        return collapse(tree)
    if isTree(tree['left']) or isTree(tree['right']):  # if there exist a subtree in either left or right leaf
        leftSet, rightSet = biSplitDataset(testset, tree['spIdx'], tree['spValue'])
    if isTree(tree['left']):
            tree['left'] = pruning(tree['left'], leftSet)
    if isTree(tree['right']):
            tree['right'] = pruning(tree['right'], rightSet)
    elif not isTree(tree['left']) and not isTree(tree['right']):
        # compute the left/right set in order the further error computing.
        leftSet, rightSet = biSplitDataset(testset, tree['spIdx'], tree['spValue'])
        treeMean = (tree['left'] + tree['right']) / 2
        # error without merge
        errorNoMerge = np.sum((leftSet[:, -1] - tree['left'])**2) + np.sum((rightSet[:, -1] - tree['right'])**2)
        # error with merge
        errorMerge = np.sum((testset[:, -1] - treeMean)**2)
        if errorMerge < errorNoMerge:
            return treeMean
        else:
            return tree
    return tree


def treeForecast(tree, input):
    """
    input a test array and forecast the regression value.
    :param tree: tree model
    :param input: inout test array
    :return: forcasting regression value
    """
    if input[tree['spIdx']] < tree['spValue']:  # if data array belongs to the left leaf node
        if isTree(tree['left']):
            return treeForecast(tree['left'], input)
        else:
            return tree['left']
    else:
        if isTree(tree['right']):
            return treeForecast(tree['right'], input)
        else:
            return tree['right']


def createForecast(tree, testset):
    estimateValue = np.zeros((len(testset), 1))
    for i in range(len(testset)):
        estimateValue[i, 0] = treeForecast(tree, testset[i])
    return estimateValue


def forecastLR(trainset, testset):
   """
   estimate the forecasting values using linear regression
   :param trainset:
   :param testset:
   :return: estimating values
   """
   length, dim = np.shape(trainset)
   #  add a row of "1"s for generalization, return a trraining data with size: dimension * length
   traindata = np.vstack(((np.ones((1, length))), trainset[:, :dim-1].T))
   testdata = np.vstack(((np.ones((1, length))), testset[:, :dim - 1].T))
   beta = np.linalg.inv(traindata.dot(traindata.T)).dot(traindata).dot(trainset[:, -1])
   estimateValue = beta.T.dot(testdata)
   return estimateValue.T


