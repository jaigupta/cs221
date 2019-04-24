#!/usr/bin/python

import random
import collections
import math
import sys
from util import *

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    if x is None or len(x) == 0:
        return {}
    res = {}
    for w in x.split():
        res[w] = res.get(w, 0) + 1
    return res
    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    def predictor(x):
        if dotProduct(weights, featureExtractor(x)) < 0:
            return -1
        return 1

    for _ in range(numIters):
        for x, y in trainExamples:
            features = featureExtractor(x)
            loss = 1-dotProduct(weights, features)*y
            if loss <= 0:
                continue
            # w = w - eta* gradient = w - eta*(-y*features) = w + (eta*y)features
            increment(weights, eta*y, features)
    # END_YOUR_CODE
    return weights

############################################################
# Problem 3c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        words = random.sample(weights.keys(), random.randint(1, len(weights)))
        phi = {w:random.randint(1, 10) for w in words}
        score = dotProduct(weights, phi)
        if score == 0:
            print "Found 0 score ", weights
            return generateExample()
        y = (-1 if  score < 0 else 1)
        # END_YOUR_CODE
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 3e: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        x = ''.join(x.split())
        res = {}
        for i in range(len(x)-n + 1):
            w = x[i:i+n]
            res[w] = res.get(w, 0) + 1
        return res
        # END_YOUR_CODE
    return extract

############################################################
# Problem 4: k-means
############################################################


def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 32 lines of code, but don't worry if you deviate from this)
    def getMean(examples):
        mean = {}
        for example in examples:
            for feature in example.items():
                mean[feature[0]] = mean.get(feature[0], 0.0) + feature[1]
        num_examples = len(examples)
        for k in mean.keys():
            mean[k] /= float(num_examples)
        return mean

    def distSq(x, y, x_normsq, y_normsq):
        # ||x-y||^2 = ||x||^2 + ||y||^2 -2(x.y) --> for speedup
        return x_normsq + y_normsq - 2*dotProduct(x, y)

    def getNormSq(x):
        return sum([v*v for v in x.values()])

    examples_normsq = [getNormSq(e) for e in examples]
    means = random.sample(examples, K)
    means_normsq = [getNormSq(u) for u in means]
    assignments = []
    for _ in range(maxIters):
        new_assignments = []
        for ei, example in enumerate(examples):
            min_idx = 0
            min_distsq = distSq(means[0], example, means_normsq[0], examples_normsq[ei])
            for j in range(1, K):
                d = distSq(means[j], example, means_normsq[j], examples_normsq[ei])
                if d < min_distsq:
                    min_distsq = d
                    min_idx = j
            new_assignments.append(min_idx)
        if new_assignments == assignments:
            break
        assignments = new_assignments

        cluster_to_examples = {}
        for i, example in enumerate(examples):
            cluster_to_examples[assignments[i]] = cluster_to_examples.get(assignments[i], []) + [example]
        means = [getMean(cluster_to_examples.get(i, [])) for i in  range(K)]
        means_normsq = [getNormSq(u) for u in means]
    return (means, assignments, sum([distSq(example, means[assignments[i]], examples_normsq[i], means_normsq[assignments[i]]) for i, example in enumerate(examples)]))
    
    # END_YOUR_CODE
