#!/usr/bin/python

import random
from typing import Callable, Dict, List, Tuple, TypeVar

import math
from util import *
from collections import defaultdict

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    feature_vector: FeatureVector = defaultdict(int)
    for word in x.split():
        feature_vector[word] += 1
    return dict(feature_vector)


############################################################
# Problem 3b: stochastic gradient descent

T = TypeVar('T')


def learnPredictor(trainExamples: List[Tuple[T, int]],
                   validationExamples: List[Tuple[T, int]],
                   featureExtractor: Callable[[T], FeatureVector],
                   numEpochs: int, eta: float) -> WeightVector:
    '''
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Notes:
    - Only use the trainExamples for training!
    - You should call evaluatePredictor() on both trainExamples and
      validationExamples to see how you're doing as you learn after each epoch.
    - The predictor should output +1 if the score is precisely 0.
    '''
    weights: WeightVector = {}  # feature => weight

    def predict(x: T) -> int:
        features = featureExtractor(x)
        score = sum(weights.get(f, 0) * v for f, v in features.items())
        return 1 if score >= 0 else -1

    def evaluatePredictor(examples: List[Tuple[T, int]]) -> float:
        return sum(1 for x, y in examples if predict(x) == y) / len(examples)

    for epoch in range(numEpochs):
        random.shuffle(trainExamples)
        for x, y in trainExamples:
            features = featureExtractor(x)
            score = sum(weights.get(f, 0) * v for f, v in features.items())
            if y * score < 1:  # Hinge loss
                for f, v in features.items():
                    weights[f] = weights.get(f, 0) + eta * y * v

        trainAccuracy = evaluatePredictor(trainExamples)
        validationAccuracy = evaluatePredictor(validationExamples)
        #print(
        #    f"Epoch {epoch + 1}: train accuracy = {trainAccuracy:.4f}, validation accuracy = {validationAccuracy:.4f}")

    return weights


############################################################
# Problem 3c: generate test case


def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    '''
    Return a set of examples (phi(x), y) randomly which are classified
      correctly by |weights|.
    '''
    random.seed(42)

    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a score for the given weight vector.
    # note that there is intentionally flexibility in how you define phi.
    # y should be 1 or -1 as classified by the weight vector.
    # IMPORTANT: In the case that the score is 0, y should be set to 1.

    # Note that the weight vector can be arbitrary during testing.
    def generateExample() -> Example:
        phi = {}
        for feature in weights.keys():
            if random.random() < 0.5:  # Randomly include or exclude features
                phi[feature] = random.randint(-10, 10)  # Random feature value

        score = sum(weights.get(f, 0) * v for f, v in phi.items())
        y = 1 if score >= 0 else -1  # IMPORTANT: y is 1 when score is 0

        return phi, y

    return [generateExample() for _ in range(numExamples)]


############################################################
# Problem 3d: character features


def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that 1 <= n <= len(x).
    '''
    def extract(x: str) -> FeatureVector:
        x = ''.join(x.split())  # Remove all spaces
        features: FeatureVector = defaultdict(int)
        for i in range(len(x) - n + 1):
            ngram = x[i:i+n]
            features[ngram] += 1
        return dict(features)

    return extract


############################################################
# Problem 3e:


def testValuesOfN(n: int):
    '''
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in sentiment.pdf.
    '''
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples,
                             validationExamples,
                             featureExtractor,
                             numEpochs=20,
                             eta=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights,
                        'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(
        trainExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(
        validationExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" %
           (trainError, validationError)))


############################################################
# Problem 5: k-means
############################################################


def kmeans(examples: List[Dict[str, float]], K: int,
           maxEpochs: int) -> Tuple[List, List, float]:
    '''
    Perform K-means clustering on |examples|, where each example is a sparse feature vector.
    '''

    def squared_norm(v: Dict[str, float]) -> float:
        return sum(val ** 2 for val in v.values())

    # Precompute squared norms of examples
    example_norms = [squared_norm(ex) for ex in examples]

    # Initialize centroids randomly
    centroids = random.sample(examples, K)
    centroid_norms = [squared_norm(c) for c in centroids]

    for epoch in range(maxEpochs):
        new_assignments = []
        new_centroids = [defaultdict(float) for _ in range(K)]
        cluster_sizes = [0] * K

        for i, example in enumerate(examples):
            best_distance = float('inf')
            best_centroid = 0
            for j, centroid in enumerate(centroids):
                # Use ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
                distance = example_norms[i] + centroid_norms[j] - 2 * dotProduct(example, centroid)
                if distance < best_distance:
                    best_distance = distance
                    best_centroid = j

            new_assignments.append(best_centroid)
            cluster_sizes[best_centroid] += 1
            for k, v in example.items():
                new_centroids[best_centroid][k] += v

        # Compute new centroids
        for j in range(K):
            if cluster_sizes[j] > 0:
                new_centroids[j] = {k: v / cluster_sizes[j] for k, v in new_centroids[j].items()}
            else:
                new_centroids[j] = random.choice(examples)  # Reinitialize empty clusters

        # Update centroid norms
        centroid_norms = [squared_norm(c) for c in new_centroids]

        # Check for convergence
        if all(sparse_dot(c1, c1) + sparse_dot(c2, c2) - 2 * sparse_dot(c1, c2) < 1e-10
               for c1, c2 in zip(centroids, new_centroids)):
            break

        centroids = new_centroids
        assignments = new_assignments

    # Compute final loss
    loss = sum(example_norms[i] + centroid_norms[assignments[i]] -
               2 * sparse_dot(example, centroids[assignments[i]])
               for i, example in enumerate(examples))

    return centroids, assignments, loss
