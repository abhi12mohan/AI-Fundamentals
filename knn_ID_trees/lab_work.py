# MIT 6.034 Lab 6: k-Nearest Neighbors and Identification Trees
# Written by 6.034 Staff

from api import *
from data import *
import math
log2 = lambda x: math.log(x, 2)
INF = float('inf')
from collections import defaultdict

################################################################################
############################# IDENTIFICATION TREES #############################
################################################################################


#### Part 1A: Classifying points ###############################################

def id_tree_classify_point(point, id_tree):
    """Uses the input ID tree (an IdentificationTreeNode) to classify the point.
    Returns the point's classification."""
    while not id_tree.is_leaf():
        id_tree = id_tree.apply_classifier(point)
    return id_tree.get_node_classification()


#### Part 1B: Splitting data with a classifier #################################

def split_on_classifier(data, classifier):
    """Given a set of data (as a list of points) and a Classifier object, uses
    the classifier to partition the data.  Returns a dict mapping each feature
    values to a list of points that have that value."""
    feature_dict = defaultdict(list)
    for point in data:
        feature_dict[classifier.classify(point)].append(point)
    return feature_dict


#### Part 1C: Calculating disorder #############################################

def branch_disorder(data, target_classifier):
    """Given a list of points representing a single branch and a Classifier
    for determining the true classification of each point, computes and returns
    the disorder of the branch."""
    disorder = 0
    split = split_on_classifier(data, target_classifier)

    for vals in split:
        val1 = -len(split[vals])/len(data)
        val2 = log2(len(split[vals])/len(data))
        disorder += (val1*val2)
    return disorder

def average_test_disorder(data, test_classifier, target_classifier):
    """Given a list of points, a feature-test Classifier, and a Classifier
    for determining the true classification of each point, computes and returns
    the disorder of the feature-test stump."""
    disorder = 0
    split = split_on_classifier(data, test_classifier)

    for vals in split:
        val1 = len(split[vals])/len(data)
        val2 = branch_disorder(split[vals], target_classifier)
        disorder += (val1*val2)
    return disorder



#### Part 1D: Constructing an ID tree ##########################################

def find_best_classifier(data, possible_classifiers, target_classifier):
    """Given a list of points, a list of possible Classifiers to use as tests,
    and a Classifier for determining the true classification of each point,
    finds and returns the classifier with the lowest disorder.  Breaks ties by
    preferring classifiers that appear earlier in the list.  If the best
    classifier has only one branch, raises NoGoodClassifiersError."""
    disorder = float("inf")
    classifier = None

    for curr_class in possible_classifiers:
        curr_disorder = average_test_disorder(data, curr_class, target_classifier)
        if curr_disorder < disorder:
            disorder, classifier = curr_disorder, curr_class

    if len(split_on_classifier(data, classifier)) == 1:
        raise NoGoodClassifiersError()
    else:
        return classifier


## To find the best classifier from 2014 Q2, Part A, uncomment:
# print(find_best_classifier(tree_data, tree_classifiers, feature_test("tree_type")))

def construct_greedy_id_tree(data, possible_classifiers, target_classifier, id_tree_node=None):
    """Given a list of points, a list of possible Classifiers to use as tests,
    a Classifier for determining the true classification of each point, and
    optionally a partially completed ID tree, returns a completed ID tree by
    adding classifiers and classifications until either perfect classification
    has been achieved, or there are no good classifiers left."""
    if id_tree_node is None:
        id_tree_node = IdentificationTreeNode(target_classifier)
    if branch_disorder(data, target_classifier) == 0:
        id_tree_node.set_node_classification(target_classifier.classify(data[0]))
        return id_tree_node

    try:
        best_classifier = find_best_classifier(data, possible_classifiers, target_classifier)
        split = split_on_classifier(data, best_classifier)
        id_tree_node = id_tree_node.set_classifier_and_expand(best_classifier, split)

        possible_classifiers.remove(best_classifier)
        branches = id_tree_node.get_branches()

        for branch in branches:
            new_greedy_tree = construct_greedy_id_tree(split[branch], possible_classifiers, target_classifier, branches[branch])
            branches[branch] = new_greedy_tree

    except NoGoodClassifiersError:
        id_tree_node.set_node_classification(None)

    return id_tree_node


#### Part 2B: Distance metrics #################################################

def dot_product(u, v):
    """Computes dot product of two vectors u and v, each represented as a tuple
    or list of coordinates.  Assume the two vectors are the same length."""
    return sum([u[i]*v[i] for i in range(len(u))])

def norm(v):
    "Computes length of a vector v, represented as a tuple or list of coords."
    return dot_product(v,v) ** 0.5

def euclidean_distance(point1, point2):
    "Given two Points, computes and returns the Euclidean distance between them."
    return sum([(point2.coords[i]-point1.coords[i])**2 for i in range(len(point1.coords))]) ** 0.5

def manhattan_distance(point1, point2):
    "Given two Points, computes and returns the Manhattan distance between them."
    return sum([abs(point2.coords[i]-point1.coords[i]) for i in range(len(point1.coords))])

def hamming_distance(point1, point2):
    "Given two Points, computes and returns the Hamming distance between them."
    return sum([point2.coords[i] != point1.coords[i] for i in range(len(point1.coords))])

def cosine_distance(point1, point2):
    """Given two Points, computes and returns the cosine distance between them,
    where cosine distance is defined as 1-cos(angle_between(point1, point2))."""
    numerator = dot_product(point1.coords, point2.coords)
    denom = norm(point1.coords) * norm(point2.coords)
    return 1 - (numerator/denom)


#### Part 2C: Classifying points ###############################################

def get_k_closest_points(point, data, k, distance_metric):
    """Given a test point, a list of points (the data), an int 0 < k <= len(data),
    and a distance metric (a function), returns a list containing the k points
    from the data that are closest to the test point, according to the distance
    metric.  Breaks ties lexicographically by coordinates."""
    distance_pts = [(distance_metric(point, pt), pt) for pt in data]
    distance_pts.sort(key = lambda pt: (pt[0], pt[1].coords))
    return [pt[1] for pt in distance_pts][:k]

def knn_classify_point(point, data, k, distance_metric):
    """Given a test point, a list of points (the data), an int 0 < k <= len(data),
    and a distance metric (a function), returns the classification of the test
    point based on its k nearest neighbors, as determined by the distance metric.
    Assumes there are no ties."""
    closest_pts = get_k_closest_points(point, data, k, distance_metric)
    classifications = [pt.classification for pt in closest_pts]
    return max(classifications, key = classifications.count)


## To run your classify function on the k-nearest neighbors problem from 2014 Q2
## part B2, uncomment the line below and try different values of k:
# print(knn_classify_point(knn_tree_test_point, knn_tree_data, 1, euclidean_distance))


#### Part 2C: Choosing k #######################################################

def cross_validate(data, k, distance_metric):
    """Given a list of points (the data), an int 0 < k <= len(data), and a
    distance metric (a function), performs leave-one-out cross-validation.
    Return the fraction of points classified correctly, as a float."""
    point_count = 0
    for i in range(len(data)):
        left_out = data[i]
        diff_data = data[:i] + data[i+1:]

        if left_out.classification == knn_classify_point(left_out, diff_data, k, distance_metric):
            point_count += 1
    return point_count / len(data)

def find_best_k_and_metric(data):
    """Given a list of points (the data), uses leave-one-out cross-validation to
    determine the best value of k and distance_metric, choosing from among the
    four distance metrics defined above.  Returns a tuple (k, distance_metric),
    where k is an int and distance_metric is a function."""
    best_found = tuple()
    best_val = 0

    for metric in [euclidean_distance, manhattan_distance, hamming_distance, cosine_distance]:
        for k_val in range(1, len(data)-1):
            cross_val = cross_validate(data, k_val, metric)
            if cross_val > best_val:
                best_found, best_val = (k_val, metric), cross_val
    return best_found
