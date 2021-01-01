# MIT 6.034 Lab 9: Boosting (Adaboost)
# Written by 6.034 staff

from math import log as ln
from utils import *


#### Part 1: Helper functions ##################################################

def initialize_weights(training_points):
    """Assigns every training point a weight equal to 1/N, where N is the number
    of training points.  Returns a dictionary mapping points to weights."""
    num_points = len(training_points)
    return {pt:make_fraction(1, num_points) for pt in training_points}

def calculate_error_rates(point_to_weight, classifier_to_misclassified):
    """Given a dictionary mapping training points to their weights, and another
    dictionary mapping classifiers to the training points they misclassify,
    returns a dictionary mapping classifiers to their error rates."""
    return {c: sum([point_to_weight[pt] for pt in classifier_to_misclassified[c]]) for c in classifier_to_misclassified}

def pick_best_classifier(classifier_to_error_rate, use_smallest_error=True):
    """Given a dictionary mapping classifiers to their error rates, returns the
    best* classifier, or raises NoGoodClassifiersError if best* classifier has
    error rate 1/2.  best* means 'smallest error rate' if use_smallest_error
    is True, otherwise 'error rate furthest from 1/2'."""
    if use_smallest_error:
        classifiers = sorted([(c, make_fraction(classifier_to_error_rate[c])) for c in classifier_to_error_rate], key = lambda c: (c[1], c[0]))
    else:
        classifiers = sorted([(c, abs(make_fraction(1,2) - classifier_to_error_rate[c])) for c in classifier_to_error_rate], key = lambda c: c[0])
        classifiers.sort(key = lambda c: c[1], reverse = True)

    best_c = classifiers[0][0]

    if approx_equal(classifier_to_error_rate[best_c], make_fraction(1, 2)):
        raise NoGoodClassifiersError
    else:
        return best_c

def calculate_voting_power(error_rate):
    """Given a classifier's error rate (a number), returns the voting power
    (aka alpha, or coefficient) for that classifier."""
    if error_rate == 0:
        return float("INF")
    elif error_rate == 1:
        return float("-INF")
    else:
        return make_fraction(1, 2) * ln(make_fraction(1 - error_rate, error_rate))

def get_overall_misclassifications(H, training_points, classifier_to_misclassified):
    """Given an overall classifier H, a list of all training points, and a
    dictionary mapping classifiers to the training points they misclassify,
    returns a set containing the training points that H misclassifies.
    H is represented as a list of (classifier, voting_power) tuples."""
    misclassifiers = set()

    for pt in training_points:
        correct = sum([vp for c, vp in H if pt not in classifier_to_misclassified[c]])
        incorrect = sum([vp for c, vp in H if pt in classifier_to_misclassified[c]])

        if incorrect >= correct:
            misclassifiers.add(pt)

    return misclassifiers

def is_good_enough(H, training_points, classifier_to_misclassified, mistake_tolerance=0):
    """Given an overall classifier H, a list of all training points, a
    dictionary mapping classifiers to the training points they misclassify, and
    a mistake tolerance (the maximum number of allowed misclassifications),
    returns False if H misclassifies more points than the tolerance allows,
    otherwise True.  H is represented as a list of (classifier, voting_power)
    tuples."""
    return len(get_overall_misclassifications(H, training_points, classifier_to_misclassified)) <= mistake_tolerance

def update_weights(point_to_weight, misclassified_points, error_rate):
    """Given a dictionary mapping training points to their old weights, a list
    of training points misclassified by the current weak classifier, and the
    error rate of the current weak classifier, returns a dictionary mapping
    training points to their new weights.  This function is allowed (but not
    required) to modify the input dictionary point_to_weight."""
    for pt in point_to_weight:
        if pt in misclassified_points:
            point_to_weight[pt] *= make_fraction(1, 2) * make_fraction(1, error_rate)
        else:
            point_to_weight[pt] *= make_fraction(1, 2) * make_fraction(1, 1-error_rate)

    return point_to_weight


#### Part 2: Adaboost ##########################################################

def adaboost(training_points, classifier_to_misclassified,
             use_smallest_error=True, mistake_tolerance=0, max_rounds=INF):
    """Performs the Adaboost algorithm for up to max_rounds rounds.
    Returns the resulting overall classifier H, represented as a list of
    (classifier, voting_power) tuples."""

    weights = initialize_weights(training_points)
    H = []

    while max_rounds > 0:
        if is_good_enough(H, training_points, classifier_to_misclassified, mistake_tolerance):
            return H

        classifier_to_error = calculate_error_rates(weights, classifier_to_misclassified)

        try:
            best_c = pick_best_classifier(classifier_to_error, use_smallest_error)
            error_rate = classifier_to_error[best_c]

            vp = calculate_voting_power(error_rate)
            H.append((best_c, vp))

            misc_pts = classifier_to_misclassified[best_c]
            weights = update_weights(weights, misc_pts, error_rate)
            max_rounds -= 1

        except NoGoodClassifiersError:
            return H

    return H
