# MIT 6.034 Lab 8: Support Vector Machines
# Written by 6.034 staff

from svm_data import *
from functools import reduce


#### Part 1: Vector Math #######################################################

def dot_product(u, v):
    """Computes the dot product of two vectors u and v, each represented
    as a tuple or list of coordinates. Assume the two vectors are the
    same length."""
    return sum([u[i]*v[i] for i in range(len(u))])

def norm(v):
    """Computes the norm (length) of a vector v, represented
    as a tuple or list of coords."""
    return dot_product(v, v)**0.5


#### Part 2: Using the SVM Boundary Equations ##################################

def positiveness(svm, point):
    """Computes the expression (w dot x + b) for the given Point x."""
    return dot_product(svm.w,  point.coords) + svm.b

def classify(svm, point):
    """Uses the given SVM to classify a Point. Assume that the point's true
    classification is unknown.
    Returns +1 or -1, or 0 if point is on boundary."""
    classification = positiveness(svm, point)
    return 1 if classification > 0 else (-1 if classification < 0 else 0)

def margin_width(svm):
    """Calculate margin width based on the current boundary."""
    return 2 / norm(svm.w)

def check_gutter_constraint(svm):
    """Returns the set of training points that violate one or both conditions:
        * gutter constraint (positiveness == classification, for support vectors)
        * training points must not be between the gutters
    Assumes that the SVM has support vectors assigned."""
    violations = {pt for pt in svm.support_vectors if positiveness(svm, pt) != pt.classification}
    all_v = violations.union({pt for pt in svm.training_points if abs(positiveness(svm, pt)) < 1})
    return all_v


#### Part 3: Supportiveness ####################################################

def check_alpha_signs(svm):
    """Returns the set of training points that violate either condition:
        * all non-support-vector training points have alpha = 0
        * all support vectors have alpha > 0
    Assumes that the SVM has support vectors assigned, and that all training
    points have alpha values assigned."""
    support_v = set(svm.support_vectors)
    return {pt for pt in svm.training_points if (pt.alpha < 0) or (pt in support_v and pt.alpha <= 0) or (pt not in support_v and pt.alpha != 0)}


def check_alpha_equations(svm):
    """Returns True if both Lagrange-multiplier equations are satisfied,
    otherwise False. Assumes that the SVM has support vectors assigned, and
    that all training points have alpha values assigned."""
    eq5  = [scalar_mult(classify(svm, pt)*pt.alpha, pt.coords) for pt in svm.training_points]
    ans = eq5[0]
    for i in range(1, len(eq5)):
        eq2 = eq5[i]
        ans = vector_add(ans, eq2)

    eq4 = [classify(svm, pt)*pt.alpha for pt in svm.training_points]

    return ans == svm.w and sum(eq4) == 0


#### Part 4: Evaluating Accuracy ###############################################

def misclassified_training_points(svm):
    """Returns the set of training points that are classified incorrectly
    using the current decision boundary."""
    return {pt for pt in svm.training_points if classify(svm, pt) != pt.classification}


#### Part 5: Training an SVM ###################################################

def update_svm_from_alphas(svm):
    """Given an SVM with training data and alpha values, use alpha values to
    update the SVM's support vectors, w, and b. Return the updated SVM."""
    svm.support_vectors = [pt for pt in svm.training_points if pt.alpha > 0]

    eq5  = [scalar_mult(pt.classification*pt.alpha, pt.coords) for pt in svm.training_points]
    ans = eq5[0]
    for i in range(1, len(eq5)):
        eq2 = eq5[i]
        ans = vector_add(ans, eq2)

    min_b = min([pt.classification - dot_product(ans, pt.coords) for pt in svm.support_vectors if pt.classification < 0])
    max_b = max([pt.classification - dot_product(ans, pt.coords) for pt in svm.support_vectors if pt.classification > 0])

    new_boundary = svm.set_boundary(ans, (min_b+max_b)/2)
    return new_boundary
