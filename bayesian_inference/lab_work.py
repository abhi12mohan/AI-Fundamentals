# MIT 6.034 Lab 5: Bayesian Inference
# Written by 6.034 staff

from nets import *


#### Part 1: Warm-up; Ancestors, Descendents, and Non-descendents ##############

def get_ancestors(net, var):
    "Return a set containing the ancestors of var"
    ancestors = set()
    stack = [var]
    while stack:
        anc = stack.pop()
        ancestors = ancestors.union(net.get_parents(anc))
        stack.extend(list(net.get_parents(anc)))
    return ancestors

def get_descendants(net, var):
    "Returns a set containing the descendants of var"
    descendants = set()
    stack = [var]
    while stack:
        des = stack.pop()
        descendants = descendants.union(net.get_children(des))
        stack.extend(list(net.get_children(des)))
    return descendants

def get_nondescendants(net, var):
    "Returns a set containing the non-descendants of var"
    nondescendants = set(net.get_variables()).difference(get_descendants(net, var))
    nondescendants.remove(var)
    return nondescendants


#### Part 2: Computing Probability #############################################

def simplify_givens(net, var, givens):
    """
    If givens include every parent of var and no descendants, returns a
    simplified list of givens, keeping only parents.  Does not modify original
    givens.  Otherwise, if not all parents are given, or if a descendant is
    given, returns original givens.
    """
    given_set = set(givens.keys())
    parent_set = net.get_parents(var)
    descendant_set = get_descendants(net, var)

    if parent_set.issubset(given_set) and not descendant_set.intersection(given_set):
        new_givens = {}
        for given in given_set:
            if given in parent_set:
                new_givens[given] = givens[given]
        return new_givens

    return givens

def probability_lookup(net, hypothesis, givens=None):
    "Looks up a probability in the Bayes net, or raises LookupError"
    for key in hypothesis:
        hyp = key

    if givens:
        givens = simplify_givens(net, hyp, givens)

    try:
        prob = net.get_probability(hypothesis, givens)
        return prob
    except:
        raise LookupError

def probability_joint(net, hypothesis):
    "Uses the chain rule to compute a joint probability"
    probability = 1
    givens = {}

    for hyp in net.topological_sort():
        hypo = {hyp:hypothesis[hyp]}
        probability *= probability_lookup(net, hypo, givens)
        givens[hyp] = hypothesis[hyp]

    return probability

def probability_marginal(net, hypothesis):
    "Computes a marginal probability as a sum of joint probabilities"
    probability = 0
    combos = net.combinations(net.get_variables(), hypothesis)

    for combo in combos:
        probability += probability_joint(net, combo)

    return probability

def probability_conditional(net, hypothesis, givens=None):
    "Computes a conditional probability as a ratio of marginal probabilities"
    if givens:
        hyp1 = dict(hypothesis, **givens)
        hyp2 = dict(givens, **hypothesis)

        if hyp1 != hyp2:
            return 0
        else:
            return probability_marginal(net, hyp1) / probability_marginal(net, givens)

    return probability_marginal(net, hypothesis)

def probability(net, hypothesis, givens=None):
    "Calls previous functions to compute any probability"
    return probability_conditional(net, hypothesis, givens)


#### Part 3: Counting Parameters ###############################################

def number_of_parameters(net):
    """
    Computes the minimum number of parameters required for the Bayes net.
    """
    params = 0
    for var in net.get_variables():
        if net.get_parents(var):
            params += product([len(net.get_domain(parent)) for parent in net.get_parents(var)] + [len(net.get_domain(var))-1])
        else:
            params += len(net.get_domain(var))-1
    return params

#### Part 4: Independence ######################################################

def is_independent(net, var1, var2, givens=None):
    """
    Return True if var1, var2 are conditionally independent given givens,
    otherwise False. Uses numerical independence.
    """
    if not givens:
        givens = {}

    first_hyp = {var1: net.get_domain(var1)[-1]}
    sec_hyp = {var2: net.get_domain(var2)[-1]}

    first_given = dict(givens, **first_hyp)
    second_given = dict(givens, **sec_hyp)

    first_indep = approx_equal(probability(net, first_hyp, givens), probability(net, first_hyp, second_given))
    sec_indep = approx_equal(probability(net, sec_hyp, givens), probability(net, sec_hyp, first_given))

    return first_indep or sec_indep


def is_structurally_independent(net, var1, var2, givens=None):
    """
    Return True if var1, var2 are conditionally independent given givens,
    based on the structure of the Bayes net, otherwise False.
    Uses structural independence only (not numerical independence).
    """
    ancestors = get_ancestors(net, var1).union(get_ancestors(net, var2))
    new_net = [var1, var2]

    if givens:
        for given in givens:
            new_net.append(given)
            ancestors = ancestors.union(get_ancestors(net, given))

    no_ancestor_net = net.subnet(new_net)
    ancestor_net = net.subnet(new_net + list(ancestors))

    for var in no_ancestor_net.topological_sort():
        parents = list(no_ancestor_net.get_parents(var))
        for curr in range(len(parents)):
            for next in range(curr, len(parents)):
                ancestor_net.link(parents[curr], parents[next])
    ancestor_net.make_bidirectional()

    if givens:
        for given in givens:
            ancestor_net.remove_variable(given)

    return not ancestor_net.find_path(var1, var2)
