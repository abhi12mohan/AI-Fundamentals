# MIT 6.034 Lab 3: Constraint Satisfaction Problems
# Written by 6.034 staff

from constraint_api import *
from test_problems import get_pokemon_problem


#### Part 1: Warmup ############################################################

def has_empty_domains(csp) :
    """Returns True if the problem has one or more empty domains, otherwise False"""
    for domain in csp.domains:
        if len(csp.domains[domain]) == 0:
            return True
    return False

def check_all_constraints(csp) :
    """Return False if the problem's assigned values violate some constraint,
    otherwise True"""
    for var1 in csp.assignments:
        for var2 in csp.assignments:
            for constraint in csp.constraints_between(var1, var2):
                if not constraint.check(csp.get_assignment(var1), csp.get_assignment(var2)):
                    return False
    return True


#### Part 2: Depth-First Constraint Solver #####################################

def solve_constraint_dfs(problem):
    """
    Solves the problem using depth-first search.  Returns a tuple containing:
    1. the solution (a dictionary mapping variables to assigned values)
    2. the number of extensions made (the number of problems popped off the agenda).
    If no solution was found, return None as the first element of the tuple.
    """
    agenda = [problem]
    extension_count = 0
    while agenda:
        csp = agenda.pop(0)
        extension_count += 1
        if not has_empty_domains(csp) and check_all_constraints(csp):
            if not csp.unassigned_vars:
                return (csp.assignments, extension_count)
            else:
                new_probs = []
                var = csp.pop_next_unassigned_var()
                for val in csp.get_domain(var):
                    new_csp = csp.copy()
                    new_csp.set_assignment(var, val)
                    new_probs.append(new_csp)
                agenda = new_probs + agenda

    return (None, extension_count)

# print(solve_constraint_dfs(get_pokemon_problem()))

# QUESTION 1: How many extensions does it take to solve the Pokemon problem
#    with DFS?

# Hint: Use get_pokemon_problem() to get a new copy of the Pokemon problem
#    each time you want to solve it with a different search method.

ANSWER_1 = 20


#### Part 3: Forward Checking ##################################################

def eliminate_from_neighbors(csp, var) :
    """
    Eliminates incompatible values from var's neighbors' domains, modifying
    the original csp.  Returns an alphabetically sorted list of the neighboring
    variables whose domains were reduced, with each variable appearing at most
    once.  If no domains were reduced, returns empty list.
    If a domain is reduced to size 0, quits immediately and returns None.
    """
    def get_new_domain(csp, var, neighbor_domain):
        new_domain = []
        for neighbor_val in neighbor_domain:
            violations = 0
            for var_val in csp.get_domain(var):
                failures = sum(not constraint.check(neighbor_val, var_val) for constraint in csp.constraints_between(var, neighbor))
                if failures > 0:
                    violations += 1
            if violations < len(csp.get_domain(var)):
                new_domain.append(neighbor_val)
        return new_domain

    changed_neighbors = []
    for neighbor in csp.get_neighbors(var):
        neighbor_domain = csp.get_domain(neighbor)
        new_domain = get_new_domain(csp, var, neighbor_domain)

        # Set domain with new values
        csp.set_domain(neighbor, new_domain)
        if not new_domain:
            return None
        elif len(neighbor_domain) > len(new_domain):
            changed_neighbors.append(neighbor)

    return sorted(changed_neighbors)

# Because names give us power over things (you're free to use this alias)
forward_check = eliminate_from_neighbors

def solve_constraint_forward_checking(problem) :
    """
    Solves the problem using depth-first search with forward checking.
    Same return type as solve_constraint_dfs.
    """
    agenda = [problem]
    extension_count = 0
    while agenda:
        csp = agenda.pop(0)
        extension_count += 1
        if not has_empty_domains(csp) and check_all_constraints(csp):
            if not csp.unassigned_vars:
                return (csp.assignments, extension_count)
            else:
                new_probs = []
                var = csp.pop_next_unassigned_var()
                for val in csp.get_domain(var):
                    new_csp = csp.copy()
                    new_csp.set_assignment(var, val)
                    forward_check(new_csp, var)
                    new_probs.append(new_csp)
                agenda = new_probs + agenda

    return (None, extension_count)

# QUESTION 2: How many extensions does it take to solve the Pokemon problem
#    with DFS and forward checking?

ANSWER_2 = 9


#### Part 4: Domain Reduction ##################################################

def domain_reduction(csp, queue=None) :
    """
    Uses constraints to reduce domains, propagating the domain reduction
    to all neighbors whose domains are reduced during the process.
    If queue is None, initializes propagation queue by adding all variables in
    their default order.
    Returns a list of all variables that were dequeued, in the order they
    were removed from the queue.  Variables may appear in the list multiple times.
    If a domain is reduced to size 0, quits immediately and returns None.
    This function modifies the original csp.
    """
    if queue is None:
        queue = csp.get_all_variables()
    dequeued = []

    while queue:
        var = queue.pop(0)
        dequeued.append(var)
        changed_neighbors = eliminate_from_neighbors(csp, var)
        if changed_neighbors is None:
            return None
        else:
            for changed_n in changed_neighbors:
                if changed_n not in queue:
                    queue.append(changed_n)
    return dequeued


# QUESTION 3: How many extensions does it take to solve the Pokemon problem
#    with DFS (no forward checking) if you do domain reduction before solving it?

ANSWER_3 = 6


def solve_constraint_propagate_reduced_domains(problem) :
    """
    Solves the problem using depth-first search with forward checking and
    propagation through all reduced domains.  Same return type as
    solve_constraint_dfs.
    """
    agenda = [problem]
    extension_count = 0
    while agenda:
        csp = agenda.pop(0)
        extension_count += 1
        if not has_empty_domains(csp) and check_all_constraints(csp):
            if not csp.unassigned_vars:
                return (csp.assignments, extension_count)
            else:
                new_probs = []
                var = csp.pop_next_unassigned_var()
                for val in csp.get_domain(var):
                    new_csp = csp.copy()
                    new_csp.set_assignment(var, val)
                    domain_reduction(new_csp, [var])
                    forward_check(new_csp, var)
                    new_probs.append(new_csp)
                agenda = new_probs + agenda

    return (None, extension_count)

# QUESTION 4: How many extensions does it take to solve the Pokemon problem
#    with forward checking and propagation through reduced domains?

ANSWER_4 = 7


#### Part 5A: Generic Domain Reduction #########################################

def propagate(enqueue_condition_fn, csp, queue=None) :
    """
    Uses constraints to reduce domains, modifying the original csp.
    Uses enqueue_condition_fn to determine whether to enqueue a variable whose
    domain has been reduced. Same return type as domain_reduction.
    """
    if queue is None:
        queue = csp.get_all_variables()
    dequeued = []

    while queue:
        var = queue.pop(0)
        dequeued.append(var)
        changed_neighbors = eliminate_from_neighbors(csp, var)
        if changed_neighbors is None:
            return None
        else:
            for changed_n in changed_neighbors:
                if changed_n not in queue and enqueue_condition_fn(csp, changed_n):
                    queue.append(changed_n)

    return dequeued

def condition_domain_reduction(csp, var) :
    """Returns True if var should be enqueued under the all-reduced-domains
    condition, otherwise False"""
    return True

def condition_singleton(csp, var) :
    """Returns True if var should be enqueued under the singleton-domains
    condition, otherwise False"""
    return len(csp.get_domain(var)) == 1

def condition_forward_checking(csp, var) :
    """Returns True if var should be enqueued under the forward-checking
    condition, otherwise False"""
    return False


#### Part 5B: Generic Constraint Solver ########################################

def solve_constraint_generic(problem, enqueue_condition=None) :
    """
    Solves the problem, calling propagate with the specified enqueue
    condition (a function). If enqueue_condition is None, uses DFS only.
    Same return type as solve_constraint_dfs.
    """
    if not enqueue_condition:
        return solve_constraint_dfs(problem)

    agenda = [problem]
    extension_count = 0
    while agenda:
        csp = agenda.pop(0)
        extension_count += 1
        if not has_empty_domains(csp) and check_all_constraints(csp):
            if not csp.unassigned_vars:
                return (csp.assignments, extension_count)
            else:
                new_probs = []
                var = csp.pop_next_unassigned_var()
                for val in csp.get_domain(var):
                    new_csp = csp.copy()
                    new_csp.set_assignment(var, val)
                    propagate(enqueue_condition, new_csp, [var])
                    new_probs.append(new_csp)
                agenda = new_probs + agenda

    return (None, extension_count)

# QUESTION 5: How many extensions does it take to solve the Pokemon problem
#    with forward checking and propagation through singleton domains? (Don't
#    use domain reduction before solving it.)

ANSWER_5 = 8


#### Part 6: Defining Custom Constraints #######################################

def constraint_adjacent(m, n) :
    """Returns True if m and n are adjacent, otherwise False.
    Assume m and n are ints."""
    return abs(m-n) == 1

def constraint_not_adjacent(m, n) :
    """Returns True if m and n are NOT adjacent, otherwise False.
    Assume m and n are ints."""
    return abs(m-n) != 1

def all_different(variables) :
    """Returns a list of constraints, with one difference constraint between
    each pair of variables."""
    return [Constraint(variables[i], variables[j], constraint_different) for i in range(len(variables)-1) for j in range(i+1, len(variables))]
