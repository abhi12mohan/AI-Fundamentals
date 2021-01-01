# MIT 6.034 Lab 4: Rule-Based Systems
# Written by 6.034 staff

from production import IF, AND, OR, NOT, THEN, DELETE, forward_chain, pretty_goal_tree
from data import *
import pprint

pp = pprint.PrettyPrinter(indent=1)
pprint = pp.pprint

#### Part 2: Transitive Rule #########################################

# Fill this in with your rule
transitive_rule = IF( AND("(?x) beats (?y)", "(?y) beats (?z)"), THEN("(?x) beats (?z)") )


#### Part 3: Family Relations #########################################

# Define your rules here. We've given you an example rule whose lead you can follow:
friend_rule = IF( AND("person (?x)", "person (?y)"), THEN ("friend (?x) (?y)", "friend (?y) (?x)") )
repeat_rule = IF(OR("person (?x)"), THEN("repeat (?x) (?x)"))
parent_and_child_rule = IF(("parent (?x) (?y)"), THEN ("child (?y) (?x)") )
child_and_sibling_rule = IF( AND("child (?x) (?y)", "child (?z) (?y)", NOT("repeat (?x) (?z)")), THEN ("sibling (?x) (?z)"))
cousin_rule = IF(AND("parent (?a) (?b)", "parent (?x) (?y)", "sibling (?x) (?a)"), THEN("cousin (?y) (?b)", "cousin (?b) (?y)"))
grandparent_and_grandchild_rule = IF( AND("parent (?x) (?y)", "parent (?y) (?z)"), THEN ("grandparent (?x) (?z)", "grandchild (?z) (?x)"))

# Add your rules to this list:
family_rules = [friend_rule, repeat_rule, parent_and_child_rule, child_and_sibling_rule, cousin_rule, grandparent_and_grandchild_rule]

# The following should generate 14 cousin relationships, representing 7 pairs
# of people who are cousins:
harry_potter_family_cousins = [
    relation for relation in
    forward_chain(family_rules, harry_potter_family_data, verbose=False)
    if "cousin" in relation ]


#### Part 4: Backward Chaining #########################################

# Import additional methods for backchaining
from production import PASS, FAIL, match, populate, simplify, variables

def backchain_to_goal_tree(rules, hypothesis):
    """
    Takes a hypothesis (string) and a list of rules (list
    of IF objects), returning an AND/OR tree representing the
    backchain of possible statements we may need to test
    to determine if this hypothesis is reachable or not.

    This method should return an AND/OR tree, that is, an
    AND or OR object, whose constituents are the subgoals that
    need to be tested. The leaves of this tree should be strings
    (possibly with unbound variables), *not* AND or OR objects.
    Make sure to use simplify(...) to flatten trees where appropriate.
    """
    goal_tree = OR(hypothesis)

    for rule in rules:
        binding_set = match(rule.consequent(), hypothesis)
        pattern = rule.antecedent()

        if binding_set is not None:
            inst_ant = [backchain_to_goal_tree(rules, populate(ant, binding_set)) for ant in pattern]

            if isinstance(pattern, str):
                bound_ant = populate(pattern, binding_set)
                goal_tree.append(bound_ant)
                goal_tree.append(backchain_to_goal_tree(rules, bound_ant))
            elif isinstance(pattern, AND):
                goal_tree.append(AND(inst_ant))
            elif isinstance(pattern, OR):
                goal_tree.append(OR(inst_ant))

    return simplify(goal_tree)
