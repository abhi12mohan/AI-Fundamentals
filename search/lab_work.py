# MIT 6.034 Lab 1: Search
# Written by 6.034 staff

from search import Edge, UndirectedGraph, do_nothing_fn, make_generic_search
import read_graphs
from functools import reduce

all_graphs = read_graphs.get_graphs()
GRAPH_0 = all_graphs['GRAPH_0']
GRAPH_1 = all_graphs['GRAPH_1']
GRAPH_2 = all_graphs['GRAPH_2']
GRAPH_3 = all_graphs['GRAPH_3']
GRAPH_FOR_HEURISTICS = all_graphs['GRAPH_FOR_HEURISTICS']


# Please see wiki lab page for full description of functions and API.

#### PART 1: Helper Functions ##################################################

def path_length(graph, path):
    """Returns the total length (sum of edge weights) of a path defined by a
    list of nodes coercing an edge-linked traversal through a graph.
    (That is, the list of nodes defines a path through the graph.)
    A path with fewer than 2 nodes should have length of 0.
    You can assume that all edges along the path have a valid numeric weight."""
    length =  0
    for i in range(len(path)-1):
        length += graph.get_edge(path[i], path[i+1]).length
    return length

def has_loops(path):
    """Returns True if this path has a loop in it, i.e. if it
    visits a node more than once. Returns False otherwise."""
    visited = set()
    for node in path:
        if node in visited:
            return True
        else:
            visited.add(node)
    return False

def extensions(graph, path):
    """Returns a list of paths. Each path in the list should be a one-node
    extension of the input path, where an extension is defined as a path formed
    by adding a neighbor node (of the final node in the path) to the path.
    Returned paths should not have loops, i.e. should not visit the same node
    twice. The returned paths should be sorted in lexicographic order."""
    return [path + [neighbor] for neighbor in sorted(graph.get_neighbors(path[-1])) if neighbor not in set(path)]

def sort_by_heuristic(graph, goalNode, nodes):
    """Given a list of nodes, sorts them best-to-worst based on the heuristic
    from each node to the goal node. Here, and in general for this lab, we
    consider a smaller heuristic value to be "better" because it represents a
    shorter potential path to the goal. Break ties lexicographically by
    node name."""
    return sorted(nodes, key=lambda node: (graph.get_heuristic_value(node, goalNode), node))

# You can ignore the following line.  It allows generic_search (PART 3) to
# access the extensions and has_loops functions that you just defined in PART 1.
generic_search = make_generic_search(extensions, has_loops)  # DO NOT CHANGE


#### PART 2: Basic Search ######################################################

def basic_dfs(graph, startNode, goalNode):
    """
    Performs a depth-first search on a graph from a specified start
    node to a specified goal node, returning a path-to-goal if it
    exists, otherwise returning None.
    Uses backtracking, but does not use an extended set.
    """
    queue = [[startNode]]
    while queue:
        curr_path = queue.pop(0)
        if curr_path[-1] == goalNode:
            return curr_path
        else:
            queue = extensions(graph, curr_path) + queue
    return None

def basic_bfs(graph, startNode, goalNode):
    """
    Performs a breadth-first search on a graph from a specified start
    node to a specified goal node, returning a path-to-goal if it
    exists, otherwise returning None.
    """
    queue = [[startNode]]
    while queue:
        curr_path = queue.pop(0)
        if curr_path[-1] == goalNode:
            return curr_path
        else:
            queue += extensions(graph, curr_path)
    return None


#### PART 3: Generic Search ####################################################

# Generic search requires four arguments (see wiki for more details):
# sort_new_paths_fn: a function that sorts new paths that are added to the agenda
# add_paths_to_front_of_agenda: True if new paths should be added to the front of the agenda
# sort_agenda_fn: function to sort the agenda after adding all new paths
# use_extended_set: True if the algorithm should utilize an extended set


# Define your custom path-sorting functions here.
# Each path-sorting function should be in this form:
# def my_sorting_fn(graph, goalNode, paths):
#     # YOUR CODE HERE
#     return sorted_paths

def lexi_sort(graph, goalNode, paths):
    return sorted(paths)

def curr_heuristic_sort(graph, goalNode, paths):
    return sorted(paths, key=lambda path: graph.get_heuristic_value(path[-1], goalNode))

def path_length_sort(graph, goalNode, paths):
    return sorted(paths, key=lambda path: path_length(graph, path))

def path_heuristic_length_sort(graph, goalNode, paths):
    return sorted(paths, key=lambda path: graph.get_heuristic_value(path[-1], goalNode) + path_length(graph, path))

generic_dfs = [lexi_sort, True, do_nothing_fn, False]

generic_bfs = [lexi_sort, False, do_nothing_fn, False]

generic_hill_climbing = [curr_heuristic_sort, True, do_nothing_fn, False]

generic_best_first = [do_nothing_fn, True, curr_heuristic_sort, False]

generic_branch_and_bound = [do_nothing_fn, False, path_length_sort, False]

generic_branch_and_bound_with_heuristic = [do_nothing_fn, False, path_heuristic_length_sort, False]

generic_branch_and_bound_with_extended_set = [do_nothing_fn, False, path_length_sort, True]

generic_a_star = [do_nothing_fn, False, path_heuristic_length_sort, True]

#### PART 4: Heuristics ########################################################

def is_admissible(graph, goalNode):
    """Returns True if this graph's heuristic is admissible; else False.
    A heuristic is admissible if it is either always exactly correct or overly
    optimistic; it never over-estimates the cost to the goal."""
    a_fn = generic_search(*generic_a_star)
    for node in graph.nodes:
        if graph.get_heuristic_value(node, goalNode) > path_length(graph, a_fn(graph, goalNode, node)):
            return False
    return True

def is_consistent(graph, goalNode):
    """Returns True if this graph's heuristic is consistent; else False.
    A consistent heuristic satisfies the following property for all
    nodes v in the graph:
        Suppose v is a node in the graph, and N is a neighbor of v,
        then, heuristic(v) <= heuristic(N) + edge_weight(v, N)
    In other words, moving from one node to a neighboring node never unfairly
    decreases the heuristic.
    This is equivalent to the heuristic satisfying the triangle inequality."""
    for edge in graph.edges:
        if edge.length < abs(graph.get_heuristic_value(edge.startNode, goalNode) - graph.get_heuristic_value(edge.endNode, goalNode)):
            return False
    return True
