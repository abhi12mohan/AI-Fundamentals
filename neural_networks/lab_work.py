# MIT 6.034 Lab 7: Neural Nets
# Written by 6.034 Staff

from nn_problems import *
from math import e
INF = float('inf')


#### Part 2: Coding Warmup #####################################################

# Threshold functions
def stairstep(x, threshold=0):
    "Computes stairstep(x) using the given threshold (T)"
    return x >= threshold

def sigmoid(x, steepness=1, midpoint=0):
    "Computes sigmoid(x) using the given steepness (S) and midpoint (M)"
    return (1 / (1 + pow(e, -steepness*(x-midpoint))))

def ReLU(x):
    "Computes the threshold of an input using a rectified linear unit."
    max_val = max(0, x)
    return max_val

# Accuracy function
def accuracy(desired_output, actual_output):
    "Computes accuracy. If output is binary, accuracy ranges from -0.5 to 0."
    return -0.5 * pow(desired_output - actual_output, 2)


#### Part 3: Forward Propagation ###############################################

def node_value(node, input_values, neuron_outputs):  # PROVIDED BY THE STAFF
    """
    Given
     * a node (as an input or as a neuron),
     * a dictionary mapping input names to their values, and
     * a dictionary mapping neuron names to their outputs
    returns the output value of the node.
    This function does NOT do any computation; it simply looks up
    values in the provided dictionaries.
    """
    if isinstance(node, str):
        # A string node (either an input or a neuron)
        if node in input_values:
            return input_values[node]
        if node in neuron_outputs:
            return neuron_outputs[node]
        raise KeyError("Node '{}' not found in either the input values or neuron outputs dictionary.".format(node))

    if isinstance(node, (int, float)):
        # A constant input, such as -1
        return node

    raise TypeError("Node argument is {}; should be either a string or a number.".format(node))

def forward_prop(net, input_values, threshold_fn=stairstep):
    """Given a neural net and dictionary of input values, performs forward
    propagation with the given threshold function to compute binary output.
    This function should not modify the input net.  Returns a tuple containing:
    (1) the final output of the neural net
    (2) a dictionary mapping neurons to their immediate outputs"""
    outputs = input_values.copy()
    for neuron in net.topological_sort():
        curr_val = 0
        for node in net.get_incoming_neighbors(neuron):
            wire = net.get_wires(node, neuron)[0]
            node_val = node_value(wire.startNode, input_values, outputs)
            curr_val += (wire.get_weight() * node_val)
        outputs[neuron] = threshold_fn(curr_val)
    final_net = outputs[net.get_output_neuron()]
    return (final_net, outputs)


#### Part 4: Backward Propagation ##############################################

def gradient_ascent_step(func, inputs, step_size):
    """Given an unknown function of three variables and a list of three values
    representing the current inputs into the function, increments each variable
    by +/- step_size or 0, with the goal of maximizing the function output.
    After trying all possible variable assignments, returns a tuple containing:
    (1) the maximum function output found, and
    (2) the list of inputs that yielded the highest function output."""
    max_val = float("-inf")
    max_input = []
    poss_steps = [0, -step_size, step_size]

    for step1 in poss_steps:
        for step2 in poss_steps:
            for step3 in poss_steps:
                input = (inputs[0]+step1, inputs[1]+step2, inputs[2]+step3)
                val = func(input[0], input[1], input[2])
                if val > max_val:
                    max_val = val
                    max_input = input

    return (max_val, max_input)

from collections import deque
def get_back_prop_dependencies(net, wire):
    """Given a wire in a neural network, returns a set of inputs, neurons, and
    Wires whose outputs/values are required to update this wire's weight."""
    dependency_set = set([wire, wire.startNode, wire.endNode])
    seen = set()

    queue = deque()
    queue.append(wire.endNode)

    while queue:
        curr = queue.popleft()
        dependency_set.add(curr)

        for wire in net.get_wires(curr):
            if wire.endNode not in seen:
                queue.append(wire.endNode)
                seen.add(wire.endNode)
            dependency_set.add(wire)

    return dependency_set

def calculate_deltas(net, desired_output, neuron_outputs):
    """Given a neural net and a dictionary of neuron outputs from forward-
    propagation, computes the update coefficient (delta_B) for each
    neuron in the net. Uses the sigmoid function to compute neuron output.
    Returns a dictionary mapping neuron names to update coefficient (the
    delta_B values). """
    deltas = {}
    reverse_order = net.topological_sort()
    reverse_order.reverse()

    for neuron in reverse_order:
        val = neuron_outputs[neuron]
        if net.is_output_neuron(neuron):
            deltas[neuron] = val * (1 - val) * (desired_output - val)
        else:
            total = 0
            for wire in net.get_wires(neuron):
                total += wire.get_weight() * deltas[wire.endNode]
            deltas[neuron] = val * (1 - val) * total

    return deltas

def update_weights(net, input_values, desired_output, neuron_outputs, r=1):
    """Performs a single step of back-propagation.  Computes delta_B values and
    weight updates for entire neural net, then updates all weights.  Uses the
    sigmoid function to compute neuron output.  Returns the modified neural net,
    with the updated weights."""
    delta_B = calculate_deltas(net, desired_output, neuron_outputs)
    for wire in net.get_wires():
        node_val = node_value(wire.startNode, input_values, neuron_outputs)
        delta_val = delta_B[wire.endNode]
        weight = wire.get_weight() + (r * node_val * delta_val)
        wire.set_weight(weight)
    return net

def back_prop(net, input_values, desired_output, r=1, minimum_accuracy=-0.001):
    """Updates weights until accuracy surpasses minimum_accuracy.  Uses the
    sigmoid function to compute neuron output.  Returns a tuple containing:
    (1) the modified neural net, with trained weights
    (2) the number of iterations (that is, the number of weight updates)"""
    forward = forward_prop(net, input_values, sigmoid)
    iterations = 0

    while accuracy(desired_output, forward[0]) <= minimum_accuracy:
        net = update_weights(net, input_values, desired_output, forward[1], r)
        forward = forward_prop(net, input_values, sigmoid)
        iterations += 1

    return (net, iterations)
