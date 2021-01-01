# MIT 6.034 Lab 2: Games
# Written by 6.034 staff

from game_api import *
from boards import *
from toytree import GAME1

INF = float('inf')

# Please see wiki lab page for full description of functions and API.

#### Part 1: Utility Functions #################################################

def is_game_over_connectfour(board):
    """Returns True if game is over, otherwise False."""
    for chain in board.get_all_chains():
        if len(chain) >= 4:
            return True

    return True if board.count_pieces() == 42 else False

def next_boards_connectfour(board):
    """Returns a list of ConnectFourBoard objects that could result from the
    next move, or an empty list if no moves can be made."""
    if is_game_over_connectfour(board):
        return []

    return [board.add_piece(col) for col in range(7) if not board.is_column_full(col)]

def endgame_score_connectfour(board, is_current_player_maximizer):
    """Given an endgame board, returns 1000 if the maximizer has won,
    -1000 if the minimizer has won, or 0 in case of a tie."""
    for chain in board.get_all_chains(is_current_player_maximizer):
        if len(chain) >= 4:
            return 1000

    for chain in board.get_all_chains(not is_current_player_maximizer):
        if len(chain) >= 4:
            return -1000

    return 0

def endgame_score_connectfour_faster(board, is_current_player_maximizer):
    """Given an endgame board, returns an endgame score with abs(score) >= 1000,
    returning larger absolute scores for winning sooner."""
    for chain in board.get_all_chains(is_current_player_maximizer):
        if len(chain) >= 4:
            return 1000 + (42 - board.count_pieces())

    for chain in board.get_all_chains(not is_current_player_maximizer):
        if len(chain) >= 4:
            return -1000 - (42 - board.count_pieces())

    return 0

def heuristic_connectfour(board, is_current_player_maximizer):
    """Given a non-endgame board, returns a heuristic score with
    abs(score) < 1000, where higher numbers indicate that the board is better
    for the maximizer."""
    score_dict = {1:25, 2:50, 3:100}

    curr_score = 0
    for chain in board.get_all_chains(is_current_player_maximizer):
        curr_score += score_dict[len(chain)]

    other_score = 0
    for chain in board.get_all_chains(not is_current_player_maximizer):
        other_score += score_dict[len(chain)]

    if abs(curr_score - other_score) >= 1000:
        if curr_score - other_score > 1000:
            return 999
        else:
            return -999
    else:
        return curr_score - other_score

# Now we can create AbstractGameState objects for Connect Four, using some of
# the functions you implemented above.  You can use the following examples to
# test your dfs and minimax implementations in Part 2.

# This AbstractGameState represents a new ConnectFourBoard, before the game has started:
state_starting_connectfour = AbstractGameState(snapshot = ConnectFourBoard(),
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "NEARLY_OVER" from boards.py:
state_NEARLY_OVER = AbstractGameState(snapshot = NEARLY_OVER,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "BOARD_UHOH" from boards.py:
state_UHOH = AbstractGameState(snapshot = BOARD_UHOH,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)


#### Part 2: Searching a Game Tree #############################################

# Note: Functions in Part 2 use the AbstractGameState API, not ConnectFourBoard.

def dfs_maximizing(state) :
    """Performs depth-first search to find path with highest endgame score.
    Returns a tuple containing:
     0. the best path (a list of AbstractGameState objects),
     1. the score of the leaf node (a number), and
     2. the number of static evaluations performed (a number)"""
    max_path = []
    static_evals = 0

    if state.is_game_over():
        max_path, max_score, static_evals = [state], state.get_endgame_score(), 1
    else:
        children = state.generate_next_states()
        max_score = float("-inf")
        for child in children:
            child_path, child_score, child_evals = dfs_maximizing(child)
            static_evals += child_evals
            if child_score > max_score:
                max_path, max_score = [state] + child_path, child_score

    return (max_path, max_score, static_evals)

# Uncomment the line below to try your dfs_maximizing on an
# AbstractGameState representing the games tree "GAME1" from toytree.py:

# pretty_print_dfs_type(dfs_maximizing(GAME1))


def minimax_endgame_search(state, maximize=True) :
    """Performs minimax search, searching all leaf nodes and statically
    evaluating all endgame scores.  Same return type as dfs_maximizing."""
    minimax_path = []
    static_evals = 0

    if state.is_game_over():
        minimax_path, minimax_score, static_evals = [state], state.get_endgame_score(maximize), 1
    else:
        children = state.generate_next_states()
        if maximize:
            minimax_score = float("-inf")
            for child in children:
                child_path, child_score, child_evals = minimax_endgame_search(child, not maximize)
                static_evals += child_evals
                if child_score > minimax_score:
                    minimax_path, minimax_score = [state] + child_path, child_score
        else:
            minimax_score = float("inf")
            for child in children:
                child_path, child_score, child_evals = minimax_endgame_search(child, not maximize)
                static_evals += child_evals
                if child_score < minimax_score:
                    minimax_path, minimax_score = [state] + child_path, child_score

    return (minimax_path, minimax_score, static_evals)

# Uncomment the line below to try your minimax_endgame_search on an
# AbstractGameState representing the ConnectFourBoard "NEARLY_OVER" from boards.py:

# pretty_print_dfs_type(minimax_endgame_search(state_NEARLY_OVER))


def minimax_search(state, heuristic_fn=always_zero, depth_limit=INF, maximize=True) :
    """Performs standard minimax search. Same return type as dfs_maximizing."""
    minimax_path = []
    all_evals = 0

    if state.is_game_over():
        minimax_path, minimax_score, all_evals = [state], state.get_endgame_score(maximize), 1
    elif depth_limit == 0:
        minimax_path, minimax_score, all_evals = [state], heuristic_fn(state.get_snapshot(), maximize), 1
    else:
        children = state.generate_next_states()
        if maximize:
            minimax_score = float("-inf")
            for child in children:
                child_path, child_score, child_evals = minimax_search(child, heuristic_fn, depth_limit-1, not maximize)
                all_evals += child_evals
                if child_score > minimax_score:
                    minimax_path, minimax_score = [state] + child_path, child_score
        else:
            minimax_score = float("inf")
            for child in children:
                child_path, child_score, child_evals = minimax_search(child, heuristic_fn, depth_limit-1, not maximize)
                all_evals += child_evals
                if child_score < minimax_score:
                    minimax_path, minimax_score = [state] + child_path, child_score

    return (minimax_path, minimax_score, all_evals)


# Uncomment the line below to try minimax_search with "BOARD_UHOH" and
# depth_limit=1. Try increasing the value of depth_limit to see what happens:

# pretty_print_dfs_type(minimax_search(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=1))


def minimax_search_alphabeta(state, alpha=-INF, beta=INF, heuristic_fn=always_zero,
                             depth_limit=INF, maximize=True) :
    """"Performs minimax with alpha-beta pruning. Same return type
    as dfs_maximizing."""
    minimax_path = []
    all_evals = 0

    if state.is_game_over():
        minimax_path, minimax_score, all_evals = [state], state.get_endgame_score(maximize), 1
    elif depth_limit == 0:
        minimax_path, minimax_score, all_evals = [state], heuristic_fn(state.get_snapshot(), maximize), 1
    else:
        children = state.generate_next_states()
        if maximize:
            minimax_score = float("-inf")
            for child in children:
                child_path, child_score, child_evals = minimax_search_alphabeta(child, alpha, beta, heuristic_fn, depth_limit-1, not maximize)
                all_evals += child_evals
                if child_score > minimax_score:
                    minimax_path, minimax_score = [state] + child_path, child_score

                alpha = max(alpha, minimax_score)
                if alpha >= beta:
                    return (minimax_path, minimax_score, all_evals)
        else:
            minimax_score = float("inf")
            for child in children:
                child_path, child_score, child_evals = minimax_search_alphabeta(child, alpha, beta, heuristic_fn, depth_limit-1, not maximize)
                all_evals += child_evals
                if child_score < minimax_score:
                    minimax_path, minimax_score = [state] + child_path, child_score

                beta = min(beta, minimax_score)
                if alpha >= beta:
                    return (minimax_path, minimax_score, all_evals)

    return (minimax_path, minimax_score, all_evals)


# Uncomment the line below to try minimax_search_alphabeta with "BOARD_UHOH" and
# depth_limit=4. Compare with the number of evaluations from minimax_search for
# different values of depth_limit.

# pretty_print_dfs_type(minimax_search_alphabeta(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4))


def progressive_deepening(state, heuristic_fn=always_zero, depth_limit=INF,
                          maximize=True) :
    """Runs minimax with alpha-beta pruning. At each level, updates anytime_value
    with the tuple returned from minimax_search_alphabeta. Returns anytime_value."""
    anytime_value = AnytimeValue()
    for i in range(1, depth_limit+1):
        anytime_value.set_value(minimax_search_alphabeta(state, -INF, INF, heuristic_fn, i, maximize))
    return anytime_value

# Uncomment the line below to try progressive_deepening with "BOARD_UHOH" and
# depth_limit=4. Compare the total number of evaluations with the number of
# evaluations from minimax_search or minimax_search_alphabeta.

# progressive_deepening(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4).pretty_print()

TEST_PROGRESSIVE_DEEPENING = True
if not TEST_PROGRESSIVE_DEEPENING:
    def not_implemented(*args): raise NotImplementedError
    progressive_deepening = not_implemented
