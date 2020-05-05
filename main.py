import copy
import sys
from collections import defaultdict
import numpy as np
sys.setrecursionlimit(100000)

class Game:
    def __init__(self, root, payoffs, num_players, info_sets):
        """
        Create a new instance of the game.
        """
        self.root = root
        self.payoffs = payoffs
        self.num_players = num_players
        self.info_sets = info_sets

class Node:
    def __init__(self, node_id, actions= {}, outcome=None, player=1):
        self.actions = actions
        self.outcome = outcome
        self.player = player
        self.node_id = node_id

class Strategy:
    def __init__(self, outcome, actions = {}):
        self.outcome = outcome
        self.actions = actions

    def __repr__(self):
        if self.actions is None:
            return f"Outcome: {str(self.outcome)}"

        return f"Outcome: {str(self.outcome)}, Actions: {self.actions}"

def get_info_set(game, node): 
    for info_set in game.info_sets: 
        if node.node_id in info_set:
            return info_set

    return set([node.node_id])

def get_subperf_equil(game, node, trace=False):
    '''If node is a subgame, return a dictionary of all subgame perfect equillibria.
       The key of the dictionary is the node you want the strategy for, and it maps to a list
       of all of the strategies.
       Otherwise, return the current subtree as a set of nodes.'''
    print(f'>>> Looking at {node.node_id} <<<')

    # Check if curr node is in its own information set.
    curr_subtree = set()
    for info_set in game.info_sets:
        if node.node_id in info_set and len(info_set) != 1:
            if trace:
                print(f"{node.node_id} is not in its own info set.")
            curr_subtree.add(node.node_id)

    # Base case for leaf node.
    if len(node.actions.keys()) == 0:
        if trace:
            print(f"{node.node_id}: This is a leaf!")
        if len(curr_subtree) == 0:
            return {node.node_id: [Strategy(node.outcome)]}, None

        return None, curr_subtree

    # Call function recursively for all actions.
    all_strategies = defaultdict(list)
    for action in node.actions.keys():
        strategies, subtree = get_subperf_equil(game, node.actions[action], trace)
        if strategies is not None:
            for strategy_node in strategies.keys():
                all_strategies[strategy_node] += strategies[strategy_node]
        
        if subtree is not None:
            curr_subtree = curr_subtree.union(subtree)

    if trace:
        print(f"{node.node_id}: After calling recursively the subtree is: {curr_subtree}")

    # At this point curr_subtree is the set of all nodes_ids that made the
    # subtree NOT a subgame. Remove of nodes from it whose information sets
    # are now in the subtree.
    new_subtree = set()
    for node_id in curr_subtree:
        for info_set in game.info_sets:
            if node_id in info_set and not info_set.issubset(curr_subtree):
                new_subtree.add(node_id)
            elif node_id in info_set:
                continue

    if trace:
        print(f"{node.node_id}: After removing nodes from subtree it is: {new_subtree}")

    # Still not a valid game, return the current strategies we have but not find a
    # nash equil.
    if len(new_subtree) != 0:
        return all_strategies, curr_subtree

    if trace:
        print(f"{node.node_id}: valid subgame.")
        print(f"{node.node_id}: {all_strategies}")

    # It is a valid subgame - find the Nash equil.
    return get_nash_equil(game, node, all_strategies)

def get_payoff(p1_strategy, p2_strategy, node, game, info_set_p1, info_set_p2, all_strategies): 
    while True: 
        # If the node has already been solved, it's already in our strategy set.
        if node.node_id in all_strategies:
            outcome = all_strategies[node.node_id][0].outcome
            return game.payoffs[outcome], outcome

        if node.player == 0: 
            # Find the information set this node is part of 
            for i, info_set in enumerate(info_set_p1.keys()):
                if node.node_id in info_set:
                    action_to_take = p1_strategy[i]
        else: 
            for i, info_set in enumerate(info_set_p2.keys()):
                if node.node_id in info_set:
                    action_to_take = p2_strategy[i]

        node = node.actions[list(node.actions.keys())[action_to_take]]
        if node.outcome is not None:
            return game.payoffs[node.outcome], node.outcome

def get_nash_equil(game, node, all_strategies):
    # First, find all of the information sets for each player (here we assume num_players = 2).
    # Note: This is slow. Pass the nodes up the same way we check for if something is a subgame?
    info_sets_p1 = {}
    info_sets_p2 = {}

    num_strategies_p1 = 1
    num_strategies_p2 = 1
    
    outcomes = {}
    
    unvisited = []
    unvisited += [node]

    # Figure out which information sets are relevant for this subgame, 
    # and how many possible strategies there are.
    while len(unvisited) != 0: 
        curr_node = unvisited[-1]
        unvisited = unvisited[:-1]

        # If the node has already been solved, it should not count. 
        if curr_node.node_id in all_strategies:
            continue

        for action in curr_node.actions:
            unvisited += [curr_node.actions[action]]
        
        info_set = tuple(get_info_set(game, curr_node))
        if curr_node.player == 0 and info_set not in info_sets_p1.keys():
            if len(curr_node.actions) != 0:
                info_sets_p1[info_set] = len(curr_node.actions)
                num_strategies_p1 *= len(curr_node.actions)
        elif curr_node.player == 1 and info_set not in info_sets_p2.keys():
            if len(curr_node.actions) != 0:
                info_sets_p2[info_set] = len(curr_node.actions)
                num_strategies_p2 *= len(curr_node.actions)

    p1_strategy = [0] * len(info_sets_p1.keys())
    p2_strategy = [0] * len(info_sets_p2.keys())

    payoff_matrix = np.zeros((num_strategies_p1, num_strategies_p2, 2))
    payoff_matrix_strategies = {} 
    for i in range(num_strategies_p1):
        for j in range(num_strategies_p2):
            payoff, _ = get_payoff(p1_strategy, p2_strategy, node, game, info_sets_p1, info_sets_p2, all_strategies)
            payoff_matrix[i, j] = payoff
            payoff_matrix_strategies[(i, j)] = (tuple(p1_strategy), tuple(p2_strategy))

            for p, key in enumerate(info_sets_p2.keys()):
                if p2_strategy[p] < info_sets_p2[key] - 1:
                    p2_strategy[p] += 1
                    break
            
                p2_strategy[p] = 0

        for p, key in enumerate(info_sets_p1.keys()):
            if p1_strategy[p] < info_sets_p1[key] - 1:
                p1_strategy[p] += 1
                break
            
    # For each row, pick p2's best responses.
    all_best_responses = []
    for i in range(num_strategies_p1):
        best_responses = []
        best_response_val = -float('inf')
        for j in range(num_strategies_p2): 
            if payoff_matrix[i, j, 1] == best_response_val:
                best_responses += [(i, j)]
            elif payoff_matrix[i, j, 1] > best_response_val:
                best_response_val = payoff_matrix[i, j, 1]
                best_responses = [(i, j)]
        all_best_responses += best_responses

    # For each col, pick p1's best response.
    all_best_responses_2 = []
    for j in range(num_strategies_p2):
        best_responses = []
        best_response_val = -float('inf')
        for i in range(num_strategies_p1): 
            if payoff_matrix[i, j, 0] == best_response_val:
                best_responses += [(i, j)]
            elif payoff_matrix[i, j, 0] > best_response_val:
                best_response_val = payoff_matrix[i, j, 0]
                best_responses = [(i, j)]
        all_best_responses_2 += best_responses

    # Get the strategies for each index in the payoff matrix.
    equils = set(all_best_responses).intersection(set(all_best_responses_2))
    strats = []
    for e in equils:
        strats += [payoff_matrix_strategies[e]]

    # Make the new strategies
    new_strategies = defaultdict(list)
    for strat in strats:
        p_0_moves = strat[0]
        p_1_moves = strat[1]
        actions = {}
        for i in range(len(p_0_moves)):
            info_set = list(info_sets_p1.keys())[i]
            for curr_node in info_set:
                actions[curr_node] = p_0_moves[i]

        for i in range(len(p_1_moves)):
            info_set = list(info_sets_p2.keys())[i]
            for curr_node in info_set:
                actions[curr_node] = p_1_moves[i]

        new_strat = Strategy(get_payoff(p_0_moves, p_1_moves, node, game, info_sets_p1, info_sets_p2, all_strategies)[1], actions)
        all_substrategies = update(new_strat, all_strategies, node)
        for substrategy in all_substrategies:  
            subactions = copy.deepcopy(substrategy.actions)
            new_strat.actions.update(subactions)
            new_strategies[node.node_id] += [new_strat]

    return new_strategies, None

def update(strat, all_strategies, node):
    while True: 
        if node.node_id in all_strategies:
            return all_strategies[node.node_id]
        
        action_idx = strat.actions[node.node_id]
        node = node.actions[list(node.actions.keys())[action_idx]]
    return strat

def generate_centipede_nodes(n, k=2, k2=None):
    info_sets = []
    # Set state for the end of the game
    if n % 2 == 0:
        curr_player = 0
    else:
        curr_player = 1

    # Constant amount added to pot each turn.
    if k2 is None:
        pot_amounts = [k*(i+1) for i in range(1, n+1)]
    else:
        pot_amounts = [2 * k]
        amount_to_add = k
        for i in range(2, n+1):
            pot_amounts += [pot_amounts[-1] + amount_to_add]
            amount_to_add -= k2
    nodes = []
    payoffs = {}
    node_id = 0
    for round_num in range(n, 0, -1):
        pot_amount = pot_amounts[round_num - 1]
        # At each round, the current player can take:
        if curr_player == 0:
            payoffs[f'Take: {pot_amount}'] = [(pot_amount / 2) + 2, (pot_amount / 2) - 2]
        else:
            payoffs[f'Take: {pot_amount}'] = [(pot_amount / 2) - 2, (pot_amount / 2) + 2]
        take_node = Node(node_id, {}, f'Take: {pot_amount}', curr_player)
        info_sets.append(set([node_id]))
        node_id += 1

        # The player can also pass (UNLESS IT IS THE LAST ROUND)
        if round_num != n:
            pass_node = Node(node_id, {'Take'}, None, curr_player)
            info_sets.append(set([node_id]))
            node_id += 1
            curr_node = Node(node_id,
                {'Take': take_node,
                 'Pass': nodes[-1]},
                None, curr_player)
            info_sets.append(set([node_id]))
            node_id += 1
            nodes.append(curr_node)
        else:
            split_node = Node(node_id, {}, f'Split: {pot_amount}', curr_player)
            payoffs[f'Split: {pot_amount}'] = [pot_amount / 2, pot_amount / 2]
            info_sets.append(set([node_id]))
            node_id += 1
            curr_node = Node(node_id,
                {'Take': take_node,
                 'Split': split_node},
                None, curr_player)
            info_sets.append(set([node_id]))
            node_id += 1
            nodes.append(curr_node)

        if curr_player == 0:
            curr_player = 1
        else:
            curr_player = 0

    return payoffs, nodes[-1], info_sets

def generate_incomplete_game():
    # Leaf nodes
    node_15  = Node(15, {}, 'H', 0)
    node_14  = Node(14, {}, 'G', 0)
    node_13  = Node(13, {}, 'F', 0)
    node_12  = Node(12, {}, 'E', 0)
    node_11  = Node(11, {}, 'D', 0)
    node_10  = Node(10, {}, 'C', 0)
    node_9   = Node(9,  {}, 'B', 0)
    node_8   = Node(8,  {}, 'A', 0)

    node_7   = Node(7, {'m': node_14, 'n': node_15}, None, 0)
    node_6   = Node(6, {'k': node_12, 'l': node_13}, None, 0)
    node_5   = Node(5, {'i': node_10, 'j': node_11}, None, 0)
    node_4   = Node(4, {'g': node_8,  'h': node_9},  None, 0)

    node_3   = Node(3, {'e': node_6, 'f': node_7}, None, 1)
    node_2   = Node(2, {'c': node_4, 'd': node_5}, None, 1)

    node_1   = Node(1, {'a': node_2, 'b': node_3}, None, 0)

    payoffs  = {'A': [2, 0], 'B': [0, 1], 'C': [2, 0], 'D': [0, 0], 
                'E': [0, 0], 'F': [1, 0], 'G': [0, 0], 'H': [1, 0]}
    
    info_sets = [set([2, 3])]
    return payoffs, node_1, info_sets

def main():
    # payoffs, root, info_sets = generate_centipede_nodes(10, 8, 0)
    # num_players = 2
    # game = Game(root, payoffs, num_players, info_sets)
    # res, subtree = get_subperf_equil(game, game.root)
    # print(res)

    payoffs, root, info_sets = generate_incomplete_game()
    num_players = 2
    game = Game(root, payoffs, num_players, info_sets)
    res, subtree = get_subperf_equil(game, game.root, True)
    print(res, subtree)

if __name__ == "__main__":
    main()
