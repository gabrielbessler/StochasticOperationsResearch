import copy
import sys
from collections import defaultdict
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
    max_payoff = -float('inf')
    max_payoff_actions = []
    for action in node.actions:
        outcome = all_strategies[node.actions[action].node_id][0].outcome
        payoff = game.payoffs[outcome][node.player]
        if payoff > max_payoff:
            max_payoff = payoff
            max_payoff_actions = [action]
        elif payoff == max_payoff:
            max_payoff_actions += [action]

    new_strategies = defaultdict(list)
    for action in max_payoff_actions: 
        for strategy in all_strategies[node.actions[action].node_id]: 
            new_actions = copy.deepcopy(strategy.actions) # each strategy is a dictionary, so copy it
            new_actions[node.node_id] = action
            new_strategies[node.node_id] += [Strategy(strategy.outcome, new_actions)]

    return new_strategies, None

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
    
    node_six  = Node(6, {}, 'D', 0) # leaf
    node_five  = Node(5, {}, 'C', 0) # leaf
    node_four  = Node(4, {}, 'B', 0) # leaf
    node_three = Node(3, {}, 'A', 0) # leaf 

    node_two   = Node(2, {'c': node_five,  'd': node_six}, None, 0)
    node_one   = Node(1, {'e': node_three, 'f': node_four}, None, 0)

    root_node  = Node(0, {'a': node_one, 'b': node_two}, None, 1)

    payoffs = {'A': [0, 0], 'B': [0, 0], 'C': [0, 0], 'D': [0, 0]}
    
    info_sets = [set([1, 2])]
    # info_sets = [set([1]), set([2])]
    return payoffs, root_node, info_sets

def main():
    # payoffs, root, info_sets = generate_centipede_nodes(10, 10, 0)
    # num_players = 2
    # game = Game(root, payoffs, num_players, info_sets)
    # res, subtree = get_subperf_equil(game, game.root)
    # print(res)

    payoffs, root, info_sets = generate_incomplete_game()
    num_players = 1
    game = Game(root, payoffs, num_players, info_sets)
    res, subtree = get_subperf_equil(game, game.root, True)
    print(res, subtree)

if __name__ == "__main__":
    main()

