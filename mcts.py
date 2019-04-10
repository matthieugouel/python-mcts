"""MCTS implementation."""

import datetime
import numpy as np
import pickle
import random


class State(object):
    """Game state."""

    def __init__(self, game, observation, reward, done, player):
        """Initialization of `State` class."""
        self.identifier = game.state(observation, player)
        self.observation = observation
        self.reward = reward
        self.done = done
        self.player = player
        self.next_nodes = [Node(action) for action in game.actions(observation)]

    def __eq__(self, other):
        return self.identifier == other.identifier

    def __hash__(self):
        return self.id

    def __str__(self):
        return f"({self.player}: {self.next_nodes})"

    def __repr__(self):
        return self.__str__()


class Node(object):
    """Monte-Carlo Tree Node."""

    def __init__(self, action, state=None, parent=None):
        """Initialization of `Node` class."""
        self.action = action
        self.state = state
        self.parent = parent

        self.visit_count = 1
        self.accumulated_reward = 0.0

    @property
    def is_evaluated(self):
        """Check if the node is evaluated."""
        return bool(self.state)

    @property
    def is_expanded(self):
        """Check if the node is fully expanded."""
        return all([action.is_evaluated for action in self.state.next_nodes])

    def __eq__(self, other):
        return self.state == other.state

    def __str__(self):
        return f"[({self.visit_count}, {self.accumulated_reward}):{self.state}]"

    def __repr__(self):
        return self.__str__()


class MCTS(object):
    """Monte-Carlo Tree Search Algorithm."""

    def __init__(self, game, player, budget=1):
        """Initialization of `MCTS` class."""
        self.game = game.clone()
        self.player = player
        self.budget = datetime.timedelta(seconds=budget)
        self.tree = {}

    def register(self, node):
        """Register a node in the tree."""
        self.tree[self.game.state(node.state.observation, node.state.player)] = node

    def randomax(self, elements, key=None):
        """Get the max of a list of element given an optional key.

        If there is multiple maximum, then pick one randomly.
        """
        if key is None:
            return max(elements)

        candidates = []
        best_value = None
        for element in elements:
            value = key(element)
            if best_value is None:
                best_value = value
            if value > best_value:
                best_value = value
                candidates = [element]
            if value == best_value:
                candidates.append(element)
        return random.choice(candidates)

    def act(self, observation):
        """Decide of a move based on the observation."""
        root_state = State(self.game, observation, 0, False, self.player)
        root_node = self.tree.get(
            self.game.state(root_state.observation, root_state.player)
        )
        if not root_node:
            root_node = Node(None, state=root_state, parent=None)
            self.register(root_node)

        begin = datetime.datetime.now()
        while datetime.datetime.now() - begin < self.budget:
            self.run(root_node, observation)
        return self.play(root_node)

    def run(self, root_node, observation):
        """Searching procedure."""
        # Selection
        parent_node, selected_node = self.selection(root_node)

        # Expansion
        self.expansion(parent_node, selected_node)

        # Simulation
        winner, reward = self.simulation(selected_node)

        # Back propagation
        self.backpropagation(selected_node, root_node, winner, reward)

    def UCB1(self, node, c=1.41):
        """Upper confidence bound."""
        return (node.accumulated_reward / node.visit_count) + c * np.sqrt(
            np.log(node.parent.visit_count) / node.visit_count
        )

    def selection(self, node):
        """Select a node to expand."""
        while not node.state.done and node.is_expanded:
            node = self.randomax(node.state.next_nodes, key=lambda x: self.UCB1(x))

        if node.state.done:
            return node.parent, node

        return (
            node,
            random.choice(
                [node for node in node.state.next_nodes if not node.is_evaluated]
            ),
        )

    def expansion(self, node, next_node):
        """Expand a new node."""
        if next_node.state is not None:
            return next_node

        self.game.reset(node.state.player, node.state.observation)
        observation, reward, done, player = self.game.step(
            node.state.player, next_node.action
        )
        next_node.state = State(self.game, observation, reward, done, player)
        next_node.parent = node

        self.register(next_node)
        return next_node

    def simulation(self, node):
        """Simulate the game from the specified node."""
        observation = node.state.observation
        reward = node.state.reward
        done = node.state.done
        player = node.state.player

        while not done:

            observation, reward, done, player = self.game.step(
                player, random.choice(self.game.actions(observation))
            )

        return player, reward

    def backpropagation(self, node, root_node, winner, reward):
        """Backpropagate the results."""
        if winner is None:
            winner = random.choice(self.game.players)

        while node != root_node:
            node.visit_count += 1
            if node.parent.state.player == winner:
                node.accumulated_reward += reward
            else:
                node.accumulated_reward -= reward

            node = node.parent

    def play(self, root_node):
        """Select the best play."""
        next_node = self.randomax(
            root_node.state.next_nodes, key=lambda x: x.visit_count
        )
        return next_node.action

    def dumps(self, filename="mcts.ai"):
        """Dumps the tree into a file."""
        pickle.dump(self.tree, open(filename, "wb"))

    def loads(self, filename="mcts.ai"):
        """Loads the tree from a file."""
        self.tree = pickle.load(open(filename, "rb"))

    def __str__(self):
        return str(self.tree)

    def __repr__(self):
        return self.__str__()
