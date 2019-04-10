"""Player strategies."""

import mcts
import numpy as np
import random

from abc import ABC, abstractmethod


class BasePlayer(ABC):
    """Player of a game."""

    def __init__(self, game, token: str):
        """Initialization of `BasePlayer` abstract class."""
        self.game = game
        self.token = token
        self.score = {"wins": 0, "ties": 0, "losses": 0}

    @abstractmethod
    def act(self, *args, **kwargs):
        """Move logic."""
        pass

    def __str__(self):
        """String representation."""
        return str({self.token: self.score})

    def __repr__(self):
        """String representation."""
        return self.__str__()


class Iterate(BasePlayer):
    """Iterate through the board until a space is free."""

    def act(self, observation):
        """Move logic."""
        for i in range(observation.shape[0]):
            for j in range(observation.shape[1]):
                if observation[i][j] is None:
                    return (i, j)


class Random(BasePlayer):
    """Randomly pick an empty space."""

    def act(self, observation):
        """Move logic."""
        candidates = [
            (i, j) for i, j in zip(*np.where(observation == None))
        ]  # noqa: E711
        return random.choice(candidates)


class MCTS(BasePlayer):
    """Use MCTS algorithm to make a move."""

    def __init__(self, *args, **kwargs):
        """Initialization of `MCTS` abstract class."""
        super().__init__(*args, **kwargs)
        self.mcts = mcts.MCTS(self.game, self.token, budget=0.05)

    def act(self, observation):
        """Move logic."""
        return self.mcts.act(observation)
