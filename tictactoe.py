"""TicTacToe."""

import copy
import numpy as np
import random
import xxhash

from typing import Optional


class GameError(Exception):
    """Custom game exception."""

    pass


class FixedTupleSpace(object):
    """Fixed tuple space."""

    def __init__(self, size, depth):
        """Initialization of `FixedTupleSpace` class."""
        self.space = ([i for i in range(depth)],) * size

    def sample(self):
        """Return a sample of the space."""
        return tuple(random.choice(d) for d in self.space)


class TicTacToe(object):
    """Tic Tac Toe game."""

    def __init__(self, size: int = 3):
        """Initialization of `TicTacToe` class."""
        self.size = size
        self.reset()
        self.players = ["X", "O"]
        self.current_player: Optional[str] = None

    def reset(self, custom_current_player=None, custom_start=None):
        """Reset a game, eventually in a custom start."""
        self.current_player = custom_current_player
        self.observation_space = (
            np.copy(custom_start)
            if custom_start is not None
            else np.full((self.size,) * 2, None)
        )
        self.action_space = FixedTupleSpace(2, self.size)
        return self.observation_space

    def clone(self):
        """Clone the game."""
        return copy.deepcopy(self)

    def step(self, player: str, move: tuple):
        """A player move."""
        if player not in self.players:
            raise GameError("Players must either be `X` or `O`.")

        if self.current_player is not None and player != self.current_player:
            raise GameError(f"This is not the turn of {player}.")

        if move is None or len(move) != 2:
            raise GameError(f"{player} made an invalid play.")

        try:
            if self.observation_space[move[0]][move[1]]:
                raise GameError("This movement has already been made.")
        except IndexError:
            raise GameError("Invalid movement.")

        self.observation_space[move[0]][move[1]] = player

        if self.is_winner(player):
            done = True
            reward = 1
            info = player
        elif self.is_finished:
            done = True
            reward = 0
            info = None  # type: ignore
        else:
            done = False
            reward = 0
            info = self._switch_player(player)

        self.current_player = info
        return np.copy(self.observation_space), reward, done, info

    def state(self, observation, player):
        """Return a state ID."""
        h = xxhash.xxh64()
        state = np.copy(observation)
        state[state == player] = 1
        state[state == self._switch_player(player)] = -1
        state[state == None] = 0  # noqa: E711
        h.update(state)
        return h.intdigest()

    def actions(self, observation):
        """Return the valid actions."""
        return [(i, j) for i, j in zip(*np.where(observation == None))]  # noqa: E711

    @property
    def is_finished(self):
        """Check if the game is ended."""
        return None not in self.observation_space

    def is_winner(self, token):
        """Check if there is a winner."""

        def check(array, token):
            """Check if a row is complete."""
            return np.count_nonzero(array == token) == self.size

        for i in range(self.size):
            if check(self.observation_space[i, :], token):
                return True
            if check(self.observation_space[:, i], token):
                return True
        if check(self.observation_space.diagonal(), token):
            return True
        if check(self.observation_space[:, ::-1].diagonal(), token):
            return True
        return False

    def _switch_player(self, player):
        """Swich players."""
        return [p for p in self.players if p != player][0]

    def __str__(self):
        """String representation."""
        return str(self.observation_space)

    def __repr__(self):
        """String representation."""
        return self.__str__()
