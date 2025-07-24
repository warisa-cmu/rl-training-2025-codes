import sys
from contextlib import closing
from io import StringIO

import gymnasium as gym
import numpy as np

# Define actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class GridworldEnv(gym.Env):
    """
    A 4x4 Grid World environment from Sutton's RL Book.
    Terminal states: top left & bottom right corner.

    Actions: (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave the agent in the current state.
    Reward of -1 at each step until a terminal state is reached.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 15}

    def __init__(self):
        super().__init__()
        self.shape = (4, 4)
        self.nS = np.prod(self.shape)
        self.nA = 4

        self.observation_space = gym.spaces.Discrete(self.nS)
        self.action_space = gym.spaces.Discrete(self.nA)

        # Build transition table
        self.P = {}
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            self.P[s] = {a: [] for a in range(self.nA)}
            self.P[s][UP] = self._transition_prob(position, [-1, 0])
            self.P[s][RIGHT] = self._transition_prob(position, [0, 1])
            self.P[s][DOWN] = self._transition_prob(position, [1, 0])
            self.P[s][LEFT] = self._transition_prob(position, [0, -1])

        self.s = None

    def _limit_coordinates(self, coord):
        coord[0] = min(max(coord[0], 0), self.shape[0] - 1)
        coord[1] = min(max(coord[1], 0), self.shape[1] - 1)
        return coord

    def _transition_prob(self, current, delta):
        current_state = np.ravel_multi_index(tuple(current), self.shape)
        if current_state == 0 or current_state == self.nS - 1:
            return [(1.0, current_state, 0, True)]

        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        done = new_state == 0 or new_state == self.nS - 1
        return [(1.0, new_state, -1, done)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.s = self.np_random.integers(0, self.nS)
        # Optionally, avoid terminal states at reset:
        while self.s == 0 or self.s == self.nS - 1:
            self.s = self.np_random.integers(0, self.nS)
        return self.s, {}

    def step(self, action):
        transitions = self.P[self.s][action]
        prob, next_state, reward, done = transitions[0]  # deterministic
        self.s = next_state
        return self.s, reward, done, False, {}

    def render(self, mode="human"):
        outfile = StringIO() if mode == "ansi" else sys.stdout
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = " x "
            elif s == 0 or s == self.nS - 1:
                output = " T "
            else:
                output = " o "
            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"
            outfile.write(output)
        outfile.write("\n")
        if mode == "ansi":
            with closing(outfile):
                return outfile.getvalue()


# Register environment
gym.register(
    id="GridWorld-v0",
    entry_point="gridworld:GridworldEnv",
    max_episode_steps=300,
)
