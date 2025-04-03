# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math
from collections import defaultdict


class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved  # Record if the move was valid

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)

def rot90(i, j):
    return 3 - j, i

def reflect(i, j):
    return i, 3 - j

def init_weight():
    return 0

class NTupleApproximator:
    def __init__(self, board_size, patterns):
        self.board_size = board_size
        self.patterns = patterns
        self.weights = [defaultdict(init_weight) for _ in patterns]
        self.symmetry_patterns = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            for syms_ in syms:
                self.symmetry_patterns.append(syms_)

    def rotate90(self, pattern):
        new_pattern = []
        for i, j in pattern:
            new_pattern.append(rot90(i, j))
        return new_pattern

    def reflection(self, pattern):
        new_pattern = []
        for i, j in pattern:
            new_pattern.append(reflect(i, j))
        return new_pattern

    def generate_symmetries(self, pattern):
        p90 = self.rotate90(pattern)
        p180 = self.rotate90(p90)
        p270 = self.rotate90(p180)
        r0 = self.reflection(pattern)
        r90 = self.reflection(p90)
        r180 = self.reflection(p180)
        r270 = self.reflection(p270)
        return [pattern, p90, p180, p270, r0, r90, r180, r270]

    def tile_to_index(self, tile):
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        key = []
        for i, j in coords:
            key.append(self.tile_to_index(board[i, j]))
        return tuple(key)

    def value(self, board):
        value_estimate = 0
        n = 0
        for i in range(len(self.patterns)):
            for j in range(8):
                pattern_coords = self.symmetry_patterns[i * 8 + j]
                feat = self.get_feature(board, pattern_coords)
                if feat in self.weights[i]:
                    value_estimate += self.weights[i][feat]
                    n += 1
        if n > 0:
            value_estimate *= len(self.patterns) * 8 / n
        return value_estimate

    def update(self, board, delta, alpha):
        norm = len(self.patterns) * 8
        for i in range(len(self.patterns)):
            for j in range(8):
                pattern_coords = self.symmetry_patterns[i * 8 + j]
                self.weights[i][self.get_feature(board, pattern_coords)] += alpha * delta / norm

import sys
sys.modules['__main__'].NTupleApproximator = NTupleApproximator
sys.modules['__main__'].init_weight = init_weight

def compute_afterstate(env, a):
    test_env = copy.deepcopy(env)
    s_prev = test_env.score
    if a == 0:
        test_env.move_up()
    elif a == 1:
        test_env.move_down()
    elif a == 2:
        test_env.move_left()
    elif a == 3:
        test_env.move_right()
    return test_env.board, test_env.score - s_prev

def evaluate(env, approximator, a):
    s_, r = compute_afterstate(env, a)
    return r + approximator.value(s_)

class TD_MCTS_Node:
    def __init__(self, parent=None, action=None):
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = [a for a in range(4)]

    def fully_expanded(self):
        return len(self.untried_actions) == 0

class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def select_child(self, node):
        children = []
        temp = []
        for child in node.children.values():
            temp.append(child.total_reward + self.c * np.sqrt(np.log(node.visits) / child.visits))
            children.append(child)
        return children[np.argmax(temp)]

    def rollout(self, action_sequence):
        sim_env = copy.deepcopy(self.env)
        for action in action_sequence:
            if not sim_env.is_move_legal(action):
                return 0
            sim_env.step(action)
        action_values = []
        for a in range(4):
            action_values.append(evaluate(sim_env, self.approximator, a))
        return sim_env.score + max(action_values)

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.total_reward += (1 / node.visits) * (reward - node.total_reward)
            node = node.parent

    def run_simulation(self, root):
        node = root
        action_sequence = []
        while node.fully_expanded():
            node = self.select_child(node)
            action_sequence.append(node.action)
        action = np.random.choice(node.untried_actions)
        node.untried_actions.remove(action)
        new_node = TD_MCTS_Node(node, action)
        node.children[action] = new_node
        node = new_node
        action_sequence.append(action)

        rollout_reward = self.rollout(action_sequence)
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        # Compute the normalized visit count distribution for each child of the root.
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution

with open("last_3.pkl", "rb") as f:
    approximator = pickle.load(f)

env = Game2048Env()
#td_mcts = TD_MCTS(env, approximator, iterations=2000, exploration_constant=500)

def get_action(state, score):
    env.board = state.copy()
    env.score = score
    '''root = TD_MCTS_Node(None, None)
    for _ in range(td_mcts.iterations):
        td_mcts.run_simulation(root)
    best_act, dist = td_mcts.best_action_distribution(root)'''

    action_values = []
    legal_moves = [a for a in range(4) if env.is_move_legal(a)]
    action = 0
    if legal_moves:
        for a in legal_moves:
            action_values.append(evaluate(env, approximator, a))
        action = legal_moves[np.argmax(action_values)]

    print(score, flush=True)
    return action
    
    # You can submit this random agent to evaluate the performance of a purely random strategy.


