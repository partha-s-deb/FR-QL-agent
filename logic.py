import numpy as np
import random

class FallacyAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Q-table
        self.q_table = np.zeros((n_states, n_actions))

        # Fallacy System
        self.fp = 0
        self.fp_max = 10
        self.history = []  # to track last moves and path costs

    def choose_action(self, state):
        """Epsilon-greedy action selection influenced by FP system."""
        if random.uniform(0, 1) < self.epsilon:
            # Exploration influenced by FP
            if self.fp > self.fp_max // 2:
                # Force less frequent move exploration
                return self._least_frequent_action(state)
            else:
                return random.randint(0, self.n_actions - 1)
        else:
            # Exploitation
            return np.argmax(self.q_table[state])

    def update_q(self, state, action, reward, next_state):
        """Standard Q-learning update rule."""
        best_next = np.max(self.q_table[next_state])
        self.q_table[state, action] = self.q_table[state, action] + \
            self.alpha * (reward + self.gamma * best_next - self.q_table[state, action])

    def update_fallacy_system(self, action, path_cost):
        """
        Adjust FP using a Christofides-like heuristic.
        - Looks at last 5 steps' path cost.
        - If action is less frequent and reduces cost => FP+1
        - If action is less frequent and increases cost => FP-1
        """
        self.history.append((action, path_cost))
        if len(self.history) > 10:
            self.history.pop(0)

        if len(self.history) >= 6:
            last_five = self.history[-6:-1]
            avg_prev_cost = np.mean([c for _, c in last_five])
            freq = [a for a, _ in last_five].count(action)

            # If less frequent in last 5 moves
            if freq < 2:
                if path_cost < avg_prev_cost:
                    self.fp += 1
                elif path_cost > avg_prev_cost:
                    self.fp -= 1

        # Clamp FP between 0 and fp_max, reset if exceeded
        if self.fp > self.fp_max:
            self.fp = 0
        if self.fp < 0:
            self.fp = 0

    def _least_frequent_action(self, state):
        """Pick the least used action in recent history."""
        last_actions = [a for a, _ in self.history[-5:]]
        action_counts = {a: last_actions.count(a) for a in range(self.n_actions)}
        min_freq = min(action_counts.values())
        least_used = [a for a, count in action_counts.items() if count == min_freq]
        return random.choice(least_used)
