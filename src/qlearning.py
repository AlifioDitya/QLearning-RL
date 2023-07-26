'''
This file contains the QLearning demonstration for a simple game with the following ruleset:

- There is a 1-dimensional board (only move left and right) 10 squares in size
- There is a hole at point 0, and an apple at point 9
- Player starts at point 2 and can move left or right
- If the player falls into the hole, the player receives -100 points
- If the player gets the apple, the player receives +100. 
- If the player occupies any other square, the players' points are deducted by 1
- If the player falls into a hole or gets an apple, the player returns to point 3
- Player wins when total points reached +500
- Players loses when total points reached -200
'''

import numpy as np

# Constants
BOARD_SIZE = 10
NUM_ACTIONS = 2
NUM_EPISODES = 5000
MAX_STEPS = 100
STARTING_STATE = 2

# Q-Learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON = 0.1

# Helper function to choose an action based on epsilon-greedy policy
# - With probability epsilon, choose a random action
# - With probability 1 - epsilon, choose the greedy action (the action with the highest Q-value)
# This is to balance exploration and exploitation
def choose_action(q_table, state):
    if np.random.uniform(0, 1) < EPSILON:
        return np.random.randint(NUM_ACTIONS)  # Random action
    else:
        return np.argmax(q_table[state])  # Greedy action

# Run Q-Learning algorithm
def qlearning(verbose=True):
    # Initialize the Q-table, n * m
    # n = number of states (BOARD_SIZE)
    # m = number of actions (2)
    q_table = np.zeros((BOARD_SIZE, NUM_ACTIONS))

    # Loop through episodes
    for episode in range(NUM_EPISODES):
        state = STARTING_STATE  # Start state
        total_reward = 0

        # Loop through steps
        for step in range(MAX_STEPS):
            # Epsilon-greedy policy
            action = choose_action(q_table, state)

            # Apply the selected action
            if action == 0:  # Move left
                next_state = max(state - 1, 0)
            else:  # Move right
                next_state = min(state + 1, BOARD_SIZE - 1)

            # Initialize teleportation variable
            teleport = False

            # Calculate the reward for the action
            if next_state == 0:  # Fell into the hole
                reward = -100
                teleport = True
            elif next_state == BOARD_SIZE - 1:  # Got an apple
                reward = 100
                teleport = True
            else:  # Occupied another point
                reward = -1

            # Update the Q-table using the Bellman equation
            # Q(s, a) = Q(s, a) + alpha * (r + gamma * max(Q(s', a')) - Q(s, a))
            # Where:
            # - s = current state
            # - a = current action
            # - r = reward for the current state and action
            # - s' = next state
            # - a' = next action
            # - alpha = learning rate
            # - gamma = discount factor
            # - Q(s, a) = Q-value for the current state and action
            # - Q(s', a') = Q-value for the next state and action
            # - max(Q(s', a')) = maximum Q-value for all actions from the next state
            q_table[state, action] += LEARNING_RATE * (
                reward + DISCOUNT_FACTOR * np.max(q_table[next_state]) - q_table[state, action]
            )

            total_reward += reward
            state = next_state

            # Check if the game has ended
            if total_reward <= -200 or total_reward >= 500:
                break

            # Teleport if the player falls into a hole or gets an apple
            if teleport:
                state = 3

        # Print learning status every 100 episodes if verbose is True
        if verbose and (episode + 1) % 100 == 0:
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

    # Find the optimal path
    state = 3  # Start state
    path = [state]

    # Keep moving until the player falls into a hole or gets an apple
    while state != 0 and state != BOARD_SIZE - 1:
        action = np.argmax(q_table[state])
        if action == 0:  # Move left
            state = max(state - 1, 0)
        else:  # Move right
            state = min(state + 1, BOARD_SIZE - 1)
        path.append(state)

    if verbose:
        print()

    return q_table, path

def draw_path(path):
    for i in range(len(path)):
        for j in range(len(f"Step {i+1}: ")):
            print(" ", end="")
        for j in range(BOARD_SIZE):
            print("+---", end="")
        print("+")
        print(f"Step {i+1}: ", end="")
        print("|", end="")
        for j in range(BOARD_SIZE):
            if j == path[i]:
                print(" X |", end="")
            elif j == 0:
                print(" H |", end="")
            elif j == BOARD_SIZE - 1:
                print(" A |", end="")
            else:
                print("   |", end="")
        print()
        for j in range(len(f"Step {i+1}: ")):
            print(" ", end="")
        for j in range(BOARD_SIZE):
            print("+---", end="")
        print("+")

# Driver code
if __name__ == "__main__":
    table, path = qlearning()
    path.insert(0, STARTING_STATE)
    print("Final Q-Table:")
    print(table)
    print()
    print("Optimal Path:")
    draw_path(path)
