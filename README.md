# Q-Learning
This repository features a Demonstration of Q-Learning — a model-free Reinforcement Learning algorithm — in a simple game.

## Rules of the Game
- There is a 1-dimensional board (only move left and right) 10 squares in size
- There is a hole at point 0, and an apple at point 9
- Player starts at point 2 and can move left or right
- If the player falls into the hole, the player receives -100 points
- If the player gets the apple, the player receives +100. 
- If the player occupies any other square, the players' points are deducted by 1
- If the player falls into a hole or gets an apple, the player returns to point 3
- Player wins when total points reached +500
- Players loses when total points reached -200

## How to Run
Simply head over to `src/qlearning.py` and run the file. The game will run for 5000 episodes, and the Q-table will be printed out at the end.

## Q-Learning
Q-Learning is a model-free reinforcement learning algorithm. It is a table of state-action pairs, where each state-action pair has a value associated with it. The value represents the expected reward for taking that action in that state. The Q-table is updated after each action taken by the agent, using the Bellman equation as follows:
$$Q(s,a) = Q(s,a) + \alpha(r + \gamma \max_{a'}Q(s',a') - Q(s,a))$$
where $s$ is the current state, $a$ is the action taken, $r$ is the reward received, $s'$ is the next state, $\alpha$ is the learning rate, and $\gamma$ is the discount factor. The discount factor is used to weigh the importance of future rewards. The learning rate is used to weigh the importance of new information.  

For a more in-depth explanation, head over to `docs/Questions and Answers.pdf`
