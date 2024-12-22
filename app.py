import numpy as np
import random
import gym
from gym import spaces
import streamlit as st
import time


# Define a custom environment
class GridEnv(gym.Env):
    def __init__(self):
        super(GridEnv, self).__init__()
        self.grid_size = 5
        self.state = (0, 0)  # Start at the top-left corner
        self.goal = (4, 4)  # Goal position
        self.obstacles = [(2, 2), (3, 3)]

        # Define action space: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.grid_size),
            spaces.Discrete(self.grid_size)
        ))

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        x, y = self.state

        # Perform action
        if action == 0:  # Up
            x = max(x - 1, 0)
        elif action == 1:  # Down
            x = min(x + 1, self.grid_size - 1)
        elif action == 2:  # Left
            y = max(y - 1, 0)
        elif action == 3:  # Right
            y = min(y + 1, self.grid_size - 1)

        self.state = (x, y)

        # Check reward
        if self.state == self.goal:
            return self.state, 100, True, {}
        elif self.state in self.obstacles:
            return self.state, -10, False, {}
        else:
            return self.state, -1, False, {}

    def render(self):
        grid = np.full((self.grid_size, self.grid_size), '.', dtype=str)
        grid[self.goal] = 'G'
        for obs in self.obstacles:
            grid[obs] = 'X'
        grid[self.state] = 'A'
        return grid


# Streamlit interface
st.title("Q-Learning in Grid Environment")
st.write("The grid is displayed step by step, showing the agent (A) moving toward the goal (G) while avoiding obstacles (X).")
st.sidebar.header("Parameters")

# Parameters
alpha = st.sidebar.slider("Learning Rate (α)", 0.01, 1.0, 0.1)
gamma = st.sidebar.slider("Discount Factor (γ)", 0.1, 1.0, 0.9)
epsilon = st.sidebar.slider("Exploration Rate (ε)", 0.01, 1.0, 0.1)
episodes = st.sidebar.slider("Training Episodes", 100, 1000, 500)
grid_size = st.sidebar.slider("Grid Size", 5, 10, 5)

# Create the environment
env = GridEnv()
q_table = np.zeros((grid_size, grid_size, env.action_space.n))

if st.sidebar.button("Train Agent"):
    progress_bar = st.progress(0)
    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                x, y = state
                action = np.argmax(q_table[x, y])  # Exploit

            next_state, reward, done, _ = env.step(action)
            x, y = state
            nx, ny = next_state

            # Update Q-value
            q_table[x, y, action] = q_table[x, y, action] + alpha * (
                    reward + gamma * np.max(q_table[nx, ny]) - q_table[x, y, action]
            )

            state = next_state

        progress_bar.progress((episode + 1) / episodes)

    st.write(q_table)
    st.success("Training Completed!")

# Testing and visualization
if st.sidebar.button("Test Agent"):
    state = env.reset()
    done = False
    st.write("### Agent's Path")
    while not done:
        x, y = state
        action = np.argmax(q_table[x, y])  # Exploit
        state, _, done, _ = env.step(action)
        grid = env.render()
        st.write(grid)
        time.sleep(0.5)
