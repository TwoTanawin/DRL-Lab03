import torch
import numpy as np
import random

# Define the environment
# For simplicity, let's consider a simple 3x3 grid world
# S: starting point, G: goal, x: obstacle
# S - - 
# - x - 
# - - G 
# Actions: 0 - Up, 1 - Right, 2 - Down, 3 - Left
# Reward: +1 for reaching the goal, -1 for hitting an obstacle, 0 otherwise
# The goal is to find the optimal path from S to G avoiding the obstacle

# Define the environment
env = np.array([['S', '-', '-'],
                ['-', 'x', '-'],
                ['-', '-', 'G']])

# Define the reward matrix
# -1 for hitting an obstacle, 0 otherwise, 1 for reaching the goal
reward_matrix = np.array([[-1, 0, 0],
                           [0, -1, 0],
                           [0, 0, 1]])

# Define Q-network
class QNetwork(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc = torch.nn.Linear(state_size, 128)
        self.fc2 = torch.nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        x = self.fc2(x)
        return x

# Define the parameters
alpha = 0.1  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.1  # exploration rate
num_episodes = 1000

# Initialize Q-network
state_size = 2  # (row, col) of the grid
action_size = 4  # up, right, down, left
q_network = QNetwork(state_size, action_size)
optimizer = torch.optim.Adam(q_network.parameters(), lr=alpha)
criterion = torch.nn.MSELoss()

# Define the epsilon-greedy policy
def epsilon_greedy(state):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)  # explore randomly
    else:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        q_values = q_network(state_tensor)
        return torch.argmax(q_values).item()

# Define the main Q-learning algorithm
def q_learning():
    for episode in range(num_episodes):
        state = [0, 0]  # starting state
        state_tensor = torch.tensor(state, dtype=torch.float32)
        while True:
            action = epsilon_greedy(state)
            next_state = state.copy()
            if action == 0:  # Up
                next_state[0] = max(0, state[0] - 1)
            elif action == 1:  # Right
                next_state[1] = min(2, state[1] + 1)
            elif action == 2:  # Down
                next_state[0] = min(2, state[0] + 1)
            elif action == 3:  # Left
                next_state[1] = max(0, state[1] - 1)
            
            # Calculate target Q-value
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
            q_values_next = q_network(next_state_tensor)
            max_q_next = torch.max(q_values_next)
            target_q = reward_matrix[next_state[0], next_state[1]] + gamma * max_q_next

            # Get the current Q-value
            q_values = q_network(state_tensor)
            q_value = q_values[action]

            # Update Q-value using Bellman equation
            loss = criterion(q_value, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            state_tensor = next_state_tensor
            
            # Check if the episode is done
            if env[state[0], state[1]] == 'G':
                break

# Run the Q-learning algorithm
q_learning()

# Print the learned Q values
print("Learned Q values:")
for param in q_network.parameters():
    print(param.data)
