import gym
import numpy as np
import torch

# Define the run_episode function
def run_episode(env, episode_index, weight, show=False):
    state, info = env.reset()
    total_reward = 0
    terminated = False
    truncated = False
    step_starting_index = 0
    while not (terminated or truncated):
        state = torch.from_numpy(state).float()
        action = torch.argmax(torch.matmul(state, weight))
        state, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward
        if show and (terminated or truncated):
            env.render()
        step_starting_index += 1
    if terminated or truncated:
        step_starting_index += 1
    env.close()
    return total_reward

# Define the Monte Carlo Policy Evaluation function
def monte_carlo_policy_evaluation(env, num_episodes, gamma=1.0):
    # Initialize value function
    V = np.zeros(env.observation_space.shape[0])
    # Initialize returns and counts
    returns_sum = np.zeros(env.observation_space.shape[0])
    returns_count = np.zeros(env.observation_space.shape[0])
    weight = np.random.rand(env.observation_space.shape[0], env.action_space.n)
    
    for _ in range(num_episodes):
        episode = []
        state = env.reset()
        done = False
        while not done:
            action = np.argmax(np.matmul(state, weight))
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        # Compute returns and update value function
        G = 0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            if state not in [x[0] for x in episode[0:t]]:
                returns_sum += G * state
                returns_count += 1
                V = returns_sum / returns_count

    return V

# Create the Cartpole environment
env = gym.make('CartPole-v1')

# Number of episodes for Monte Carlo Policy Evaluation
num_episodes = 1000

# Run Monte Carlo Policy Evaluation
V = monte_carlo_policy_evaluation(env, num_episodes)

# Print the value function
print("Value Function:")
print(V)

# Close the environment
env.close()
