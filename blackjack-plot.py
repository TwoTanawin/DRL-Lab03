import gym
import torch
from collections import defaultdict
import matplotlib.pyplot as plt

# Define environment
env = gym.make('Blackjack-v1', render_mode="rgb_array")

# Function to run an episode
def run_episode(env, Q, epsilon, n_action):
    """
    Run an episode and perform epsilon-greedy policy
    @param env: OpenAI Gym environment
    @param Q: Q-function
    @param epsilon: the trade-off between exploration and exploitation
    @param n_action: action space
    @return: resulting states, actions, and rewards for the entire episode
    """
    state, info = env.reset()
    rewards = []
    actions = []
    states = []
    is_done = False
    truncated = False
    while not (is_done or truncated):
        probs = torch.ones(n_action) * epsilon / n_action
        best_action = torch.argmax(Q[state]).item()
        probs[best_action] += 1.0 - epsilon
        action = torch.multinomial(probs, 1).item()
        actions.append(action)
        states.append(state)
        state, reward, is_done, truncated, info = env.step(action)
        rewards.append(reward)
    return states, actions, rewards

# Function for Monte Carlo Control with epsilon-greedy
def mc_control_epsilon_greedy(env, gamma, n_episode, epsilon):
    """
    Obtain the optimal policy with on-policy MC control with epsilon_greedy
    @param env: OpenAI Gym environment
    @param gamma: discount factor
    @param n_episode: number of episodes
    @param epsilon: the trade-off between exploration and exploitation
    @return: the optimal Q-function, and the optimal policy
    """
    n_action = env.action_space.n
    G_sum = defaultdict(float)
    N = defaultdict(int)
    Q = defaultdict(lambda: torch.empty(n_action))
    training_rewards = []  # to store training rewards
    for episode in range(n_episode):
        if (episode + 1) % 1000 == 0:
            print("Training episode {}".format(episode+1))
        states_t, actions_t, rewards_t = run_episode(env, Q, epsilon, n_action)
        return_t = 0
        G = {}
        for state_t, action_t, reward_t in zip(states_t[::-1], actions_t[::-1], rewards_t[::-1]):
            return_t = gamma * return_t + reward_t
            G[(state_t, action_t)] = return_t
        for state_action, return_t in G.items():
            state, action = state_action
            G_sum[state_action] += return_t
            N[state_action] += 1
            Q[state][action] = G_sum[state_action] / N[state_action]
        # Calculate total reward for this episode and store it
        training_rewards.append(sum(rewards_t))
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()
    return Q, policy, training_rewards

# Function to simulate an episode using a given policy
def simulate_episode(env, policy):
    state, info = env.reset()
    is_done = False
    truncated = False
    while not (is_done or truncated):
        action = policy[state]
        state, reward, is_done, truncated, info = env.step(action)
    return reward

# Set parameters
gamma = 1
n_episode = 100000
epsilon = 0.1
n_episode_testing = 50000

# Run Monte Carlo Control
optimal_Q, optimal_policy, training_rewards = mc_control_epsilon_greedy(env, gamma, n_episode, epsilon)

# Simulate testing episodes
n_win_optimal = 0
n_lose_optimal = 0
for episode in range(n_episode_testing):
    if (episode + 1) % 1000 == 0:
        print("Testing episode {}".format(episode+1))
    reward = simulate_episode(env, optimal_policy)
    if reward == 1:
        n_win_optimal += 1
    elif reward == -1:
        n_lose_optimal += 1

# Calculate winning and losing probabilities
winning_prob = n_win_optimal / n_episode_testing
losing_prob = n_lose_optimal / n_episode_testing

print('Winning probability under the optimal policy: {}'.format(winning_prob))
print('Losing probability under the optimal policy: {}'.format(losing_prob))

# Plot the results
episodes = range(1000, n_episode + 1, 1000)
plt.figure(figsize=(12, 6))

# Plot Training Rewards
plt.subplot(1, 2, 1)
plt.plot(episodes, training_rewards)
plt.title('Training Progress')
plt.xlabel('Episodes')
plt.ylabel('Total Rewards')

# Plot Testing Probabilities
plt.subplot(1, 2, 2)
plt.bar(['Winning', 'Losing'], [winning_prob, losing_prob], color=['green', 'red'])
plt.title('Testing Results')
plt.ylabel('Probability')

plt.tight_layout()
plt.show()
