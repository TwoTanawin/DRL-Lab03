import gym
import torch
import matplotlib.pyplot as plt


def run_episode(env, policy):
    state, info = env.reset()

    # Modify state for LunarLander-v2
    state = tuple((state * 10).astype(int))
    
    print(state)

    rewards = []
    states = [state]
    is_done = False
    truncated = False
    while not (is_done or truncated):
        state_index = hash(state) % len(policy)  # Wrap hash value within the range of policy tensor
        action = int(policy[state_index].item())
        state, reward, is_done, truncated, info = env.step(action)
        # Modify state for LunarLander-v2
        state = tuple((state * 10).astype(int))
        states.append(state)
        rewards.append(reward)
    env.close()

    states = torch.tensor(states)
    rewards = torch.tensor(rewards)

    return states, rewards


def mc_prediction_first_visit(env, policy, gamma, n_episode):
    n_state = policy.shape[0]
    V = torch.zeros(n_state)
    N = torch.zeros(n_state)
    rewards_history = []

    for episode in range(n_episode):
        states_t, rewards_t = run_episode(env, policy)
        return_t = 0
        first_visit = torch.zeros(n_state)
        G = torch.zeros(n_state, dtype=rewards_t.dtype)
        for state_t, reward_t in zip(reversed(states_t)[1:], reversed(rewards_t)):
            state_index = hash(state_t) % n_state  # Map state to tensor index
            return_t = gamma * return_t + reward_t
            G[state_index] = return_t
            first_visit[state_index] = 1
        for state in range(n_state):
            if first_visit[state] > 0:
                V[state] += G[state]
                N[state] += 1

        # Calculate average reward for this episode and store for plotting
        avg_reward = torch.sum(rewards_t) / len(rewards_t)
        rewards_history.append(avg_reward.item())

    for state in range(n_state):
        if N[state] > 0:
            V[state] = V[state] / N[state]

    return V, rewards_history


gamma = 1
n_episode = 1000
env = gym.make("LunarLander-v2", render_mode="rgb_array")
optimal_policy = torch.tensor([0., 3., 0., 3., 0., 0., 0., 0., 3., 1., 0., 0., 0., 2., 1., 0.])

value, rewards_history = mc_prediction_first_visit(env, optimal_policy, gamma, n_episode)
print('The value function calculated by first-visit MC prediction:\n', value)

# Plot rewards history
plt.figure(figsize=(10, 5))
plt.plot(rewards_history, label='Average Reward per Episode', color='blue')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Training Progress')
plt.legend()
plt.grid(True)
plt.show()

# Plot value function
plt.figure(figsize=(8, 6))
plt.plot(value, marker='o', linestyle='-', color='r')
plt.xlabel('State')
plt.ylabel('Value')
plt.title('Value Function')
plt.grid(True)
plt.show()
