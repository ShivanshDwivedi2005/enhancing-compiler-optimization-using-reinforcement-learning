import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Categorical

from env import CompilerEnv, PASSES
from model import Policy

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def masked_logits(logits, state_tensor):
    used_actions = state_tensor[3:3 + len(PASSES)]
    masked = logits.clone()
    masked[used_actions > 0.5] = -1e9
    return masked


def discounted_returns(rewards, gamma):
    returns = []
    running_total = 0.0

    for reward in reversed(rewards):
        running_total = reward + gamma * running_total
        returns.insert(0, running_total)

    returns = torch.tensor(returns, dtype=torch.float32)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def train(num_episodes=5000, gamma=0.97, learning_rate=0.003, entropy_weight=0.02):
    env = CompilerEnv()
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    rewards_history = []

    for episode in range(num_episodes):
        state = torch.tensor(env.reset(), dtype=torch.float32)
        log_probs = []
        rewards = []
        entropies = []

        for _ in range(len(PASSES)):
            logits = policy(state)
            action_logits = masked_logits(logits, state)
            dist = Categorical(logits=action_logits)

            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())

            next_state, reward, done = env.step(action.item())
            rewards.append(reward)
            state = torch.tensor(next_state, dtype=torch.float32)

            if done:
                break

        returns = discounted_returns(rewards, gamma)
        log_probs = torch.stack(log_probs)
        entropy_bonus = torch.stack(entropies).mean()

        loss = -(log_probs * returns).mean() - entropy_weight * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        total_reward = float(sum(rewards))
        rewards_history.append(total_reward)

        if episode % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:]) if rewards_history else total_reward
            print(f"Episode {episode}, Reward: {total_reward:.3f}, Avg(50): {avg_reward:.3f}")

    torch.save(policy.state_dict(), "policy.pth")

    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Progress")
    plt.savefig("training_plot.png")
    plt.show()


if __name__ == "__main__":
    train()
