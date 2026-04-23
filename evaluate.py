import torch
from env import CompilerEnv, PASSES
from model import Policy

def masked_logits(logits, state_tensor):
    used_actions = state_tensor[3:3 + len(PASSES)]
    masked = logits.clone()
    masked[used_actions > 0.5] = -1e9
    return masked


def run_policy_episode(policy, env):
    state = torch.tensor(env.get_state(), dtype=torch.float32)
    total_reward = 0.0

    for _ in range(len(PASSES)):
        logits = policy(state)
        action_logits = masked_logits(logits, state)
        action = torch.argmax(action_logits).item()

        next_state, reward, done = env.step(action)
        total_reward += reward
        state = torch.tensor(next_state, dtype=torch.float32)

        if done:
            break

    return total_reward


def heuristic(env):
    # Weaker heuristic: just try actions in order without checking state
    total = 0
    for action in [0, 1, 2, 3, 4]:
        _, reward, done = env.step(action)
        total += reward
        if done:
            break
    return total


def evaluate(num_programs=100):
    policy = Policy()
    policy.load_state_dict(torch.load("policy.pth", map_location="cpu"))
    policy.eval()

    rl_score = 0
    heuristic_score = 0

    for _ in range(num_programs):
        base_env = CompilerEnv()
        initial_state = base_env.reset()
        program_snapshot = dict(base_env.program)

        rl_env = CompilerEnv()
        rl_env.program = dict(program_snapshot)
        rl_env.steps = 0
        rl_env.used_actions = initial_state[3:3 + len(PASSES)].copy()

        heuristic_env = CompilerEnv()
        heuristic_env.program = dict(program_snapshot)
        heuristic_env.steps = 0
        heuristic_env.used_actions = initial_state[3:3 + len(PASSES)].copy()

        rl_score += run_policy_episode(policy, rl_env)
        heuristic_score += heuristic(heuristic_env)

    print("\n=== RESULTS ===")
    print("RL Average Score:", rl_score / num_programs)
    print("Heuristic Score:", heuristic_score / num_programs)


if __name__ == "__main__":
    evaluate()
