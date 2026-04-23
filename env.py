import numpy as np

PASSES = ["inline", "unroll", "vectorize", "dead_code", "register_alloc"]
STATE_SCALE = np.array([4.0, 4.0, 200.0], dtype=np.float32)

def generate_program():
    return {
        "loop_depth": np.random.randint(1, 5),
        "branching": np.random.randint(1, 5),
        "instruction_count": np.random.randint(50, 200)
    }

def raw_pass_gain(program, action):
    score = 0.0

    if action == 0: 
        score += 0.15 if program["instruction_count"] < 120 else -0.2

    elif action == 1:  
        score += 0.2 if program["loop_depth"] > 2 else -0.15

    elif action == 2: 
        score += 0.25 if program["loop_depth"] > 1 else -0.1

    elif action == 3:  
        score += 0.08

    elif action == 4:  
        score += 0.15 if program["branching"] < 3 else -0.1

    return score


def apply_pass(program, action):
    # Allow negative rewards so agent learns to avoid bad actions
    return raw_pass_gain(program, action)


class CompilerEnv:
    def __init__(self):
        self.program = None
        self.steps = 0
        self.used_actions = None

    def reset(self):
        self.program = generate_program()
        self.steps = 0
        self.used_actions = np.zeros(len(PASSES), dtype=np.float32)
        return self.get_state()

    def get_state(self):
        base_state = np.array([
            self.program["loop_depth"],
            self.program["branching"],
            self.program["instruction_count"]
        ], dtype=np.float32)
        normalized_state = base_state / STATE_SCALE
        progress = np.array([self.steps / len(PASSES)], dtype=np.float32)
        return np.concatenate([normalized_state, self.used_actions, progress]).astype(np.float32)

    def step(self, action):
        if self.used_actions[action] == 1.0:
            self.steps += 1
            done = self.steps >= len(PASSES)
            return self.get_state(), 0.0, done

        reward = apply_pass(self.program, action)
        self.used_actions[action] = 1.0

        if action == 1:  # unroll
            self.program["loop_depth"] = max(1, self.program["loop_depth"] - 1)

        elif action == 0:  # inline
            self.program["instruction_count"] = min(220, self.program["instruction_count"] + 10)

        elif action == 2:  # vectorize
            self.program["instruction_count"] = max(40, self.program["instruction_count"] - 5)

        self.steps += 1
        done = self.steps >= len(PASSES) or np.all(self.used_actions == 1.0)

        return self.get_state(), reward, done
