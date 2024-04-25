import numpy as np
import mdp

class MultiAgentRandomMDP(mdp.MDP):
    def __init__(self, n_agents, n_states, n_actions, reward_funcs, seed=None):
        """
            n_agents: number of agents
            n_states: number of states
            n_actions: number of actions
            reward_funcs: a list of functions that holds the reward functions for each agent 
            seed: Set to some value to ensure reproducibility
        """
        self.n_agents = n_agents
        self.n_states = n_states
        self.n_actions = n_actions
        self.reward_funcs = reward_funcs
        
        self.rng = np.random.default_rng(seed)
        
        states = list(range(n_states))
        actions = list(range(n_actions))

        P = self._generate_transition_matrix()
        R = self._generate_reward_matrix()

        d0 = np.ones(n_states) / n_states 
        
        super().__init__(states, actions, R, P, d0)
    
    def _generate_transition_matrix(self):
        P = self.rng.uniform(size=(self.n_states, self.n_actions, self.n_states))
        P /= P.sum(axis=-1, keepdims=True)
        return P
    
    def _generate_reward_matrix(self):
        R = np.zeros((self.n_states, self.n_actions, self.n_states, self.n_agents))
        for i in range(self.n_agents):
            R[:, :, :, i] = self.reward_funcs[i](self.n_states, self.n_actions)
        return R
    
    def step(self, state, actions):
        next_state = self.rng.choice(self.states, p=self.P[state, actions[0]])
        rewards = self.R[state, actions[0], next_state, :]
        return next_state, rewards


def reward_functions(n_states, n_actions):
    return np.random.uniform(size=(N_AGENTS, N_ACTIONS, N_STATES))

def main():
    rewards = [reward_functions] * N_AGENTS
    env = MultiAgentRandomMDP(n_agents=N_AGENTS, n_actions=N_ACTIONS, n_states=N_STATES, reward_funcs=rewards, seed=42)

    state = env.states[env.rng.choice(len(env.states), p=env.P0)]
    done = False

    while not done:
        actions = [env.rng.choice(env.actions) for _ in range(N_AGENTS)]  # replace this with your multi-agent RL algorithm
        next_state, rewards = env.step(state, actions)
        # update your multi-agent RL algorithm here based on the experience (state, actions, rewards, next_state)
        state = next_state
        done = env.s_terminal[state]


if __name__ == '__main__':
    N_STATES = 20
    N_AGENTS = 20
    N_ACTIONS = 2 
    
    main()