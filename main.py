import numpy as np
import mdp
import matplotlib.pyplot as plt


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
        self.n_joint_actions = n_actions**n_agents

        self.rng = np.random.default_rng(seed)

        self.states = list(range(n_states))
        self.actions = list(range(n_actions))

        self.P = self._generate_transition_matrix()
        self.R = self._generate_reward_matrix()
        self.G = self._generate_communication_matrix()
        self.L = self._generate_laplacian()
        K = 5
        self.phi = self._generate_feature_matrix(K)
        theta_dim = 5
        self.features_b = self._generate_feature_vectors_b(theta_dim)
        # print(self.L)
        # eigs = np.linalg.eig(self.L)
        # print(eigs.eigenvalues)

        d0 = np.ones(n_states) / n_states

        super().__init__(self.states, self.actions, self.R, self.P, d0)

    def _generate_transition_matrix(self):
        # we need to extend this matrix to be l arger, for multiple agents
        # here n_actions in the next line should be the number of joiint actions, i.e., 2^N
        # TODO: define joint actions
        P = self.rng.uniform(size=(self.n_states, self.n_joint_actions, self.n_states))
        # figure out how to map each action in the joint action space to a column vector
        # NOTE: I guess this is already done now? I'm not sure what other changes are needed apart
        # from defining the size based on joint actions rather than number of actions
        P /= P.sum(axis=-1, keepdims=True)
        return P

    def _generate_reward_matrix(self):
        R = np.zeros((self.n_states, self.n_actions, self.n_states, self.n_agents))
        for i in range(self.n_agents):
            R[:, :, :, i] = self.reward_funcs[i](
                self.n_states, self.n_actions
            )
        return R

    def _generate_communication_matrix(self):
        """Generate a Laplacian matrix for N agents with connectivity ratio 4/N
        Connectivity ratio is defined as 2*E/[N(N-1)] where N is the number of agents and E is the
        number of edges. For 5 agents, we need 8 edges to get that ratio"""

        # TODO: This is hard-coded for 5 agents. We need to generalize this

        G = np.zeros((self.n_agents, self.n_agents))
        E = 1
        while E != 8:
            idxs = np.random.randint(0, 5, 2)
            while idxs[0] == idxs[1]:
                idxs = np.random.randint(0, 5, 2)
            G[idxs[0], idxs[1]] = np.random.randint(0, 2)
            E = G.sum()
        return G

    def _generate_laplacian(self):
        indegrees = np.diag(np.sum(self.G, axis=1))
        L = indegrees - self.G
        return L

    def step(self, state, joint_action):
        # Convert joint_action to an index in the joint action space
        action_index = sum(a * (self.n_actions**i) for i, a in enumerate(joint_action))

        # next_state = self.rng.choice(self.states, p=self.P[state, actions[0]])
        next_state = self.rng.choice(self.states, p=self.P[state, action_index])
        # connected_agents = np.where(
        #     self.L[np.arange(self.n_agents), joint_action] != 0
        # )[0]

        # updated_joint_action = list(joint_action)
        # for agent in connected_agents:
        #     updated_joint_action[agent] = self.rng.choice(self.actions)

        rewards = np.array(
            [self.reward_funcs[i](state, joint_action[i]) for i in range(self.n_agents)]
        )
        return next_state, rewards

    def _generate_feature_matrix(self, k):

        feature_matrix = np.zeros((self.n_states, 2**self.n_agents, k))
        for state in range(self.n_states):
            # Iterate over each possible joint action

            for joint_action in range(2**self.n_agents):
                # Convert joint action to binary representation
                action_vector = [
                    int(x) for x in bin(joint_action)[2:].zfill(self.n_agents)
                ]
                # print(action_vector, joint_action)

                # Example: Define feature vector based on state and joint action
                feature_vector = np.random.rand(
                    k
                )  # Example: Random feature vector of length k

                # Store feature vector in the feature matrix
                feature_matrix[state, joint_action, :] = feature_vector

        return feature_matrix

    def _generate_feature_vectors_b(self, m):
        feature_matrix_b = np.zeros((self.n_states, self.n_actions, m))
        for state in range(self.n_states):
            for action in range(self.n_actions):
                feature_matrix_b[state, action] = np.random.rand(m)

        return feature_matrix_b

    def evalQ(self, state, joint_action):
        joint_action_idx = int("".join(map(str, joint_action)), 2)
        feature_vector = self.phi[state, joint_action_idx, :]
        return feature_vector

    def evalPolicy(self, state, action, theta):
        feature_a = self.features_b[state, action, :]
        num = np.exp(feature_a @ theta)
        den = 0
        for i in range(2):
            den += np.exp(self.features_b[state, i, :] @ theta)
        policy = num / den
        return policy

    def choose_action(self, state, theta):
        p_1 = self.evalPolicy(state, 1, theta)
        p_0 = self.evalPolicy(state, 0, theta)
        choice = np.random.uniform()
        if choice < p_0:
            return 0
        else:
            return 1


def create_weight_matrix_Ct(L):
    N = L.shape[0]
    Ct = np.zeros((N, N))
    degrees = np.sum(L, axis=1)

    for i in range(N):
        for j in range(N):
            if L[i, j] == 1:
                Ct[i, j] = 1 / (1 + max(degrees[i], degrees[j]))
    for i in range(N):
        Ct[i, i] = 1 - np.sum(Ct[i, :])

    return Ct


def create_reward_functions(n_agents, n_states, n_actions):
    # create an empty list to store the reward functions
    reward_funcs = []

    for agent in range(n_agents):
        base_rewards = np.random.uniform(0, 4, size=(n_states, n_actions))

        def individual_reward_function(state, action, base_reward=base_rewards):
            # Sample around the base reward for given state and action

            base_reward = base_rewards[state - 1, action - 1]
            sampled_reward = np.random.uniform(base_reward - 0.5, base_reward + 0.5)
            return sampled_reward

        reward_funcs.append(individual_reward_function)

    return reward_funcs


def main():
    rewards = create_reward_functions(N_AGENTS, N_STATES, N_ACTIONS)
    env = MultiAgentRandomMDP(
        n_agents=N_AGENTS,
        n_actions=N_ACTIONS,
        n_states=N_STATES,
        reward_funcs=rewards,
        seed=42,
    )

    state = env.states[env.rng.choice(len(env.states), p=env.P0)]
    done = False

    omega_dim = 5
    theta_dim = 5

    # Define initial parameters
    mu_0 = np.ones(N_AGENTS)
    theta_0 = np.ones((N_AGENTS, theta_dim))
    omega_0 = np.ones((N_AGENTS, omega_dim))
    omega_tilde_0 = np.ones((N_AGENTS, omega_dim))

    # Initialize parameters
    mu = mu_0
    omega = omega_0
    theta = theta_0
    omega_tilde = omega_tilde_0
    t_step = 0

    # Initialize variables
    weight_matrix = create_weight_matrix_Ct(env.G)
    td_error = np.zeros(N_AGENTS)
    A = np.zeros(N_AGENTS)
    psi = np.zeros((N_AGENTS, theta_dim))

    # Intial joint_action
    joint_action_initial = [
        env.choose_action(state, theta[ind, :]) for ind in range(N_AGENTS)
    ]

    joint_action = joint_action_initial

    # Logging variables
    average_rewards_log = []
    cumulative_reward = 0
    cumulative_rewards_log = []
    q_vales_log = []
    td_error_log = []

    while not done:

        print(f"[LOG] Iteration {t_step}", end='\r')

        # Calculate step sizes
        beta_omega = 1 / ((t_step + 1) ** 0.65)
        beta_theta = 1 / ((t_step + 1) ** 0.85)

        # joint_action = [
        #     env.rng.choice(env.actions) for _ in range(N_AGENTS)
        # ]  # replace this with your multi-agent RL algorithm

        # Used in the loop to change the joint_action
        # for calculating the value of the advantage function
        joint_action_temp = joint_action.copy()
        next_state, rewards_ = env.step(state, joint_action)
        # update your multi-agent RL algorithm here based on the experience (state, actions, rewards, next_state)

        # Sample new actions here
        next_joint_action = [
            env.choose_action(next_state, theta[ind, :]) for ind in range(N_AGENTS)
        ]

        # Mu update
        mu = (1 - beta_omega) * mu + beta_omega * rewards_

        for i in range(N_AGENTS):
            Q_i_t = env.evalQ(state, joint_action)
            # TD error update
            td_error[i] = (
                rewards_[i]
                - mu[i]
                + omega[i, :] @ env.evalQ(next_state, next_joint_action)
                - omega[i, :].T @ Q_i_t
            )
            # Critic step
            omega_tilde[i, :] = omega[i, :] + beta_omega * td_error[i] * Q_i_t
            value_sum = 0
            for ai in range(2):
                joint_action_temp[i] = ai
                value_sum += (
                    env.evalPolicy(state, ai, theta[i, :])
                    * omega[i, :]
                    @ env.evalQ(state, joint_action_temp)
                )

            A[i] = omega[i, :] @ Q_i_t - value_sum
            psi[i, :] = env.features_b[state, joint_action[i], :] - env.evalPolicy(
                state, joint_action[i], theta[i, :]
            ) * np.sum(env.features_b[state, :, :])
            # Actor step
            theta[i, :] = theta[i, :] + beta_theta * A[i] * psi[i]

            # Consensus step
            for ag_ind in range(N_AGENTS):
                omega[i, :] = weight_matrix[i, ag_ind] * omega_tilde[ag_ind, :]

        state = next_state
        done = env.s_terminal[state]
        t_step += 1

        if t_step == EPOCHS:
            done = True

        average_rewards_log.append(np.mean(rewards_))
        cumulative_reward += np.sum(rewards_)
        cumulative_rewards_log.append(cumulative_reward)
        q_vales_log.append(np.mean(Q_i_t))
        td_error_log.append(np.mean(td_error))

    plt.subplot(221)
    plt.plot(average_rewards_log, label='average rewards')
    plt.legend()

    plt.subplot(222)
    plt.plot(cumulative_rewards_log, label='cumulative rewards')
    plt.legend()

    plt.subplot(223)
    plt.plot(q_vales_log, label='Q values')
    plt.legend()

    plt.subplot(224)
    plt.plot(td_error_log, label='TD error')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    N_STATES = 5
    N_AGENTS = 5
    N_ACTIONS = 2

    EPOCHS = 2000

    main()
