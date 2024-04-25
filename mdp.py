# -*- coding: utf-8 -*-
"""
class for Markov decision processes
"""

import numpy as np

def multinomial_sample(n, p):
    """
    draw n random samples of integers with probabilies p
    """
    if len(p.shape) < 2:
        p.shape = (1, p.shape[0])
    p_accum = np.add.accumulate(p, axis=1)
    n_v, n_c = p_accum.shape
    rnd = np.random.rand(n, n_v, 1)
    m = rnd < p_accum.reshape(1, n_v, n_c)

    m2 = np.zeros(m.shape, dtype='bool')
    m2[:, :, 1:] = m[:, :, :-1]
    np.logical_xor(m, m2, out=m)
    ind_mat = np.arange(n_c, dtype='uint8').reshape(1, 1, n_c)
    mask = np.multiply(ind_mat, m, dtype="uint8")
    S = np.add.reduce(mask, 2, dtype='uint8').squeeze()
    return S

class MDP(object):
    """
    Markov Decision Process

    consists of:
        states S:       list or n_s dimensional numpy array of states
        actions A:      list or n_a dimensional numpy array of actions
        reward_function r: S x A x S -> R
                            numpy array of shape (n_s, n_a, n_s)

                            r(s,a,s') assigns a real valued reward to the
                            transition from state s taking action a and going
                            to state s'

        state_transition_kernel P: S x A x S -> R
                            numpy array of shape (n_s, n_a, n_s)
                            p(s,a,s') assign the transition from s to s' by
                            taking action a a probability

                            sum_{s'} p(s,a,s') = 0 if a is not a valid action
                            in state s, otherwise 1
                            if p(s,a,s) = 1 for each a, s is a terminal state

        start distribution P0: S -> R
                            numpy array of shape (n_s,)
                            defines the distribution of initial states
    """

    def __init__(self, states, actions, reward_function,
                 state_transition_kernel,
                 start_distribution, terminal_trans=0):
        self.state_names = states
        self.states = np.arange(len(states))
        self.action_names = actions
        self.actions = np.arange(len(actions))
        self.r = reward_function
        self.Phi = {}

        # start distribution testing
        self.P0 = np.asanyarray(start_distribution)
        assert np.abs(np.sum(self.P0) - 1) < 1e-12
        assert np.all(self.P0 >= 0)
        assert np.all(self.P0 <= 1)
        self.terminal_trans = terminal_trans
        self.dim_S = 1
        self.dim_A = 1
        # transition kernel testing
        self.P = np.asanyarray(state_transition_kernel)
        assert np.all(self.P >= 0)
        assert np.all(self.P <= 1)

        # extract valid actions and terminal state information
        sums_s = np.sum(self.P, axis=2)
        assert np.all(np.bitwise_or(np.abs(sums_s - 1) < 0.0001,
                                    np.abs(sums_s) < 0.0001))
        self.valid_actions = np.abs(sums_s - 1) < 0.0001

        self.s_terminal = np.asarray([np.all(self.P[s, :, s] == 1)
                                      for s in self.states])

    def extract_transitions(self, episode):
        """
        takes an episode (X_0, A_0, X_1, A_1, ..., X_n) of the MDP and
        procudes a list of tuples for each transition containing
         (X_n, A, X_n+1, R)
             X_n: previous state
             X_n+1: next state
             A: action
             R: associated reward
        """
        transitions = []
        for i in range(0, len(episode) - 2, 2):
            s, a, s_n = tuple(episode[i:i + 3])
            transitions.append((s, a, s_n, self.r[s, a, s_n]))

        return transitions

    def stationary_distribution(self, iterations=10000,
                                seed=None, avoid0=False, policy="uniform"):
        """
        computes the stationary distribution by sampling
        """
        cnt = np.zeros(len(self.states), dtype='uint64')
        for s, _, _, _ in self.sample_transition(max_n=iterations,
                                                 policy=policy, seed=seed):
            cnt[s] += 1
        if avoid0 and np.any(cnt == 0):
            cnt += 1
        mu = (cnt).astype("float")
        mu = mu / mu.sum()
        return mu

    def samples_cached(self, policy, n_iter=1000, n_restarts=100,
                       no_next_noise=False, seed=None, verbose=False):
        if seed is not None:
            np.random.seed(seed)
        assert (not no_next_noise)
        assert(seed is not None)
        states = np.ones([n_restarts * n_iter, self.dim_S])

        states_next = np.ones([n_restarts * n_iter, self.dim_S])
        actions = np.ones([n_restarts * n_iter, self.dim_A])
        rewards = np.ones(n_restarts * n_iter)

        restarts = np.zeros(n_restarts * n_iter, dtype="bool")
        k = 0
        while k < n_restarts * n_iter:
            restarts[k] = True
            for s, a, s_n, r in self.sample_transition(
                    n_iter, policy, with_restart=False):
                states[k, :] = s
                states_next[k, :] = s_n
                rewards[k] = r
                actions[k, :] = a

                k += 1
                if k >= n_restarts * n_iter:
                    break
        return states, actions, rewards, states_next, restarts

    def reward_samples(self, policy, n_iter=1000, n_restarts=100, seed=None):
        if seed is not None:
            np.random.seed(seed)
        rewards = np.zeros((len(self.states), n_restarts, n_iter))
        for s0 in self.states:
            for k in range(n_restarts):
                i = 0
                for s, a, s_n, r in self.sample_transition(
                        n_iter, policy, with_restart=False, s_start=s0):
                    rewards[s0, k, i] = r
                    i += 1

        return rewards

    def samples_cached_transitions(self, policy, states, seed=None):
        n = states.shape[0]
        sn = np.zeros_like(states)
        a = np.ones([n, self.dim_A])
        r = np.ones(n)
        for i in range(n):
            a[i] = policy(states[i])
            sn[i] = multinomial_sample(1, self.P[int(states[i]), int(a[i])])
            r[i] = self.r[int(states[i]), int(a[i]), int(sn[i])]
        return a, r, sn

    def samples_featured(self, phi, policy, n_iter=1000, n_restarts=100,
                         no_next_noise=False, seed=1, n_subsample=1):
        assert(seed is not None)
        s, a, r, sn, restarts = self.samples_cached(
            policy, n_iter, n_restarts, no_next_noise, seed)

        n_feat = len(phi(0))
        feats = np.empty([n_restarts * n_iter, n_feat])
        feats_next = np.empty([n_restarts * n_iter, n_feat])

        for k in range(n_iter * n_restarts):

            feats[k, :] = phi(s[k])
            feats_next[k, :] = phi(sn[k])

        return s, a, r, sn, restarts, feats, feats_next

    def synchronous_sweep(self, seed=None, policy="uniform"):
        """
        generate samples from the MDP so that exactly one transition from each
        non-terminal-state is yielded

        Parameters
        -----------
            policy pi: policy python function

            seed: optional seed for the random generator to generate
                deterministic samples

        Returns
        ---------
            transition tuple (X_n, A, X_n+1, R)
        """
        if seed is not None:
            np.random.seed(seed)
        if policy is "uniform":
            policy = self.uniform_policy()

        for s0 in self.states:
            if self.s_terminal[s0]:
                break
            a = policy(s0)
            s1 = multinomial_sample(1, self.P[s0, a])
            r = self.r[s0, a, s1]
            yield (s0, a, s1, r)

    def sample_transition(self, max_n, policy, seed=None,
                          with_restart=True, s_start=None):
        """
        generator that samples from the MDP
        be aware that this chains can be infinitely long
        the chain is restarted if the policy changes

            max_n: maximum number of samples to draw

            policy pi: policy python function

            seed: optional seed for the random generator to generate
                deterministic samples

            with_restart: determines whether sampling with automatic restart:
                is used

            returns a transition tuple (X_n, A, X_n+1, R)
        """

        if seed is not None:
            np.random.seed(seed)

        i = 0
        term = 0
        while i < max_n:
            if s_start is None:
                s0 = multinomial_sample(1, self.P0)
            else:
                s0 = s_start
            while i < max_n:
                if self.s_terminal[s0]:
                    term += 1
                    if term > self.terminal_trans:
                        term = 0
                        break
                a = policy(s0)
                s1 = multinomial_sample(1, self.P[s0, a])
                r = self.r[s0, a, s1]
                yield (s0, a, s1, r)
                i += 1
                s0 = s1
            if not with_restart:
                break

    def policy_P(self, policy="uniform"):
        if policy is "uniform":
            policy = self.uniform_policy()
        T = self.P * policy.tab[:, :, np.newaxis]
        T = np.sum(T, axis=1)
        return T
