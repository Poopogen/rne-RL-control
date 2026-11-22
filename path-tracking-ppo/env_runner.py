from collections import deque

import numpy as np
import torch


# Compute discounted return
def compute_discounted_return(rewards, dones, last_values, last_dones, gamma=0.99):
    returns = np.zeros_like(rewards)
    n_step = len(rewards)

    for t in reversed(range(n_step)):
        if t == n_step - 1:
            returns[t] = rewards[t] + gamma * last_values * (1.0 - last_dones)
        else:
            returns[t] = rewards[t] + gamma * returns[t + 1] * (1.0 - dones[t + 1])

    return returns


# Compute gae 
## Generalized Advantage Estimation (GAE) is a method for estimating the advantage function in reinforcement learning, which helps to reduce the variance of policy gradient estimates while maintaining a low bias. 
## It computes the advantage by combining the temporal difference (TD) error with a discounted sum of future rewards, controlled by a parameter lambda (λ). 
## The GAE formula is given by: A_t = δ_t + (γ * λ) * δ_{t+1}
def compute_gae(rewards, values, dones, last_values, last_dones, gamma=0.99, lamb=0.95):
    # rewards    : (n_step, n_env)
    # values     : (n_step, n_env)
    # dones      : (n_step, n_env)
    # advs       : (n_step, n_env)
    # last_values: (n_env)
    # last_dones : (n_env)
    advs = np.zeros_like(rewards)
    n_step = len(rewards)
    last_gae_lam = 0.0

    for t in reversed(range(n_step)):
        if t == n_step - 1:
            next_nonterminal = 1.0 - last_dones ## next_nonterminal: resets the accumulation at episode boundaries 
            next_values = last_values
        else:
            next_nonterminal = 1.0 - dones[t + 1]
            next_values = values[t + 1]

        ## Lecture unit4 notes: slide 116
        ## delta δ_t: 1-step TD error (the simplest estimate of advantage)
        ## δ_t = what we actually got - what we expected        
        ## δ_t = r_t + γ * V(s_{t+1}) * (1 - d_{t+1}) - V(s_t) 
            ## Discount (γ): future rewards are worth less than immediate rewards. 
            ## (1 - d_{t+1})	: zero-out future if episode ended (done=True)
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t] 
        
        ## Lecture unit4 notes: slide 108-111
        ## advs: smoothed multi-step advantage estimate
        ## Advantage A_t: "How much better is action a_t compared to the average action at state s_t?"
            ## A_t = Q(s_t, a_t) - V(s_t)
            ## A_t > 0 → this action was better than average → increase its probability
            ## A_t < 0 → this action was worse than average → decrease its probability 
        ## A_t = δ_t + (γ * λ) * A_{t+1}
        ## Plain δ_t has high variance (1-step is noisy). Computing a full Monte Carlo return over many steps has high bias if V is wrong.
            ## lamb λ controls the trade-off between bias and variance in the advantage estimates:
            ## λ=0: A_t = δ_t — pure 1-step TD, low variance, high bias
            ## λ=1: Full Monte Carlo return minus baseline, high variance, low bias
            ## λ=0.95: Balance between the two (typical choice)
        advs[t] = last_gae_lam = delta + gamma * lamb * next_nonterminal * last_gae_lam

    return advs + values ## GAE return (G) ≈ advantage + value estimate = A_t + V(s_t) = Q(s_t, a_t) (the target for the value network(critic) update)


# Runner for multiple environment
class EnvRunner:
    # Constructor
    def __init__(self, env, s_dim, a_dim, n_step=5, gamma=0.99, lamb=0.95, device="cpu"):
        self.env = env
        self.n_env = env.n_env
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.n_step = n_step
        self.gamma = gamma
        self.lamb = lamb
        self.device = device

        # last states: (n_env, s_dim)
        # last dones : (n_env)
        self.states = self.env.reset()
        self.dones = np.ones((self.n_env), dtype=bool)

        # Storages (state, action, value, reward, a_logp, done)
        self.mb_states = np.zeros((self.n_step, self.n_env, self.s_dim), dtype=np.float32)
        self.mb_actions = np.zeros((self.n_step, self.n_env, self.a_dim), dtype=np.float32)
        self.mb_values = np.zeros((self.n_step, self.n_env), dtype=np.float32)
        self.mb_rewards = np.zeros((self.n_step, self.n_env), dtype=np.float32)
        self.mb_a_logps = np.zeros((self.n_step, self.n_env), dtype=np.float32)
        self.mb_dones = np.zeros((self.n_step, self.n_env), dtype=bool)

        # Reward & length recorder
        self.total_rewards = np.zeros((self.n_env), dtype=np.float32)
        self.total_len = np.zeros((self.n_env), dtype=np.int32)
        self.reward_buf = deque(maxlen=100)
        self.len_buf = deque(maxlen=100)

    # Run n steps to get a batch
    def run(self, policy_net, value_net):
        # 1. Run n steps
        # -------------------------------------
        for step in range(self.n_step):
            # self.states : (n_env, s_dim)
            # actions     : (n_env, a_dim)
            # self.dones  : (n_env)
            # a_logps     : (n_env)
            # values      : (n_env)
            # rewards     : (n_env)
            # TODO 3: Run a step to collect data
            ## Collects n-step trajectories from parallel environments and returns batches of states, actions, log-probabilities, values, and GAE returns for training.
            ## mb: mini-batch, a batch of data collected from n-step trajectories across multiple parallel environments.
            """
            self.mb_states[step, :]  = ...
            self.mb_dones[step, :]   = ...
            self.mb_actions[step, :] = ...
            self.mb_a_logps[step, :] = ...
            self.mb_values[step, :]  = ...
            self.states, rewards, self.dones, info = self.env.step(actions)
            self.mb_rewards[step, :] = ...
            """
            states_tensor = torch.from_numpy(self.states).float().to(self.device)
            actions, a_logps = policy_net(states_tensor)
            #print("actions:", actions) #torch.Size([n_env,1])
            #print("a_logps:", a_logps) #torch.Size([n_env])
            actions_np = actions.cpu().numpy()
            a_logps_np = a_logps.cpu().numpy()
            values_np = value_net(states_tensor).cpu().numpy()

            self.mb_states[step, :] = self.states
            self.mb_dones[step, :] = self.dones
            self.mb_actions[step, :] = actions_np
            self.mb_a_logps[step, :] = a_logps_np
            self.mb_values[step, :] = values_np
            self.states, rewards, self.dones, info = self.env.step(actions_np)
            self.mb_rewards[step, :] = rewards



        last_values = value_net(torch.from_numpy(self.states).float().to(self.device)).cpu().numpy()
        self.record()

        # 2. Compute returns
        # -------------------------------------
        mb_returns = compute_gae(
            self.mb_rewards, self.mb_values, self.mb_dones, last_values, self.dones, self.gamma, self.lamb
        )

        # mb_states : (n_step*n_env, s_dim)
        # mb_actions: (n_step*n_env) or (n_env*n_step, a_dim)
        # mb_a_logps: (n_step*n_env)
        # mb_values : (n_step*n_env)
        # mb_returns: (n_step*n_env)
        return (
            self.mb_states.reshape(self.n_step * self.n_env, self.s_dim),
            self.mb_actions.reshape(self.n_step * self.n_env, self.a_dim),
            self.mb_a_logps.flatten(),
            self.mb_values.flatten(),
            mb_returns.flatten(),
        )

    # Record return & length
    def record(self):
        for i in range(self.n_step):
            for j in range(self.n_env):
                if self.mb_dones[i, j]:
                    self.reward_buf.append(self.total_rewards[j] + self.mb_rewards[i, j])
                    self.len_buf.append(self.total_len[j] + 1)
                    self.total_rewards[j] = 0
                    self.total_len[j] = 0
                else:
                    self.total_rewards[j] += self.mb_rewards[i, j]
                    self.total_len[j] += 1

    # Get performance
    def get_performance(self):
        if len(self.reward_buf) == 0:
            mean_return = 0
            std_return = 0
        else:
            mean_return = np.mean(self.reward_buf)
            std_return = np.std(self.reward_buf)

        if len(self.len_buf) == 0:
            mean_len = 0
        else:
            mean_len = np.mean(self.len_buf)

        return mean_return, std_return, mean_len
