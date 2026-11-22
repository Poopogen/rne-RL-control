import os
import time

from matplotlib.pyplot import flag
import numpy as np
import torch
from dummy_env import DummyEnv
from stable_baselines3 import PPO
from stable_baselines3.common.utils import safe_mean


class RewardManager:
    def __init__(self):
        self.prev_observation = None
        self.observation = None
        self.frames_since_last_flag = 0 # for no progress penalty
        self.last_checkpoint_index = -1 # for no progress penalty
        self.position_history = [] # for stuck penalty
        self.frames_circling = 0 # for stuck penalty
        self.consecutive_good = 0 # for stuck penalty
        # self.target_dist_history = [] # for stuck penalty


    def update(self, observation):
        self.prev_observation = self.observation
        self.observation = observation

    def reset(self):
        self.prev_observation = None
        self.observation = None
        self.frames_since_last_flag = 0 # for no progress penalty
        self.last_checkpoint_index = -1 # for no progress penalty
        self.position_history = [] # for stuck penalty
        self.frames_circling = 0 # for stuck penalty
        self.consecutive_good = 0 # for stuck penalty
        # self.target_dist_history = [] # for stuck penalty





    def calculate_flag_capture_reward(self):
        """
        [Flag Capture Reward]
        Goal: When a new flag is capture, give a large reward to encourage the agent to move along the correct path.

        Hints:
        1. Compare 'last frame's checkpoint index' (self.prev_observation["last_checkpoint_index"])
           with 'current frame's checkpoint index' (self.observation["last_checkpoint_index"]).
        2. If the current frame's index > the previous frame's index, it means progress was made. Return a positive reward
        3. If there is no change, return 0.0.
        """
        if self.prev_observation is None:
            return 0.0
        prev_idx = self.prev_observation["last_checkpoint_index"]
        curr_idx = self.observation["last_checkpoint_index"]
        if curr_idx > prev_idx:
            return 15.0   # large positive reward for progress
        return 0.0

    def calculate_distance_reward(self):
        """
        [Distance Reward]
        Goal: Guide the agent to constantly move closer to the target point.

        Hints:
        1. Calculate 'distance to target in the previous frame' (prev_distance) and 'distance to target in the current frame' (current_distance).
           (Hint: use numpy.linalg.norm to calculate the vector length of target_position)
        2. Compare the two:
           - If current_distance < prev_distance (getting closer) -> reward
           - If current_distance > prev_distance (getting farther) -> penalize
        3. If the distance hasn't changed, return 0.0.
        """
        if self.prev_observation is None:
            return 0.0
        
        # Skip on flag-capture frame — target_position jumps to next flag,
        # making curr_dist > prev_dist even though the agent succeeded.
        if self.prev_observation["last_checkpoint_index"] != self.observation["last_checkpoint_index"]:
            return 0.0

        #target_position is a relative Vector2 : the vector from the agent to the next checkpoint, not an absolute position.
        prev_dist = np.linalg.norm(self.prev_observation["target_position"]) #np.linalg.norm: Euclidean distance 
        curr_dist = np.linalg.norm(self.observation["target_position"])
        if curr_dist < prev_dist:
            return 0.7  # getting closer
        elif curr_dist > prev_dist:
            return -0.7  # getting farther
        return 0.0

    def calculate_survival_reward(self):
        """
        [Survival Reward]
        Goal: Teach the agent the importance of survival - avoid jumping off the cliff

        Hints:
        Check if agent's health(agent_health) reaches 0
        """
        if self.prev_observation is None:
            return 0.0
        if self.observation["agent_health"] <= 0:
            return -60.0
        return 0.0

    

    def calculate_terrain_reward(self):
        """
        [Terrain Reward]
        Goal: Encourage the agent to avoid dangerous terrains (water and obstacles) before stepping on them, rather than just penalizing after stepping on them.
        """
        if self.prev_observation is None:
            return 0.0

        grid = self.observation["terrain_grid"]
        #print(f'grid: {grid}')

        # Large penalty if agent is currently on water or obstacle
        center_type = grid[2][2]["terrain_type"]
        if center_type == -1:   # water
            return -2.0#-4.0
        elif center_type == 1:    # obstacle
            return -1.0#-1.5
        else:        
            # Count dangerous neighbors and accumulate distance-weighted penalty
            # over the full 5×5 grid (distance 1 and 2 from center)
            type_grid  = [[None]*5 for _ in range(5)]
            score_grid = [[0.0]*5  for _ in range(5)]
            water_count = 0
            obstacle_count = 0
            penalty = 0.0
            for i in range(0, 5):
                for j in range(0, 5):
                    if i == 2 and j == 2:#agent's current cell
                        type_grid[i][j] = 100  # mark center for visualization
                        score_grid[i][j] = 100  # mark center for visualization
                        continue
                    t = grid[i][j]["terrain_type"]  
                    type_grid[i][j] = t # store terrain type for visualization              
                    dist = max(abs(i - 2), abs(j - 2))  # Chebyshev distance: 1 or 2
                    weight = 1.0 if dist == 1 else 0.5  # closer cells penalize more
                    if t == -1:
                        water_count += 1
                        penalty -= 0.1 * weight
                        score_grid[i][j] = 0.1 * weight # store penalty score for visualization
                    elif t == 1:
                        obstacle_count += 1
                        penalty -= 0.05 * weight
                        score_grid[i][j] = 0.05 * weight # store penalty score for visualization

                    
            

            # print("terrain_type:")
            # for row in type_grid:
            #     print([f"{v:2}" for v in row])
            # print("penalty_per_cell:")
            # for row in score_grid:
            #     print([f"{v:+.2f}" for v in row])
            # print(f"total penalty={penalty:.2f}, water_count={water_count}")

            # # # On a narrow path surrounded by water (concave bay / cliff edge),
            # # # the gradient penalty would paralyze the agent — suppress it so
            # # # the agent navigates through instead of spinning in place.
            
    
            # # Suppress only on true narrow bridge/corridor: water/obstacle on BOTH left and right sides, OR both top and bottom
            # '''
            # -1  0  0  0 -1
            # -1  0  0  0 -1
            # -1  0 [A] 0 -1   ← water/obstacle on LEFT/TOP and RIGHT/BOTTOM
            # -1  0  0  0 -1
            # -1  0  0  0 -1
            # '''
            # # Only check rows 1-3 (the central band around the agent at row 2)
            # left_danger   = any(grid[r][0]["terrain_type"] != 0 or grid[r][1]["terrain_type"] != 0 for r in [1, 3])
            # right_danger  = any(grid[r][4]["terrain_type"] != 0 or grid[r][3]["terrain_type"] != 0 for r in [1, 3])
            # top_danger    = any(grid[0][c]["terrain_type"] != 0 or grid[1][c]["terrain_type"] != 0 for c in [1, 3])
            # bottom_danger = any(grid[4][c]["terrain_type"] != 0 or grid[3][c]["terrain_type"] != 0 for c in [1, 3])

            # if (left_danger and right_danger) or (top_danger and bottom_danger):
            #     print("========= Narrow corridor detected")
            #     return penalty * 0.2  # narrow corridor: pass through, slight nudge toward open routes

            return penalty
 

    def calculate_stuck_reward(self):
        '''[Stuck Penalty] 
        Goal: Penalize the agent for being stuck in place or circling (at any speed).
        - Grace period of 10 frames (~1s) allows legitimate curves without penalty.
        - After grace, penalty grows exponentially: -0.4 * 1.05^excess, capped at -2.0.
        '''
        if self.prev_observation is None:
            return 0.0

        pos = self.observation["agent_position"]
        self.position_history.append(np.array([pos[0], pos[2]]))

        # speed = np.linalg.norm(self.observation["agent_velocity"])
        
        # print(f"position_history length: {len(self.position_history)}, current position: ({pos[0]:.2f}, {pos[2]:.2f}), speed: {speed:.4f}")

        # Keep only the most recent 30 positions (~3 seconds at 10 FPS)
        if len(self.position_history) > 30:
            self.position_history = self.position_history[-30:]

        if len(self.position_history) == 30:
            net = np.linalg.norm(self.position_history[-1] - self.position_history[0])
            total = sum(np.linalg.norm(self.position_history[k+1] - self.position_history[k])
                        for k in range(29))
            # total < 0.01  : Physically frozen e.g. stuck in swamp, frame stutter
            # net/total < 0.50: Looping/oscillating (half-circle ≈ 0.637, genuine curve ≥ 0.50)
            print(f"Stuck check: net={net:.4f}, total={total:.4f}, Physically frozen={total < 0.01}, net/total={net/total:.4f}, Looping/spinning={net / total < 0.50}")

            if total < 0.01 or net / total < 0.50:
                self.frames_circling += 1
                self.consecutive_good = 0  # bad frame detected, reset good frame counter
                if self.frames_circling > 10:  # ~1s grace period for legitimate curves
                    excess = self.frames_circling - 10
                    return -min(0.4 * (1.05 ** excess), 1.5)
                return 0.0  # within grace period — no penalty yet
            else:
                self.consecutive_good += 1
                if self.consecutive_good >= 20:  # ~2s of sustained good movement to reset
                    self.frames_circling = 0
                    self.consecutive_good = 0
        return 0.0


    # At 10 FPS: 1 second ≈ use 10 frames/sec
    def calculate_no_progress_penalty(self):
        '''[No Progress Penalty]
        Goal: Penalize the agent for not making any progress (not capturing new flags) for too long, to encourage continuous movement and discourage camping in one spot.'''
        if self.prev_observation is None:
            return 0.0
        curr_idx = self.observation["last_checkpoint_index"]
        if curr_idx > self.last_checkpoint_index:
            self.last_checkpoint_index = curr_idx
            self.frames_since_last_flag = 0
            return 0.0
        self.frames_since_last_flag += 1
        # print(f"Frames since last flag: {self.frames_since_last_flag}")
        
        if self.frames_since_last_flag > 300:  # > 30s: Stronger than on-water (-2.0), approaches obstacle level — does NOT exceed death (-10.0)
            return -1.0
        elif self.frames_since_last_flag > 100: # > 10s: Exceeds on-water penalty (-2.0) → must move
            return -0.5 
        elif self.frames_since_last_flag > 50: # > 5s: early warning, less than nearby water max (-0.8) and stronger than nearby obstacle max (-0.4) → should move soon
            return -0.3
        return 0.0

    def calculate_heading_reward(self):
        '''[Heading Reward] 
        Goal: Reward the agent for heading in the direction of the target.penalizes the spinning pattern that distance reward cannot detect.
        '''
        if self.prev_observation is None:
            return 0.0
        velocity = self.observation["agent_velocity"]
        target = self.observation["target_position"]
        speed = np.linalg.norm(velocity)
        if speed < 0.1:  # nearly stationary — no heading signal
            return 0.0
        # cosine similarity: +1 = heading directly toward target, -1 = heading away
        cos_sim = np.dot(velocity, target) / (speed * np.linalg.norm(target) + 1e-8)
        return float(cos_sim * 0.5)  # range [-0.5, +0.5]

    def calculate_time_penalty(self):   
        '''[Time Penalty]
        Goal: Encourage the agent to complete the task faster by applying a small negative reward every frame.'''
        if self.prev_observation is None:
            return 0.0
        return -0.01 # fires every frame → 60 FPS×120s× −0.01 = -72 total over 120s episode


    def calculate_reward(self):
        """
        [Main Update Loop] (executed every frame)
        Goal: Calculate the total score for this instant

        Hints:
        1. Call the reward each reward components:
           Use self.calculate_...() to get scores for each component.

        2. Sum up rewards:
           total_reward = checkpoint_score + distance_score + survival_score + ...

        Return values:
        - total_reward (float): The total score for this frame.
        """
        # TODO 6: Complete the reward function
        checkpoint_score = self.calculate_flag_capture_reward()
        distance_score = self.calculate_distance_reward()
        survival_score = self.calculate_survival_reward()

        terrain_score = self.calculate_terrain_reward()   
        stuck_score  = self.calculate_stuck_reward()  
        no_progress_score = self.calculate_no_progress_penalty() 
        heading_score = self.calculate_heading_reward()
        time_penalty = self.calculate_time_penalty()   
        print(f"Rewards: checkpoint={checkpoint_score:.2f}, distance={distance_score:.2f}, survival={survival_score:.2f}, terrain={terrain_score:.2f}, stuck={stuck_score:.2f}, no_progress={no_progress_score:.2f}, heading={heading_score:.2f}, time_penalty={time_penalty:.2f}")
        return checkpoint_score + distance_score + survival_score + terrain_score + stuck_score + no_progress_score + heading_score + time_penalty


class MLPlay:
    def __init__(self, observation_structure, action_space_info, *args, **kwargs):
        self.reward_manager = RewardManager()

        self.config = {
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "clip_range": 0.2,
            "gamma": 0.99,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "tensorboard_log": os.path.join(os.path.dirname(__file__), "tensorboard"),
            "policy_kwargs": {"net_arch": [64, 64], "activation_fn": torch.nn.Tanh},
            "device": "cpu",   ##  add this 
        }
        self.dummy_env = DummyEnv(observation_structure, action_space_info)
        self.prev_observation = None
        self.prev_action = None
        self.prev_log_prob = None
        self.prev_value = None
        self.episode_rewards = []
        self.total_steps = 0
        self.episode_count = 1
        self.update_count = 0
        self.start_time = time.strftime("%Y%m%d_%H%M%S")
        self.model_save_dir = os.path.join(os.path.dirname(__file__), "models", self.start_time)
        self.model_path = os.path.join(os.path.dirname(__file__), "model" + ".zip")

        os.makedirs(self.model_save_dir, exist_ok=True)

        self._initialize_model()
        print("PPO initialized in training mode")

    def reset(self):
        if self.episode_rewards:
            total_reward = sum(self.episode_rewards)
            print(
                f"Episode {self.episode_count}: Total Reward = {total_reward:.2f}, Steps = {len(self.episode_rewards)}"
            )
            self.episode_rewards = []

        self._update_policy()

        self.prev_observation = None
        self.prev_action = None
        self.prev_log_prob = None
        self.prev_value = None
        self.episode_count += 1

        self.reward_manager.reset()

    def update(self, raw_observation, done, *args, **kwargs):
        self.reward_manager.update(raw_observation)
        observation = raw_observation["flattened"]

        reward = self.reward_manager.calculate_reward()
        action, log_prob, value = self._predict_action(observation)

        if self.prev_observation is not None:
            self.episode_rewards.append(reward)

            if not self.model.rollout_buffer.full:
                self._add_to_rollout_buffer(
                    obs=self.prev_observation,
                    action=self.prev_action,
                    reward=reward,
                    done=done,
                    value=self.prev_value,
                    log_prob=self.prev_log_prob,
                )
                if self.model.rollout_buffer.full:
                    done_tensor = np.array([done])
                    value_tensor = torch.as_tensor(value).unsqueeze(0) if value.ndim == 0 else torch.as_tensor(value)
                    self.model.rollout_buffer.compute_returns_and_advantage(last_values=value_tensor, dones=done_tensor)

        self.prev_observation = observation
        self.prev_action = action
        self.prev_log_prob = log_prob
        self.prev_value = value
        self.total_steps += 1

        # NOTE: DO NOT MODIFY.
        # Sending additional dummy discrete actions that would not be needed for this assignment
        return action, (0, 0)

    def _initialize_model(self):
        print("Initializing PPO model...")
        if os.path.exists(self.model_path):
            try:
                self.model = PPO.load(self.model_path, env=self.dummy_env, **self.config, verbose=1)
                print(f"Model loaded from {self.model_path}")
            except Exception as e:
                print(f"Error loading model from {self.model_path}: {e}")
                print("Creating new model...")
                self.model = PPO("MlpPolicy", env=self.dummy_env, **self.config, verbose=1)
        else:
            print(f"No pre-trained model found at {self.model_path}. Creating new model...")
            self.model = PPO("MlpPolicy", env=self.dummy_env, **self.config, verbose=1)

        # NOTE: SB3 is not used in the standard way here. Normally model.learn() drives the
        # entire training loop; here, total_timesteps=0 is used only to initialize the
        # TensorBoard logger and internal SB3 state. The actual rollout collection and policy
        # updates are driven manually by mlgame3d's game loop via _add_to_rollout_buffer()
        # and _update_policy(), because mlgame3d controls the environment stepping externally.
        self.model.learn(total_timesteps=0, tb_log_name=f"PPO_{self.start_time}")

    def _save_model(self):
        if self.model is not None:
            self.model.save(self.model_path)
            print(f"Model saved to {self.model_path}")

            update_path = f"{self.model_save_dir}/ppo_model_{self.update_count}.zip"
            self.model.save(update_path)
            print(f"Model saved to {update_path}")

    def _predict_action(self, obs):
        obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(self.model.device)  ## Device Mismatch Error Fixed
        with torch.no_grad():
            action, value, log_prob = self.model.policy(obs_tensor)
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten(), value.cpu().numpy().flatten()

    def _add_to_rollout_buffer(self, obs, action, reward, done, value, log_prob):
        if not self.model.rollout_buffer.full:
            self.model.rollout_buffer.add(
                obs=torch.as_tensor(obs).unsqueeze(0),
                action=torch.as_tensor(action).unsqueeze(0),
                reward=torch.as_tensor([reward]),
                episode_start=torch.as_tensor([done]),
                value=torch.as_tensor(value).unsqueeze(0) if value.ndim == 0 else torch.as_tensor(value),
                log_prob=torch.as_tensor(log_prob).unsqueeze(0) if log_prob.ndim == 0 else torch.as_tensor(log_prob),
            )

    def _update_policy(self):
        if self.model.rollout_buffer.size() == 0 or not self.model.rollout_buffer.full:
            return

        print(f"Updating PPO policy with {self.model.rollout_buffer.size()} experiences...")

        self.model.num_timesteps += self.model.rollout_buffer.size()
        self.model.train()
        self.update_count += 1

        self.model.logger.record("train/mean_reward", safe_mean(self.model.rollout_buffer.rewards))
        self.model.logger.record("param/n_steps", self.model.n_steps)
        self.model.logger.record("param/batch_size", self.model.batch_size)
        self.model.logger.record("param/n_epochs", self.model.n_epochs)
        self.model.logger.record("param/gamma", self.model.gamma)
        self.model.logger.record("param/gae_lambda", self.model.gae_lambda)
        self.model.logger.record("param/ent_coef", self.model.ent_coef)
        self.model.logger.record("param/vf_coef", self.model.vf_coef)
        self.model.logger.record("param/max_grad_norm", self.model.max_grad_norm)
        self.model._dump_logs(self.update_count)

        self.model.rollout_buffer.reset()
        print("PPO policy updated successfully")

        self._save_model()
