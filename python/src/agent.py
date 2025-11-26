import os
import time
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from python.src.policy_tensor_su import GetWeightSwingUp
from python.src.policy_tensor_b import GetWeightBalance
from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt



# Convert list/ndarray/Tensor to Tensor on the specified device
def weight_to_tensor(w, device):
    if isinstance(w, torch.Tensor):
        return w.detach().clone().to(device=device, dtype=torch.float32)
    else:
        # w can be list or np.ndarray
        return torch.as_tensor(w, dtype=torch.float32, device=device)


class ActorNet(nn.Module):
    def __init__(self, obs_dim, act_dim, max_action):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_action = max_action

    def forward(self, x, deterministic=False):
        """
        x: Tensor of shape (batch_size, obs_dim), float32, on correct device
        deterministic: if True, use mu directly
        """
        h = self.net(x)                               
        mu = self.mean(h)                             
        ls = torch.clamp(self.log_std(h), -20, 2)     
        sigma = ls.exp()                              
        dist = Normal(mu, sigma)
        z = mu if deterministic else dist.rsample()   
        action = torch.tanh(z) * self.max_action      # scale to [-max, max]
        action = torch.clamp(action, -self.max_action, self.max_action)
        return action, mu, ls

class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def forward(self, obs, act):
        """
        obs: Tensor (batch, obs_dim)
        act: Tensor (batch, act_dim)
        return: Tensor (batch, 1)
        """
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def forward(self, obs, act):
        """
        obs: Tensor (batch, obs_dim)
        act: Tensor (batch, act_dim)
        return: Tensor (batch, 1)
        """
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=1_000_000, device='cpu'):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.device = torch.device(device)

    def push(self, obs, act, rew, nxt, done):
        """
        Store as tuple: (obs_np, act_np, rew_float, next_obs_np, done_float)
        Will convert to tensor later during sampling.
        """
        self.buffer.append((
            np.array(obs, dtype=np.float32),
            np.array(act, dtype=np.float32),
            np.float32(rew),
            np.array(nxt, dtype=np.float32),
            np.float32(done)
        ))

    def sample(self, batch_size):
        """
        Returns a tuple of tensors (batch_size, ...) all on the correct device.
        dtype float32 for obs, act, rew, next_obs, done.
        """
        batch = random.sample(self.buffer, batch_size)
        obs_np, act_np, rew_np, nxt_np, done_np = zip(*batch)

        obs_arr  = np.stack(obs_np, axis=0)   # (batch, obs_dim)
        act_arr  = np.stack(act_np, axis=0)   # (batch, act_dim)
        rew_arr  = np.stack(rew_np, axis=0)   # (batch,)
        nxt_arr  = np.stack(nxt_np, axis=0)   # (batch, obs_dim)
        done_arr = np.stack(done_np, axis=0)  # (batch,)

        obs_tensor  = torch.from_numpy(obs_arr).to(self.device)
        act_tensor  = torch.from_numpy(act_arr).to(self.device)
        rew_tensor  = torch.from_numpy(rew_arr).unsqueeze(1).to(self.device)   # (batch, 1)
        nxt_tensor  = torch.from_numpy(nxt_arr).to(self.device)
        done_tensor = torch.from_numpy(done_arr).unsqueeze(1).to(self.device)  # (batch, 1)

        return obs_tensor, act_tensor, rew_tensor, nxt_tensor, done_tensor

    def __len__(self):
        return len(self.buffer)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.buffer, f)
        print(f"[ReplayBuffer] Saved {len(self.buffer)} transitions to {path}")

    def load(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                loaded = pickle.load(f)
            if isinstance(loaded, deque):
                self.buffer = loaded
                print(f"[ReplayBuffer] Loaded {len(self.buffer)} transitions from {path}")
            else:
                print(f"[ReplayBuffer] ERROR: invalid format in {path}")
        else:
            print(f"[ReplayBuffer] Warning: no file at {path} to load")



class SACTrainer:
    def __init__(
        self,
        env,
        gamma=0.99,
        tau=0.005,
        initial_alpha=1.0,
        actor: ActorNet = None,
        q1: QNet = None,
        q2: QNet = None,
        q1_target: QNet = None,
        q2_target: QNet = None,
        lr_actor=1e-3,
        lr_critic=1e-3,
        lr_alpha=1e-3
    ):
        # 1) Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 2) Environment and dimensions
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.episode_rewards = []

        # 3) Networks (already initialized and moved to device)
        assert actor is not None and q1 is not None and q2 is not None \
               and q1_target is not None and q2_target is not None, "Must provide actor/critic!"
        self.actor = actor.to(self.device)
        self.q1 = q1.to(self.device)
        self.q2 = q2.to(self.device)
        self.q1_target = q1_target.to(self.device)
        self.q2_target = q2_target.to(self.device)

        # Copy weights from Q to target
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.q1_target.eval()
        self.q2_target.eval()

        # 4) Optimizers
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=lr_critic)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=lr_critic)

        # 5) Entropy α (auto-tuning)
        self.log_alpha = torch.tensor(np.log(initial_alpha), dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr_alpha)
        self.target_entropy = -float(self.act_dim)

        # 6) Replay buffer
        self.buffer = ReplayBuffer(capacity=1_000_000, device=self.device)

        # 7) Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = 1024

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        # 1) Sample a batch (already tensor on device)
        obs, act, rew, next_obs, done = self.buffer.sample(self.batch_size)

        # 2) Alpha (entropy coefficient)
        alpha = self.log_alpha.exp().detach()

        # --- Critic update ---
        with torch.no_grad():
            next_action, next_mu, next_ls = self.actor(next_obs, deterministic=False)
            next_ls = next_ls.clamp(-20, 2)
            dist_next = Normal(next_mu, next_ls.exp())
            log_pi_next = dist_next.log_prob(next_action).sum(dim=-1, keepdim=True)

            q1_next = self.q1_target(next_obs, next_action)
            q2_next = self.q2_target(next_obs, next_action)
            q_next_min = torch.min(q1_next, q2_next)
            q_target = rew + self.gamma * (1.0 - done) * (q_next_min - alpha * log_pi_next)

        # Q1 loss
        q1_pred = self.q1(obs, act)
        q1_loss = nn.MSELoss()(q1_pred, q_target)
        self.q1_optim.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 10.0)
        self.q1_optim.step()

        # Q2 loss
        q2_pred = self.q2(obs, act)
        q2_loss = nn.MSELoss()(q2_pred, q_target)
        self.q2_optim.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 10.0)
        self.q2_optim.step()

        # --- Actor update ---
        new_action, mu, ls = self.actor(obs, deterministic=False)
        ls = ls.clamp(-20, 2)
        dist = Normal(mu, ls.exp())
        log_pi = dist.log_prob(new_action).sum(dim=-1, keepdim=True)
        q1_new = self.q1(obs, new_action)

        actor_loss = (alpha * log_pi - q1_new).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # --- Alpha update ---
        alpha_loss = - (self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # --- Soft update targets ---
        with torch.no_grad():
            for p, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
                tp.data.mul_(1 - self.tau)
                tp.data.add_(p.data * self.tau)
            for p, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
                tp.data.mul_(1 - self.tau)
                tp.data.add_(p.data * self.tau)

    def save_checkpoint(self, path):
        """
        Save state dicts of actor, critics, optimizers, alpha, and rewards.
        """
        ckpt = {
            'actor_state_dict': self.actor.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'actor_optim_state_dict': self.actor_optim.state_dict(),
            'q1_optim_state_dict': self.q1_optim.state_dict(),
            'q2_optim_state_dict': self.q2_optim.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu().item(),
            'alpha_optim_state_dict': self.alpha_optim.state_dict(),
            'episode_rewards': self.episode_rewards
        }
        torch.save(ckpt, path)
        print(f"[Checkpoint] Saved to {path}")

    def load_checkpoint(self, path):
        """
        Load previously saved checkpoint (networks, optimizers, alpha, rewards).
        """
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor_state_dict'])
        self.q1.load_state_dict(ckpt['q1_state_dict'])
        self.q2.load_state_dict(ckpt['q2_state_dict'])
        self.actor_optim.load_state_dict(ckpt['actor_optim_state_dict'])
        self.q1_optim.load_state_dict(ckpt['q1_optim_state_dict'])
        self.q2_optim.load_state_dict(ckpt['q2_optim_state_dict'])
        self.log_alpha.data.copy_(torch.tensor(ckpt['log_alpha'], device=self.device))
        self.alpha_optim.load_state_dict(ckpt['alpha_optim_state_dict'])
        self.episode_rewards = ckpt.get('episode_rewards', []).copy()
        print(f"[Checkpoint] Loaded from {path}")

    def visualize_training(self):
        """
        Plot episode rewards and moving average.
        """
        if not self.episode_rewards:
            print("No training data to visualize.")
            return

        img_path = 'training_rewards.png'
        if os.path.exists(img_path):
            os.remove(img_path)

        rewards = np.array(self.episode_rewards, dtype=np.float32)
        episodes = np.arange(len(rewards))

        plt.figure(figsize=(10, 5))
        plt.plot(episodes, rewards, label='Episode Reward', alpha=1)

        win = 10
        if len(rewards) >= win:
            mov_avg = np.convolve(rewards, np.ones(win)/win, mode='valid')
            plt.plot(episodes[win-1:], mov_avg, label=f'Moving Avg ({win})')

        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Rewards')
        plt.legend()
        plt.grid(True)
        plt.savefig(img_path)
        plt.show()

    def train(self, episodes=500, max_steps=1000, window_length=10,
              stop_avg_value=50, train_repeat_per_episode=10, save_directory='models/swingup_policy.pth'):
        """
        SAC training: for each episode, collect transitions and call update().
        """
        try:
            for ep in range(1, episodes + 1):
                obs, _ = self.env.reset()
                ep_reward = 0.0

                for step in range(max_steps):
                    # 1) Create observation tensor on device
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

                    # 2) Select action
                    if torch.isnan(obs_tensor).any() or torch.isinf(obs_tensor).any():
                        action_np = np.zeros(self.act_dim, dtype=np.float32)
                    else:
                        alpha_val = obs_tensor[0, 2].item()
                        if abs(alpha_val) > np.deg2rad(160):
                            action_tensor, _, _ = self.actor(obs_tensor, deterministic=False)
                        else:
                            with torch.no_grad():
                                action_tensor, _, _ = self.actor(obs_tensor, deterministic=True)
                        action_np = action_tensor.detach().cpu().numpy().flatten()

                    # 3) Step environment and store transition
                    next_obs, reward, terminated, truncated, _ = self.env.step(action_np)
                    self.update()
                    done = terminated or truncated
                    self.buffer.push(obs, action_np, reward, next_obs, done)

                    obs = next_obs
                    ep_reward += reward
                    if done:
                        break

                # 5) Log and print reward
                self.episode_rewards.append(ep_reward)
                print(f"[Episode {ep:03d}] Total Reward: {ep_reward:.2f}")

                # 6) Save checkpoint every 5 episodes
                if ep % 5 == 0:
                    self.buffer.save("replay_buffer.pkl")
                    self.save_checkpoint(save_directory)

                # 7) Early stopping
                if len(self.episode_rewards) >= window_length:
                    recent_avg = np.mean(self.episode_rewards[-window_length:])
                    if recent_avg >= stop_avg_value:
                        print(f"Training complete at episode {ep}, recent avg: {recent_avg:.2f}")
                        break

        except KeyboardInterrupt:
            print("\n[Interrupted] Training stopped by user.")
            print("[Saving] Replay buffer and checkpoint before exit...")
            self.buffer.save("replay_buffer_b.pkl")
            self.save_checkpoint("models/balance_policy.pth")
            print("[Done] Buffer and checkpoint saved.")


class ActorSU(ActorNet):
    def __init__(self, obs_dim, act_dim, max_action):
        super().__init__(obs_dim, act_dim, max_action)
        self.fc1 = nn.Linear(obs_dim, 8)
        self.fc2 = nn.Linear(8, 8)
        self.net = nn.Sequential(self.fc1, nn.ReLU(), self.fc2, nn.ReLU())
        self.mean = nn.Linear(8, act_dim)
        self.log_std = nn.Linear(8, act_dim)
        self.max_action = max_action

    def load_pretrained_weights(self):
        # Load weights from GetWeightSwingUp()
        w = GetWeightSwingUp().get_all()
        device = next(self.fc1.parameters()).device
        with torch.no_grad():
            # layer1
            self.fc1.weight.copy_(weight_to_tensor(w['W1'], device))
            self.fc1.bias.copy_(weight_to_tensor(w['b1'].squeeze(), device))
            # layer2
            self.fc2.weight.copy_(weight_to_tensor(w['W2'], device))
            self.fc2.bias.copy_(weight_to_tensor(w['b2'].squeeze(), device))
            # mean
            self.mean.weight.copy_(weight_to_tensor(w['W3'].T, device))
            self.mean.bias.copy_(weight_to_tensor(w['b3'].squeeze(), device))
            # log_std
            self.log_std.weight.copy_(weight_to_tensor(w['W4'].T, device))
            self.log_std.bias.copy_(weight_to_tensor(w['b4'].squeeze(), device))

    def load_model(self, model='models/swingup_policy.pth'):
        # Load weights from saved PyTorch checkpoint
        model_actor = torch.load(model)['actor_state_dict']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            # layer1
            self.fc1.weight.copy_(weight_to_tensor(model_actor['fc1.weight'], device))
            self.fc1.bias.copy_(weight_to_tensor(model_actor['fc1.bias'], device))

            # layer2
            self.fc2.weight.copy_(weight_to_tensor(model_actor['fc2.weight'], device))
            self.fc2.bias.copy_(weight_to_tensor(model_actor['fc2.bias'], device))

            # mean
            self.mean.weight.copy_(weight_to_tensor(model_actor['mean.weight'], device))
            self.mean.bias.copy_(weight_to_tensor(model_actor['mean.bias'], device))

            # log_std
            self.log_std.weight.copy_(weight_to_tensor(model_actor['log_std.weight'], device))
            self.log_std.bias.copy_(weight_to_tensor(model_actor['log_std.bias'], device))


class CriticSU(QNet):
    def __init__(self, obs_dim, act_dim):
        super().__init__(obs_dim, act_dim)
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )


class ActorB(ActorNet):
    def __init__(self, obs_dim, act_dim, max_action):
        super().__init__(obs_dim, act_dim, max_action)
        self.fc1 = nn.Linear(obs_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.net = nn.Sequential(
            self.fc1, nn.ReLU(),
            self.fc2, nn.ReLU(),
            self.fc3, nn.ReLU(),
        )
        self.mean    = nn.Linear(32, act_dim)
        self.log_std = nn.Linear(32, act_dim)

    def load_pretrained_weights(self):
        # Load weights from GetWeightBalance()
        w = GetWeightBalance().get_all()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            # layer1
            self.fc1.weight.copy_(weight_to_tensor(w['W1'], device))
            self.fc1.bias.copy_(weight_to_tensor(w['b1'].squeeze(), device))
            # layer2
            self.fc2.weight.copy_(weight_to_tensor(w['W2'], device))
            self.fc2.bias.copy_(weight_to_tensor(w['b2'].squeeze(), device))
            # layer3
            self.fc3.weight.copy_(weight_to_tensor(w['W3'], device))
            self.fc3.bias.copy_(weight_to_tensor(w['b3'].squeeze(), device))
            # mean
            self.mean.weight.copy_(weight_to_tensor(w['W4'].T, device))
            self.mean.bias.copy_(weight_to_tensor(w['b4'].squeeze(), device))
            # log_std
            self.log_std.weight.copy_(weight_to_tensor(w['W5'].T, device))
            self.log_std.bias.copy_(weight_to_tensor(w['b5'].squeeze(), device))
            
    def load_model(self, model='models/balance_policy.pth'):
        # Load weights from a saved checkpoint
        model_actor = torch.load(model)['actor_state_dict']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with torch.no_grad():
            # layer1
            self.fc1.weight.copy_(weight_to_tensor(model_actor['fc1.weight'], device))
            self.fc1.bias.copy_(weight_to_tensor(model_actor['fc1.bias'], device))

            # layer2
            self.fc2.weight.copy_(weight_to_tensor(model_actor['fc2.weight'], device))
            self.fc2.bias.copy_(weight_to_tensor(model_actor['fc2.bias'], device))

            # layer3
            self.fc3.weight.copy_(weight_to_tensor(model_actor['fc3.weight'], device))
            self.fc3.bias.copy_(weight_to_tensor(model_actor['fc3.bias'], device))

            # mean
            self.mean.weight.copy_(weight_to_tensor(model_actor['mean.weight'], device))
            self.mean.bias.copy_(weight_to_tensor(model_actor['mean.bias'], device))

            # log_std
            self.log_std.weight.copy_(weight_to_tensor(model_actor['log_std.weight'], device))
            self.log_std.bias.copy_(weight_to_tensor(model_actor['log_std.bias'], device))


class CriticB(QNet):
    def __init__(self, obs_dim, act_dim):
        super().__init__(obs_dim, act_dim)
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def load_model(self, path='models/balance_policy.pth', critic='q1', device=None):
        """
        path: path to checkpoint file,
        critic: 'q1' or 'q2' to specify which Q-network to load,
        device: torch.device (auto-detect if None)
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(path, map_location=device)
        # Example keys: 'q1_state_dict' or 'q2_state_dict'
        key = f"{critic}_state_dict"
        if key not in ckpt:
            raise KeyError(f"'{key}' not found in checkpoint.")

        model_critic = ckpt[key]
        
        linear_idxs = [i for i, layer in enumerate(self.net) if isinstance(layer, nn.Linear)]
        # should be [0, 2, 4, 6, 8]

        with torch.no_grad():
            for idx in linear_idxs:
                layer = self.net[idx]
                w_key = f"net.{idx}.weight"
                b_key = f"net.{idx}.bias"

                if w_key not in model_critic or b_key not in model_critic:
                    raise KeyError(f"Missing '{w_key}' or '{b_key}' in checkpoint.")

                w = model_critic[w_key]
                b = model_critic[b_key]

                # Convert to correct dtype & device
                w_t = weight_to_tensor(w, device)
                b_t = weight_to_tensor(b, device)

                layer.weight.copy_(w_t)
                layer.bias.copy_(b_t)

        print(f"Loaded {critic} from {path} into QNet.")


class SACBalanceTrainer(SACTrainer):
    def __init__(self,  env, gamma=0.99, tau=0.005, initial_alpha=1, actor = None,
                        q1 = None, q2 = None, q1_target = None, q2_target = None, 
                        lr_actor=0.001, lr_critic=0.001, lr_alpha=0.001,
                        actor_su= None, load_model= None, load_pretrain= True):
        super().__init__(env, gamma, tau, initial_alpha, 
                              actor, q1, q2, q1_target, q2_target,
                              lr_actor, lr_critic, lr_alpha)

        if actor_su is None:
            raise ValueError("You must provide actor_su (swing-up policy) when initializing SACBalanceTrainer.")
        self.actor_su = actor_su.to(self.device)
        self.actor_su.load_pretrained_weights()
        self.actor_su.eval()

        if load_model:
            # Assume actor_su has load_model() method
            self.load_checkpoint(load_model)

        if load_pretrain:
            # Assume actor has load_pretrained_weights()
            self.actor.load_pretrained_weights()
                
    def train(self, episodes=500, max_steps=1000, window_length=10, stop_avg_value=50, train_repeat_per_episode=10):
        """
        Train SAC: collect all transitions in each episode, then perform multiple updates.
        """
        try:
            for ep in range(1, episodes + 1):
                obs, _ = self.env.reset()
                ep_reward = 0.0

                for step in range(max_steps):
                    # 1) Convert observation to tensor on device
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

                    # 2) Select action
                    if torch.isnan(obs_tensor).any() or torch.isinf(obs_tensor).any():
                        action_np = np.zeros(self.act_dim, dtype=np.float32)
                    else:
                        alpha_val = obs_tensor[0, 2].item()
                        if abs(alpha_val) >= np.deg2rad(160):
                            action_tensor, _, _ = self.actor(obs_tensor, deterministic=False)
                        else:
                            with torch.no_grad():
                                action_tensor, _, _ = self.actor_su(obs_tensor, deterministic=True)
                                # action_tensor = torch.tensor(0.0)
                        action_np = action_tensor.detach().cpu().numpy().flatten()

                    # 3) Step environment and store transition
                    next_obs, reward, terminated, truncated, _ = self.env.step(action_np)
                    done = terminated or truncated
                    self.buffer.push(obs, action_np, reward, next_obs, done)

                    obs = next_obs
                    ep_reward += reward
                    if done:
                        break

                # 4) After episode, perform multiple updates if buffer has enough samples
                if len(self.buffer) >= self.batch_size:
                    for _ in range(train_repeat_per_episode):
                        self.update()

                # 5) Log and print reward
                self.episode_rewards.append(ep_reward)
                print(f"[Episode {ep:03d}] Total Reward: {ep_reward:.2f}")

                # 6) Save checkpoint every 5 episodes
                if ep % 5 == 0:
                    self.buffer.save("replay_buffer.pkl")
                    self.save_checkpoint("models/balance_policy.pth")

                # 7) Early stopping
                if len(self.episode_rewards) >= window_length:
                    recent_avg = np.mean(self.episode_rewards[-window_length:])
                    if recent_avg >= stop_avg_value:
                        self.buffer.save("replay_buffer.pkl")
                        self.save_checkpoint("models/balance_policy.pth")
                        print(f"Training complete at episode {ep}, recent avg: {recent_avg:.2f}")
                        break

        except KeyboardInterrupt:
            print("\n[Interrupted] Training stopped by user.")
            print("[Saving] Replay buffer and checkpoint before exiting...")
            self.buffer.save("replay_buffer_b.pkl")
            self.save_checkpoint("models/balance_policy.pth")
            print("[Done] Saved buffer and checkpoint.")


