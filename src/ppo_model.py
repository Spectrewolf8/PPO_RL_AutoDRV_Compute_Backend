# src/ppo_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


class Actor(nn.Module):
    """Policy network - outputs action probabilities for discrete actions"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),  # Output logits for discrete actions
        )

    def forward(self, state):
        """Return action logits"""
        return self.network(state)

    def get_distribution(self, state):
        """Get categorical distribution for discrete actions"""
        logits = self.forward(state)
        return Categorical(logits=logits)


class Critic(nn.Module):
    """Value network - estimates state value"""

    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.network(state)


class PPO:
    """
    PPO implementation adapted for discrete action space (AutoDriving environment)

    Key modifications from continuous version:
    - Uses Categorical distribution instead of Normal
    - Action space is discrete (3 actions: left=0, straight=1, right=2)
    - State space is 11-dimensional (5 ray distances + 5 ray hits + 1 speed)
    """

    def __init__(
        self,
        state_dim=11,  # AutoDrivingEnv observation space
        action_dim=3,  # Discrete actions: 0=left, 1=straight, 2=right
        # Hyperparameters - easily changeable
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # GAE parameter
        epsilon=0.2,  # PPO clip parameter
        entropy_coef=0.01,  # Entropy bonus coefficient
        value_coef=0.5,  # Value loss coefficient
        max_grad_norm=0.5,  # Gradient clipping
        hidden_dim=256,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Initialize networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(state_dim, hidden_dim).to(device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Storage for trajectories
        self.reset_memory()

    def reset_memory(self):
        """Clear trajectory storage"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def select_action(self, state, deterministic=False):
        """
        Select action from policy

        Args:
            state: Current state (numpy array of shape (11,))
            deterministic: If True, return most probable action (for testing)

        Returns:
            action: Action to take (int: 0, 1, or 2)
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            dist = self.actor.get_distribution(state)
            if deterministic:
                action = torch.argmax(dist.probs, dim=-1)  # Most probable action
            else:
                action = dist.sample()

        return action.cpu().item()

    def predict(self, observation, deterministic=True):
        """
        Predict action (compatible with PPOController interface)

        Args:
            observation: Current observation (numpy array)
            deterministic: If True, return most probable action

        Returns:
            Tuple of (action, None) to match stable_baselines3 interface
        """
        action = self.select_action(observation, deterministic=deterministic)
        return action, None

    def store_transition(self, state, action, reward, done):
        """
        Store a transition in memory

        Args:
            state: Current state
            action: Action taken (int)
            reward: Reward received
            done: Whether episode ended
        """
        state_tensor = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            # Get log probability and value
            dist = self.actor.get_distribution(state_tensor.unsqueeze(0))
            log_prob = dist.log_prob(torch.tensor([action]).to(self.device))
            value = self.critic(state_tensor.unsqueeze(0))

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob.item())
        self.values.append(value.item())
        self.dones.append(done)

    def compute_gae(self, next_value=0):
        """
        Compute Generalized Advantage Estimation

        Args:
            next_value: Value of the next state (0 if terminal)

        Returns:
            advantages: Computed advantages
            returns: Computed returns (advantages + values)
        """
        advantages = []
        gae = 0

        values = self.values + [next_value]

        # Compute GAE
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(self.values).to(self.device)

        return advantages, returns

    def update(self, epochs=10, batch_size=64):
        """
        Update policy using PPO

        Args:
            epochs: Number of optimization epochs
            batch_size: Mini-batch size for updates

        Returns:
            Dictionary with loss information
        """
        if len(self.states) == 0:
            return {"actor_loss": 0, "critic_loss": 0, "entropy": 0}

        # Convert stored data to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)  # LongTensor for discrete actions
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)

        # Compute advantages and returns
        advantages, returns = self.compute_gae()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Track losses
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        num_updates = 0

        # Optimize for multiple epochs
        for epoch in range(epochs):
            # Create random mini-batches
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start in range(0, len(states), batch_size):
                end = min(start + batch_size, len(states))
                batch_idx = indices[start:end]

                # Get batch
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # Evaluate actions with current policy
                dist = self.actor.get_distribution(batch_states)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Compute ratio (pi_theta / pi_theta_old)
                ratio = torch.exp(log_probs - batch_old_log_probs)

                # Compute surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages

                # Actor loss (PPO objective)
                actor_loss = -torch.min(surr1, surr2).mean()
                actor_loss = actor_loss - self.entropy_coef * entropy

                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Critic loss
                values = self.critic(batch_states).squeeze()
                critic_loss = nn.MSELoss()(values, batch_returns)

                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # Track losses
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                num_updates += 1

        # Clear memory after update
        self.reset_memory()

        return {
            "actor_loss": total_actor_loss / num_updates if num_updates > 0 else 0,
            "critic_loss": total_critic_loss / num_updates if num_updates > 0 else 0,
            "entropy": total_entropy / num_updates if num_updates > 0 else 0,
        }

    def save(self, filepath):
        """Save model"""
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "hyperparameters": {
                    "state_dim": self.state_dim,
                    "action_dim": self.action_dim,
                    "gamma": self.gamma,
                    "gae_lambda": self.gae_lambda,
                    "epsilon": self.epsilon,
                    "entropy_coef": self.entropy_coef,
                    "value_coef": self.value_coef,
                    "max_grad_norm": self.max_grad_norm,
                },
            },
            filepath,
        )
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

        # Load hyperparameters if available
        if "hyperparameters" in checkpoint:
            hyperparams = checkpoint["hyperparameters"]
            self.gamma = hyperparams.get("gamma", self.gamma)
            self.gae_lambda = hyperparams.get("gae_lambda", self.gae_lambda)
            self.epsilon = hyperparams.get("epsilon", self.epsilon)
            self.entropy_coef = hyperparams.get("entropy_coef", self.entropy_coef)
            self.value_coef = hyperparams.get("value_coef", self.value_coef)
            self.max_grad_norm = hyperparams.get("max_grad_norm", self.max_grad_norm)

        print(f"Model loaded from {filepath}")


# =============================================================================
# USAGE EXAMPLE FOR AUTODRIVE ENVIRONMENT
# =============================================================================

if __name__ == "__main__":
    from environment import AutoDrivingEnv

    # Initialize environment
    env = AutoDrivingEnv()

    # Initialize PPO with correct dimensions
    ppo = PPO(
        state_dim=11,  # 5 ray distances + 5 ray hits + 1 speed
        action_dim=3,  # 0=left, 1=straight, 2=right
        # Hyperparameters
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        hidden_dim=256,
    )

    print("PPO Agent initialized for AutoDriving")
    print(f"Device: {ppo.device}")
    print(f"State dimension: {ppo.state_dim}")
    print(f"Action dimension: {ppo.action_dim}")
    print(f"Actor parameters: {sum(p.numel() for p in ppo.actor.parameters())}")
    print(f"Critic parameters: {sum(p.numel() for p in ppo.critic.parameters())}")

    # Example training loop
    print("\n=== Training Example ===")

    num_episodes = 5
    update_frequency = 200  # Update every N steps

    total_steps = 0

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done:
            # Select action
            action = ppo.select_action(state)

            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            ppo.store_transition(state, action, reward, done)

            episode_reward += reward
            total_steps += 1
            step += 1

            # Update policy
            if total_steps % update_frequency == 0:
                losses = ppo.update(epochs=10, batch_size=64)
                print(
                    f"Step {total_steps} - Actor Loss: {losses['actor_loss']:.4f}, "
                    f"Critic Loss: {losses['critic_loss']:.4f}, "
                    f"Entropy: {losses['entropy']:.4f}"
                )

            if done:
                break

            state = next_state

        print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}, Steps: {step + 1}")

    # Save model
    ppo.save("models/ppo_autodrive.pth")
    print("\nModel saved!")

    # Test loading
    print("\n=== Testing Model Loading ===")
    ppo_test = PPO(state_dim=11, action_dim=3)
    ppo_test.load("models/ppo_autodrive.pth")
    print("Model loaded successfully!")

    # Test inference
    test_state, _ = env.reset()
    test_action = ppo_test.select_action(test_state, deterministic=True)
    print(f"Test state: {test_state[:5]}...")
    print(f"Test action: {test_action} ({'left' if test_action == 0 else 'straight' if test_action == 1 else 'right'})")
