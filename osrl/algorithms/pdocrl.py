# reference: https://arxiv.org/abs/2505.17506
import math
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from fsrl.utils import DummyLogger, WandbLogger
from tqdm.auto import trange  # noqa

from osrl.common.net import EnsembleQCritic, SquashedGaussianMLPActor, mlp


class PDOCRL(nn.Module):
    """
    Primal-Dual algorithm for Offline Constrained RL (PDOCRL)

    Implements Algorithm 2 from Hong & Tewari (2025) with deep function
    approximation. Uses an importance weight network (w), an augmented
    Q-network (Q), a softmax-policy actor (pi), and a scalar dual variable
    (lambda) for the cost constraint.

    The Lagrangian is:
        L(w, pi; Q, lam) = (1-gamma) * Q(s0, pi)
            + (1/n) sum_j w(s_j, a_j) *
                [(r - lam * c)(s_j, a_j) + gamma * Q(s'_j, pi) - Q(s_j, a_j)]
            - lam * tau

    Update order per iteration: w -> Q -> pi -> lam -> sync target Q.

    Args:
        state_dim (int): Dimension of state space.
        action_dim (int): Dimension of action space.
        max_action (float): Action space bound.
        a_hidden_sizes (list): Hidden layer sizes for policy network.
        c_hidden_sizes (list): Hidden layer sizes for Q-network.
        w_hidden_sizes (list): Hidden layer sizes for importance weight network.
        gamma (float): Discount factor.
        tau (float): Soft update coefficient for target Q-network.
        num_q (int): Number of Q-networks in ensemble.
        slater_phi (float): Slater condition margin; dual_bound = 1 + 1/phi.
        cost_limit (int): Cost threshold per episode.
        episode_len (int): Maximum episode length.
        device (str): Compute device.
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            max_action: float,
            a_hidden_sizes: list | None = None,
            c_hidden_sizes: list | None = None,
            w_hidden_sizes: list | None = None,
            gamma: float = 0.99,
            tau: float = 0.005,
            num_q: int = 2,
            slater_phi: float = 0.1,
            cost_limit: int = 10,
            episode_len: int = 300,
            device: str = "cpu"):

        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        default_hidden = [256, 256]
        self.a_hidden_sizes = (a_hidden_sizes if a_hidden_sizes is not None
                               else default_hidden)
        self.c_hidden_sizes = (c_hidden_sizes if c_hidden_sizes is not None
                               else default_hidden)
        self.w_hidden_sizes = (w_hidden_sizes if w_hidden_sizes is not None
                               else default_hidden)
        self.gamma = gamma
        self.tau = tau
        self.num_q = num_q
        self.slater_phi = slater_phi
        self.cost_limit = cost_limit
        self.episode_len = episode_len
        self.device = device

        # dual_bound B = 1 + 1/phi, capped at 10 for stability
        self.dual_bound = min(1.0 + 1.0 / slater_phi, 10.0)

        # per-step cost threshold: average cost budget per step
        # cost_limit / episode_len gives the mean per-step cost allowance,
        # which is directly comparable to costs.mean() in dual_update.
        self.cost_threshold = cost_limit / episode_len

        # ---------- networks ----------
        self.actor = SquashedGaussianMLPActor(
            state_dim, action_dim, a_hidden_sizes, nn.ReLU
        ).to(device)

        self.q_net = EnsembleQCritic(
            state_dim, action_dim, c_hidden_sizes, nn.ReLU, num_q=num_q
        ).to(device)
        self.q_net_old = deepcopy(self.q_net)
        self.q_net_old.eval()

        # importance weight network: output >= 0 via Softplus
        self.w_net = mlp(
            [state_dim + action_dim] + list(w_hidden_sizes) + [1],
            nn.ReLU,
            nn.Softplus,
        ).to(device)

        # log-dual variable; exp(log_lambda) in [0, dual_bound]
        self.log_lambda = torch.tensor(0.0, device=device)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _soft_update(self, tgt: nn.Module, src: nn.Module, tau: float) -> None:
        """Soft-update target network parameters."""
        for tgt_p, src_p in zip(tgt.parameters(), src.parameters()):
            tgt_p.data.copy_(tau * src_p.data + (1 - tau) * tgt_p.data)

    def _actor_forward(self, obs, deterministic=False, with_logprob=True):
        """Forward pass through actor; scales output to action bounds."""
        a, logp = self.actor(obs, deterministic, with_logprob)
        return a * self.max_action, logp

    def _lam(self):
        """Current dual variable (scalar)."""
        return self.log_lambda.exp()

    # ------------------------------------------------------------------
    # loss functions
    # ------------------------------------------------------------------

    def w_loss(self, observations, actions, next_observations, rewards, costs, done):
        """
        Importance-weight player: gradient ASCENT on L w.r.t. w.

        w upweights transitions where the current Q under-estimates the
        TD target (positive Bellman residual).
        """
        lam = self._lam().detach()
        aug_reward = rewards - lam * costs

        with torch.no_grad():
            next_actions, _ = self._actor_forward(next_observations, False, True)
            q_next, _ = self.q_net_old.predict(next_observations, next_actions)
            td_target = aug_reward + self.gamma * (1 - done) * q_next

        q_curr, _ = self.q_net.predict(observations, actions)
        bellman_residual = (td_target - q_curr).detach()

        w_in = torch.cat([observations, actions], dim=-1)
        w_vals = self.w_net(w_in).squeeze(-1)  # [B], >= 0
        # clip w to reduce gradient variance (risk mitigation)
        w_vals_clipped = torch.clamp(w_vals, 0.0, 10.0)

        # negate for gradient ASCENT via minimisation optimizer
        loss_w = -(w_vals_clipped * bellman_residual).mean()

        assert not torch.isnan(loss_w), "NaN in w_loss"
        assert not torch.isinf(loss_w), "Inf in w_loss"

        self.w_optim.zero_grad()
        loss_w.backward()
        nn.utils.clip_grad_norm_(self.w_net.parameters(), max_norm=10.0)
        self.w_optim.step()

        stats_w = {
            "loss/w_loss": loss_w.item(),
            "misc/w_mean": w_vals.mean().item(),
            "misc/w_std": w_vals.std().item(),
            "misc/w_max": w_vals.max().item(),
            "misc/w_min": w_vals.min().item(),
        }
        return loss_w, stats_w

    def q_loss(self, observations, actions, next_observations, rewards, costs, done):
        """
        Q-player: importance-weighted Bellman residual minimisation.

        Implements FQE with importance weights derived from w_net.
        """
        lam = self._lam().detach()
        aug_reward = rewards - lam * costs

        with torch.no_grad():
            next_actions, _ = self._actor_forward(next_observations, False, True)
            q_next, _ = self.q_net_old.predict(next_observations, next_actions)
            q_target = aug_reward + self.gamma * (1 - done) * q_next

            w_in = torch.cat([observations, actions], dim=-1)
            w_vals = self.w_net(w_in).squeeze(-1)
            w_vals = torch.clamp(w_vals, 0.0, 10.0)

        _, q_list = self.q_net.predict(observations, actions)
        loss_q = sum([(w_vals * (q - q_target)**2).mean() for q in q_list])

        assert not torch.isnan(loss_q), "NaN in q_loss"
        assert not torch.isinf(loss_q), "Inf in q_loss"

        self.q_optim.zero_grad()
        loss_q.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.q_optim.step()

        stats_q = {"loss/q_loss": loss_q.item()}
        return loss_q, stats_q

    def actor_loss(self, observations, next_observations, actions):
        """
        Policy player: mirror-descent approximation via policy gradient.

        Implements the two-term actor gradient from Eq. (3) of Hong & Tewari (2025):
            ∇π L̂ = (1-γ) ∇π Q(s₀, π(s₀))
                  + γ/n Σⱼ w(sⱼ,aⱼ) ∇π Q(s'ⱼ, π(s'ⱼ))

        Term 1: (1-γ)·Q(s, π(s)) — batch obs used as proxy for initial state dist d₀.
        Term 2: γ·E[w(s,a) · Q(s', π(s'))] — importance-weighted next-state Q.
                w(s,a) uses dataset actions (stop-grad).
        """
        for p in self.q_net.parameters():
            p.requires_grad = False

        # Term 1: (1-γ) · Q(s, π(s))  [batch obs ≈ d₀]
        actions_pi, _ = self._actor_forward(observations, False, True)
        q_s, _ = self.q_net.predict(observations, actions_pi)
        initial_term = (1.0 - self.gamma) * q_s.mean()

        # Term 2: γ · E[w(s,a) · Q(s', π(s'))]  [w is stop-grad]
        with torch.no_grad():
            w_in = torch.cat([observations, actions], dim=-1)
            w_vals = self.w_net(w_in).squeeze(-1)
            w_vals = torch.clamp(w_vals, 0.0, 10.0)

        next_actions_pi, _ = self._actor_forward(next_observations, False, True)
        q_next, _ = self.q_net.predict(next_observations, next_actions_pi)
        next_term = self.gamma * (w_vals * q_next).mean()

        loss_actor = -(initial_term + next_term)

        assert not torch.isnan(loss_actor), "NaN in actor_loss"
        assert not torch.isinf(loss_actor), "Inf in actor_loss"

        self.actor_optim.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optim.step()

        for p in self.q_net.parameters():
            p.requires_grad = True

        stats_actor = {
            "loss/actor_loss": loss_actor.item(),
            "misc/initial_q": q_s.mean().item(),
            "misc/next_q_weighted": q_next.mean().item(),
        }
        return loss_actor, stats_actor

    def dual_update(self, observations, actions, costs):
        """
        Dual variable player: gradient descent on lambda (min over lambda).

        From the Lagrangian L = ... + (1/n)Σ w_j(r - λc)_j - λτ,
        taking ∂L/∂λ = -(1/n)Σ w_j·c_j - τ and doing gradient descent:
            λ ← λ - η·∂L/∂λ = λ + η·((1/n)Σ w_j·c_j - τ)

        The importance-weighted cost E_w[c] = (w·c).mean() is the correct
        constraint signal (reflects the -<λ, τ> term via the -τ part):
          - constraint_violation > 0  →  policy too costly  →  increase lambda
          - constraint_violation < 0  →  policy too safe    →  decrease lambda
        """
        with torch.no_grad():
            w_in = torch.cat([observations, actions], dim=-1)
            w_vals = self.w_net(w_in).squeeze(-1)
            #w_vals = torch.clamp(w_vals, 0.0, 10.0)

        # importance-weighted cost vs per-step threshold (reflects -<λ, τ> term)
        constraint_violation = (w_vals * costs).mean().detach() - self.cost_threshold

        # gradient descent on lambda
        self.log_lambda = (self.log_lambda +
                           self.lambda_lr * constraint_violation)
        self.log_lambda.data.clamp_(
            min=-10.0, max=math.log(self.dual_bound)
        )

        stats_dual = {
            "misc/lambda": self._lam().item(),
            "misc/constraint_violation": constraint_violation.item(),
            "misc/cost_threshold": self.cost_threshold,
            "misc/w_weighted_cost": (w_vals * costs).mean().item(),
        }
        return stats_dual

    # ------------------------------------------------------------------
    # optimizers & sync
    # ------------------------------------------------------------------

    def setup_optimizers(self, actor_lr, q_lr, w_lr, lambda_lr):
        """Create Adam optimizers for actor, Q-net, and w-net."""
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q_optim = torch.optim.Adam(self.q_net.parameters(), lr=q_lr)
        self.w_optim = torch.optim.Adam(self.w_net.parameters(), lr=w_lr)
        self.lambda_lr = lambda_lr

    def sync_weight(self):
        """Soft-update the target Q-network."""
        self._soft_update(self.q_net_old, self.q_net, self.tau)

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    def act(self, obs, deterministic=False, with_logprob=False):
        """Return action (and optionally log-prob) for a single observation."""
        obs = torch.tensor(obs[None, ...], dtype=torch.float32).to(self.device)
        a, logp = self._actor_forward(obs, deterministic, with_logprob)
        a = a.data.numpy() if self.device == "cpu" else a.data.cpu().numpy()
        return np.squeeze(a, axis=0), None


class PDOCRLTrainer:
    """
    Trainer for the PDOCRL algorithm.

    Wraps the PDOCRL model with the standard OSRL training/evaluation loop.

    Args:
        model (PDOCRL): The PDOCRL model instance.
        env (gym.Env): The evaluation environment.
        logger (WandbLogger or DummyLogger): Logger for W&B or stdout.
        actor_lr (float): Learning rate for policy network.
        q_lr (float): Learning rate for Q-network.
        w_lr (float): Learning rate for importance weight network.
        lambda_lr (float): Step size for dual variable update.
        reward_scale (float): Reward scaling factor (applied by dataset).
        cost_scale (float): Cost scaling factor (applied by dataset).
        device (str): Compute device.
    """

    def __init__(
            self,
            model: PDOCRL,
            env: gym.Env,
            logger: WandbLogger | None = None,
            actor_lr: float = 3e-4,
            q_lr: float = 3e-4,
            w_lr: float = 3e-4,
            lambda_lr: float = 1e-3,
            reward_scale: float = 1.0,
            cost_scale: float = 1.0,
            device: str = "cpu"):

        self.model = model
        self.logger = logger if logger is not None else DummyLogger()
        self.env = env
        self.reward_scale = reward_scale
        self.cost_scale = cost_scale
        self.device = device
        self.model.setup_optimizers(actor_lr, q_lr, w_lr, lambda_lr)

    def train_one_step(self, observations, next_observations, actions, rewards, costs,
                       done):
        """
        Execute one training iteration in order: w -> Q -> pi -> lambda ->
        sync target Q.
        """
        # 1. w-player (gradient ascent on importance weights)
        loss_w, stats_w = self.model.w_loss(
            observations, actions, next_observations, rewards, costs, done
        )
        # 2. Q-player (importance-weighted FQE)
        loss_q, stats_q = self.model.q_loss(
            observations, actions, next_observations, rewards, costs, done
        )
        # 3. pi-player (policy gradient)
        loss_actor, stats_actor = self.model.actor_loss(
            observations, next_observations, actions
        )
        # 4. lambda-player (dual gradient descent, -<λ,τ> reflected via importance weights)
        stats_dual = self.model.dual_update(observations, actions, costs)
        # 5. sync target Q-network
        self.model.sync_weight()

        self.logger.store(**stats_w)
        self.logger.store(**stats_q)
        self.logger.store(**stats_actor)
        self.logger.store(**stats_dual)

    def evaluate(self, eval_episodes):
        """Evaluate policy over eval_episodes rollouts."""
        self.model.eval()
        episode_rets, episode_costs, episode_lens = [], [], []
        for _ in trange(eval_episodes, desc="Evaluating...", leave=False):
            epi_ret, epi_len, epi_cost = self.rollout()
            episode_rets.append(epi_ret)
            episode_lens.append(epi_len)
            episode_costs.append(epi_cost)
        self.model.train()
        return (np.mean(episode_rets) / self.reward_scale,
                np.mean(episode_costs) / self.cost_scale,
                np.mean(episode_lens))

    @torch.no_grad()
    def rollout(self):
        """Single-episode rollout; returns (ret, len, cost)."""
        obs, info = self.env.reset()
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        for _ in range(self.model.episode_len):
            act, _ = self.model.act(obs, deterministic=True)
            obs_next, reward, terminated, truncated, info = self.env.step(act)
            cost = info["cost"] * self.cost_scale
            obs = obs_next
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if terminated or truncated:
                break
        return episode_ret, episode_len, episode_cost
