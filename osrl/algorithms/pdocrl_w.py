# Section 3.1 of the paper
# Saddle-point formulation: min_V max_w L(w, V)
# Policy extracted via w-weighted MLE (Eq. 5).
import math
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from fsrl.utils import DummyLogger, WandbLogger
from tqdm.auto import trange  # noqa

from osrl.common.net import SquashedGaussianMLPActor, mlp


class PDOCRL_W(nn.Module):
    """
    PDOCRL without Q-network: min_V max_w saddle-point on the Lagrangian
    from Section 3.1 of Anonymous et al. (2025), Eq. (4):

        min_V max_w [(1-γ)·E[V(s₀)] + E_D[w(s,a)·e(s,a,s')]]

    where  e(s,a,s') = (r − λ·c) + γ·V(s') − V(s)  [augmented Bellman residual].

    Policy is extracted by w-weighted maximum-likelihood (Eq. 5):
        actor_loss = −E_D[w(s,a) · log π(a|s)]

    Networks
    --------
    w_net      : (s,a) → R+   importance weight (Softplus output)
    v_net      : s → R         state-value function (no output activation)
    v_net_old  : s → R         target V-network for semi-gradient TD (no optimizer)
    actor      : s → π(·|s)   squashed Gaussian policy
    log_λ      : scalar        dual variable for cost constraint

    Args:
        state_dim (int): State space dimension.
        action_dim (int): Action space dimension.
        max_action (float): Action bound (scales actor output).
        a_hidden_sizes (list): Hidden sizes for actor MLP.
        v_hidden_sizes (list): Hidden sizes for V-network MLP.
        w_hidden_sizes (list): Hidden sizes for importance-weight MLP.
        gamma (float): Discount factor.
        tau (float): Soft update coefficient for target V-network.
        slater_phi (float): Slater margin; dual_bound = 1 + 1/phi.
        cost_limit (int): Per-episode cost budget.
        episode_len (int): Maximum episode length.
        init_lambda (float): Initial dual variable λ.
        device (str): Compute device.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        a_hidden_sizes: list | None = None,
        v_hidden_sizes: list | None = None,
        w_hidden_sizes: list | None = None,
        gamma: float = 0.99,
        tau: float = 0.005,
        slater_phi: float = 0.1,
        cost_limit: int = 10,
        episode_len: int = 300,
        init_lambda: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        default_hidden = [256, 256]
        self.a_hidden_sizes = a_hidden_sizes or default_hidden
        self.v_hidden_sizes = v_hidden_sizes or default_hidden
        self.w_hidden_sizes = w_hidden_sizes or default_hidden
        self.gamma = gamma
        self.tau = tau
        self.slater_phi = slater_phi
        self.cost_limit = cost_limit
        self.episode_len = episode_len
        self.device = device

        self.dual_bound = min(1.0 + 1.0 / slater_phi, 10.0)
        self.cost_threshold = cost_limit / episode_len

        # ---------- networks ----------
        self.actor = SquashedGaussianMLPActor(
            state_dim, action_dim, self.a_hidden_sizes, nn.ReLU
        ).to(device)

        # V(s): state-only value function, unbounded output
        self.v_net = mlp(
            [state_dim] + list(self.v_hidden_sizes) + [1], nn.ReLU
        ).to(device)
        self.v_net_old = deepcopy(self.v_net)
        self.v_net_old.eval()

        # w(s,a) ≥ 0 via Softplus
        self.w_net = mlp(
            [state_dim + action_dim] + list(self.w_hidden_sizes) + [1],
            nn.ReLU,
            nn.Softplus,
        ).to(device)

        # log dual variable; exp(log_λ) ∈ [0, dual_bound]
        self.log_lambda = torch.tensor(math.log(init_lambda), device=device)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _lam(self):
        return self.log_lambda.exp()

    def _actor_forward(self, obs, deterministic=False, with_logprob=True):
        a, logp = self.actor(obs, deterministic, with_logprob)
        return a * self.max_action, logp

    def _soft_update(self, tgt: nn.Module, src: nn.Module, tau: float) -> None:
        for tgt_p, src_p in zip(tgt.parameters(), src.parameters()):
            tgt_p.data.copy_(tau * src_p.data + (1 - tau) * tgt_p.data)

    def sync_weight(self):
        self._soft_update(self.v_net_old, self.v_net, self.tau)

    def _bellman_residual(self, obs, next_obs, aug_reward, done):
        """e = aug_r + γ·V_old(s') − V(s).  Semi-gradient: target uses v_net_old."""
        v_s = self.v_net(obs).squeeze(-1)                    # [B], gradient O
        with torch.no_grad():
            v_s_next = self.v_net_old(next_obs).squeeze(-1)  # [B], gradient X
        e = aug_reward + self.gamma * (1.0 - done) * v_s_next - v_s
        return e, v_s, v_s_next

    # ------------------------------------------------------------------
    # loss functions
    # ------------------------------------------------------------------

    def w_loss(self, observations, actions, next_observations, rewards, costs, done):
        """
        w-player: gradient ASCENT to maximise
            E_D[w·e]
        V is fixed (stop-grad).
        """
        lam = self._lam().detach()
        aug_reward = (rewards - lam * costs).detach()

        with torch.no_grad():
            e, _, _ = self._bellman_residual(observations, next_observations,
                                             aug_reward, done)

        w_in = torch.cat([observations, actions], dim=-1)
        w = self.w_net(w_in).squeeze(-1)    # [B], ≥ 0

        # negate for gradient ASCENT via minimising optimiser
        loss_w = -(w * e).mean()

        assert not torch.isnan(loss_w), "NaN in w_loss"
        assert not torch.isinf(loss_w), "Inf in w_loss"

        self.w_optim.zero_grad()
        loss_w.backward()
        nn.utils.clip_grad_norm_(self.w_net.parameters(), max_norm=10.0)
        self.w_optim.step()

        stats_w = {
            "loss/w_loss": loss_w.item(),
            "misc/w_mean": w.detach().mean().item(),
            "misc/w_max": w.detach().max().item(),
            "misc/bellman_residual": e.mean().item(),
        }
        return loss_w, stats_w

    def v_loss(self, observations, actions, next_observations, rewards, costs, done):
        """
        V-player: gradient DESCENT to minimise
            (1−γ)·E[V(s)] + E_D[w·e]
        Batch observations serve as a proxy for the initial-state distribution s₀.
        w is fixed (stop-grad).
        """
        lam = self._lam().detach()
        aug_reward = (rewards - lam * costs).detach()

        with torch.no_grad():
            w_in = torch.cat([observations, actions], dim=-1)
            w = self.w_net(w_in).squeeze(-1)   # [B], stop-grad

        # gradients flow through V(s) only; V(s') comes from v_net_old (semi-gradient)
        e, v_s, _ = self._bellman_residual(observations, next_observations,
                                           aug_reward, done)

        init_term = (1.0 - self.gamma) * v_s.mean()
        loss_v = init_term + (w * e).mean()

        assert not torch.isnan(loss_v), "NaN in v_loss"
        assert not torch.isinf(loss_v), "Inf in v_loss"

        self.v_optim.zero_grad()
        loss_v.backward()
        nn.utils.clip_grad_norm_(self.v_net.parameters(), max_norm=10.0)
        self.v_optim.step()

        stats_v = {
            "loss/v_loss": loss_v.item(),
            "misc/v_mean": v_s.detach().mean().item(),
        }
        return loss_v, stats_v

    def actor_loss(self, observations, actions):
        """
        Policy extraction via w-weighted MLE (Eq. 5 of Anonymous et al. 2025):
            actor_loss = −E_D[w(s,a) · log π(a|s)]
        No Q-gradient; purely supervised by dataset actions weighted by w.
        """
        _, _, dist = self.actor.forward(observations, False, True, True)

        with torch.no_grad():
            w_in = torch.cat([observations, actions], dim=-1)
            w = self.w_net(w_in).squeeze(-1)
            w = torch.clamp(w, 0.0, 10.0)

        log_prob = dist.log_prob(actions).sum(axis=-1)  # [B]
        loss_actor = -(w * log_prob).mean()

        assert not torch.isnan(loss_actor), "NaN in actor_loss"
        assert not torch.isinf(loss_actor), "Inf in actor_loss"

        self.actor_optim.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optim.step()

        stats_actor = {
            "loss/actor_loss": loss_actor.item(),
            "misc/w_weighted_logprob": (-loss_actor).item(),
            "misc/log_prob_mean": log_prob.detach().mean().item(),
        }
        return loss_actor, stats_actor

    def dual_update(self, observations, actions, costs):
        """
        λ-player: gradient ASCENT on λ·(E_D[w·c] − τ).
        Uses importance-weighted empirical cost as a proxy for E_π[c].
        """

        weighted_cost = costs.mean().detach()
        constraint_violation = weighted_cost - self.cost_threshold

        self.log_lambda = self.log_lambda + self.lambda_lr * constraint_violation
        self.log_lambda.data.clamp_(min=-10.0, max=math.log(self.dual_bound))

        stats_dual = {
            "misc/lambda": self._lam().item(),
            "misc/weighted_cost": weighted_cost.item(),
            "misc/constraint_violation": constraint_violation.item(),
        }
        return stats_dual

    # ------------------------------------------------------------------
    # optimizers & inference
    # ------------------------------------------------------------------

    def setup_optimizers(self, actor_lr, v_lr, w_lr, lambda_lr):
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.v_optim = torch.optim.Adam(self.v_net.parameters(), lr=v_lr)
        self.w_optim = torch.optim.Adam(self.w_net.parameters(), lr=w_lr)
        self.lambda_lr = lambda_lr

    def act(self, obs, deterministic=False, with_logprob=False):
        obs = torch.tensor(obs[None, ...], dtype=torch.float32).to(self.device)
        a, _ = self._actor_forward(obs, deterministic, with_logprob)
        a = a.data.numpy() if self.device == "cpu" else a.data.cpu().numpy()
        return np.squeeze(a, axis=0), None


class PDOCRL_WTrainer:
    """
    Trainer for PDOCRL_W.

    Update order per step: w → V → π → λ → target V sync.

    Args:
        model (PDOCRL_W): The PDOCRL_W model instance.
        env (gym.Env): Evaluation environment.
        logger (WandbLogger or DummyLogger): Logger.
        actor_lr (float): Learning rate for policy network.
        v_lr (float): Learning rate for V-network.
        w_lr (float): Learning rate for importance-weight network.
        lambda_lr (float): Step size for dual variable.
        reward_scale (float): Reward scaling factor.
        cost_scale (float): Cost scaling factor.
        device (str): Compute device.
    """

    def __init__(
        self,
        model: PDOCRL_W,
        env: gym.Env,
        logger: WandbLogger | None = None,
        actor_lr: float = 3e-4,
        v_lr: float = 3e-4,
        w_lr: float = 3e-4,
        lambda_lr: float = 1e-3,
        reward_scale: float = 1.0,
        cost_scale: float = 1.0,
        device: str = "cpu",
    ):
        self.model = model
        self.logger = logger if logger is not None else DummyLogger()
        self.env = env
        self.reward_scale = reward_scale
        self.cost_scale = cost_scale
        self.device = device
        self.model.setup_optimizers(actor_lr, v_lr, w_lr, lambda_lr)

    def train_one_step(self, observations, next_observations, actions, rewards, costs,
                       done):
        """Update order: w → V → π → λ → target V sync."""
        _, stats_w = self.model.w_loss(observations, actions, next_observations,
                                       rewards, costs, done)
        _, stats_v = self.model.v_loss(observations, actions, next_observations,
                                       rewards, costs, done)
        _, stats_actor = self.model.actor_loss(observations, actions)
        stats_dual = self.model.dual_update(observations, actions, costs)
        self.model.sync_weight()

        self.logger.store(**stats_w)
        self.logger.store(**stats_v)
        self.logger.store(**stats_actor)
        self.logger.store(**stats_dual)

    def evaluate(self, eval_episodes):
        self.model.eval()
        episode_rets, episode_costs, episode_lens = [], [], []
        for _ in trange(eval_episodes, desc="Evaluating...", leave=False):
            epi_ret, epi_len, epi_cost = self.rollout()
            episode_rets.append(epi_ret)
            episode_lens.append(epi_len)
            episode_costs.append(epi_cost)
        self.model.train()
        return (
            np.mean(episode_rets) / self.reward_scale,
            np.mean(episode_costs) / self.cost_scale,
            np.mean(episode_lens),
        )

    @torch.no_grad()
    def rollout(self):
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
