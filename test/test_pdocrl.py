"""Smoke tests for PDOCRL algorithm."""
import torch


def test_pdocrl_instantiation():
    """PDOCRL model can be instantiated with default params."""
    from osrl.algorithms import PDOCRL
    model = PDOCRL(state_dim=8, action_dim=2, max_action=1.0, device="cpu")
    assert model is not None
    assert model.dual_bound > 0


def test_pdocrl_smoke():
    """PDOCRL model runs one full gradient step with finite losses."""
    from osrl.algorithms import PDOCRL
    model = PDOCRL(state_dim=8, action_dim=2, max_action=1.0, device="cpu")
    model.setup_optimizers(actor_lr=3e-4, q_lr=3e-4, w_lr=3e-4, lambda_lr=1e-3)

    B = 32
    obs = torch.randn(B, 8)
    next_obs = torch.randn(B, 8)
    act = torch.randn(B, 2).clamp(-1.0, 1.0)
    rew = torch.randn(B)
    cost = torch.rand(B)
    done = torch.zeros(B)

    # w-player
    loss_w, stats_w = model.w_loss(obs, act, next_obs, rew, cost, done)
    assert torch.isfinite(loss_w), f"w_loss is not finite: {loss_w.item()}"

    # Q-player
    loss_q, stats_q = model.q_loss(obs, act, next_obs, rew, cost, done)
    assert torch.isfinite(loss_q), f"q_loss is not finite: {loss_q.item()}"

    # pi-player
    loss_a, stats_a = model.actor_loss(obs)
    assert torch.isfinite(loss_a), f"actor_loss is not finite: {loss_a.item()}"

    # dual update
    stats_dual = model.dual_update(obs, act)
    assert "misc/lambda" in stats_dual
    lam_val = stats_dual["misc/lambda"]
    assert 0.0 <= lam_val <= model.dual_bound + 1e-6, (
        f"lambda={lam_val} out of [0, {model.dual_bound}]"
    )

    # target sync (should not crash)
    model.sync_weight()


def test_pdocrl_w_net_positive():
    """w_net outputs must be non-negative (Softplus activation)."""
    from osrl.algorithms import PDOCRL
    model = PDOCRL(state_dim=8, action_dim=2, max_action=1.0, device="cpu")
    obs = torch.randn(64, 8)
    act = torch.randn(64, 2)
    w_in = torch.cat([obs, act], dim=-1)
    w_vals = model.w_net(w_in).squeeze(-1)
    assert (w_vals >= 0).all(), "w_net produced negative values"


def test_pdocrl_act():
    """model.act returns an action with the right shape."""
    import numpy as np
    from osrl.algorithms import PDOCRL
    model = PDOCRL(state_dim=8, action_dim=2, max_action=1.0, device="cpu")
    obs = np.random.randn(8).astype(np.float32)
    act, _ = model.act(obs, deterministic=True)
    assert act.shape == (2,), f"Expected shape (2,), got {act.shape}"
