import pytest


torch = pytest.importorskip("torch")

from experiments.benchmarl_vmas.alignment import linear_cka, nps_distance


def test_linear_cka_is_rotation_invariant():
    generator = torch.Generator().manual_seed(7)
    source = torch.randn(128, 16, generator=generator)
    q, _ = torch.linalg.qr(torch.randn(16, 16, generator=generator))
    assert float(linear_cka(source, source @ q)) == pytest.approx(0.0, abs=1e-5)


def test_c_to_a_detaches_critic_and_preserves_actor_gradient():
    actor = torch.randn(64, 5, 16, requires_grad=True)
    critic = torch.randn(64, 5, 16, requires_grad=True)
    loss = nps_distance(actor, critic, "ln_mse")
    loss.backward()
    assert actor.grad is not None and torch.isfinite(actor.grad).all()
    assert critic.grad is None


def test_nps_cka_is_computed_per_slot():
    generator = torch.Generator().manual_seed(11)
    actor = torch.randn(128, 5, 8, generator=generator)
    critic = actor.clone()
    critic[:, 1] = critic[:, 1].roll(1, dims=0)
    per_slot = torch.stack(
        [linear_cka(actor[:, slot], critic[:, slot]) for slot in range(5)]
    ).mean()
    assert nps_distance(actor, critic, "linear_cka") == pytest.approx(per_slot)
