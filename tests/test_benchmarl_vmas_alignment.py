import pytest


torch = pytest.importorskip("torch")

from experiments.benchmarl_vmas.alignment import linear_cka, nps_distance


def test_layer_normalized_linear_cka_is_invariant_to_mean_preserving_rotation():
    generator = torch.Generator().manual_seed(7)
    source = torch.randn(128, 16, generator=generator)
    # Per-sample LayerNorm singles out the all-ones direction.  Therefore the
    # full LN+CKA objective is invariant to orthogonal basis changes that
    # preserve that direction, rather than to arbitrary rotations.
    mean_direction = torch.ones(16) / (16**0.5)
    seed_basis = torch.cat(
        (mean_direction[:, None], torch.randn(16, 15, generator=generator)), dim=1
    )
    basis, _ = torch.linalg.qr(seed_basis)
    subspace_rotation, _ = torch.linalg.qr(torch.randn(15, 15, generator=generator))
    block_rotation = torch.eye(16)
    block_rotation[1:, 1:] = subspace_rotation
    rotation = basis @ block_rotation @ basis.T
    assert torch.allclose(rotation @ mean_direction, mean_direction, atol=1e-5)
    assert float(linear_cka(source, source @ rotation)) == pytest.approx(0.0, abs=1e-5)


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
