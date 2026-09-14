from scripts.run_h1_diagnostics import worker_environment


def test_diagnostic_worker_matches_float32_checkpoint_dtype():
    environment = worker_environment(
        {
            "PATH": "/example/bin",
            "LD_LIBRARY_PATH": "/stale/cuda",
            "JAX_ENABLE_X64": "true",
        },
        3,
    )

    assert environment["PATH"] == "/example/bin"
    assert environment["CUDA_VISIBLE_DEVICES"] == "3"
    assert environment["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert environment["JAX_ENABLE_X64"] == "false"
    assert "LD_LIBRARY_PATH" not in environment
