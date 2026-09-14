from scripts.run_h1_diagnostics import worker_environment


def test_diagnostic_worker_matches_float32_checkpoint_dtype():
    environment = worker_environment(
        {
            "PATH": "/example/bin",
            "LD_LIBRARY_PATH": "/stale/cuda",
            "JAX_ENABLE_X64": "true",
        },
        3,
        7,
    )

    assert environment["PATH"] == "/example/bin"
    assert environment["CUDA_VISIBLE_DEVICES"] == "3"
    assert environment["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert environment["JAX_ENABLE_X64"] == "false"
    assert "LD_LIBRARY_PATH" not in environment
    assert environment["OMP_NUM_THREADS"] == "7"
    assert environment["OPENBLAS_NUM_THREADS"] == "7"
    assert environment["MKL_NUM_THREADS"] == "7"
    assert environment["NUMEXPR_NUM_THREADS"] == "7"
    assert environment["VECLIB_MAXIMUM_THREADS"] == "7"
