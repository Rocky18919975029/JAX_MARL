import json

import numpy as np

from scripts.h1_diagnostic_data import load_diagnostics


def test_loader_reads_only_stage_arrays(tmp_path):
    np.savez_compressed(
        tmp_path / "episodes_0000.npz",
        active=np.ones((2, 3), dtype=bool),
        unused=np.ones((2, 3, 500)),
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps({"shards": [{"path": "episodes_0000.npz", "episodes": 2}]})
    )
    metadata, arrays = load_diagnostics(tmp_path, ("active",))
    assert metadata["shards"][0]["episodes"] == 2
    assert set(arrays) == {"active"}
    assert arrays["active"].shape == (2, 3)
