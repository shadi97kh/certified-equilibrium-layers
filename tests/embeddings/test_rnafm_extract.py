"""RNA-FM embedding extraction on real Huesken 2005 guide sequences.

The full extraction test needs the rna-fm ('fm') package + torch and is skipped when they
are absent (the lightweight import path never requires them); the validation test runs
without RNA-FM. Guide sequences are read from data/huesken2005.csv (real Huesken 2005
guides, the antisense / silencing strands derived from the OligoFormer Hu.csv).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cel.embeddings import RNAFM_DIM, extract_embeddings, write_embeddings  # noqa: E402
from cel.embeddings.rnafm_extract import (  # noqa: E402
    _clean_sequences,
    confirm_checkpoint,
    resolve_device,
)

HUESKEN_CSV = REPO / "data" / "huesken2005.csv"


def _three_real_guides() -> list[str]:
    if not HUESKEN_CSV.exists():
        pytest.skip(f"real Huesken 2005 guides not present at {HUESKEN_CSV}")
    with HUESKEN_CSV.open() as fh:
        rows = list(csv.DictReader(fh))
    guides = [r["guide"].strip().upper() for r in rows[:3]]
    assert len(guides) == 3 and all(19 <= len(g) <= 23 for g in guides)
    return guides


def test_clean_sequences_raises_on_missing_or_invalid() -> None:
    # No synthetic fallback: a missing / empty / non-RNA sequence must RAISE.
    with pytest.raises(ValueError):
        _clean_sequences(["ACGU", ""])
    with pytest.raises(ValueError):
        _clean_sequences(["ACGU", None])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        _clean_sequences(["ACGZ"])  # Z is not an RNA base
    # A valid guide passes and is normalized (T mapped to U, uppercased).
    assert _clean_sequences(["acgt"]) == ["ACGU"]


def test_rnafm_embeddings_on_real_huesken_guides(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("fm")  # the rna-fm package; skip the model test when absent

    guides = _three_real_guides()
    per_nt, pooled = extract_embeddings(guides, labels=["h0", "h1", "h2"])

    assert pooled.shape == (3, RNAFM_DIM)
    assert pooled.dtype == np.float32 and np.all(np.isfinite(pooled))
    assert len(per_nt) == 3
    for emb, g in zip(per_nt, guides, strict=True):
        assert emb.shape == (len(g), RNAFM_DIM)  # one 640-d vector per nucleotide
        assert np.allclose(emb.mean(axis=0), pooled[guides.index(g)], atol=1e-5)

    # Sidecar + .npy round-trip.
    per_nt2, pooled2 = write_embeddings(guides, tmp_path / "huesken_rnafm")
    import json

    side = json.loads((tmp_path / "huesken_rnafm.json").read_text())
    assert side["embedding_dim"] == RNAFM_DIM and side["n_sequences"] == 3
    loaded = np.load(tmp_path / "huesken_rnafm.npy", allow_pickle=True)
    assert loaded.shape == (3,) and loaded[0].shape == (len(guides[0]), RNAFM_DIM)
    assert np.allclose(np.array(side["mean_pooled"], dtype=np.float32), pooled2, atol=1e-5)


def test_resolve_device() -> None:
    torch = pytest.importorskip("torch")
    assert resolve_device("cpu") == torch.device("cpu")
    auto = resolve_device(None)
    assert auto.type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_confirm_checkpoint_missing_raises(tmp_path: Path) -> None:
    pytest.importorskip("fm")  # needs the rna-fm package; the checkpoint dir is empty here
    with pytest.raises(FileNotFoundError):
        confirm_checkpoint(checkpoint_dir=tmp_path)


def test_cpu_cuda_mean_pooled_agree() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("fm")
    guides = _three_real_guides()

    per_nt_cpu, pooled_cpu = extract_embeddings(guides, device="cpu")
    assert pooled_cpu.dtype == np.float32 and per_nt_cpu[0].dtype == np.float32

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; CPU path validated, CUDA half skipped cleanly")
    _per_nt_gpu, pooled_gpu = extract_embeddings(guides, device="cuda")
    assert pooled_gpu.dtype == np.float32  # stays float32 on GPU, no fp16 autocast
    # The device move must not change the embeddings beyond ULP-level float noise.
    assert float(np.max(np.abs(pooled_cpu - pooled_gpu))) < 1e-4
