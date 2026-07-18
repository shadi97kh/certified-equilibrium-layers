"""RNA-FM per-nucleotide and mean-pooled embeddings for RNA sequences.

Loads the RNA-FM foundation model via ``fm.pretrained.rna_fm_t12()`` (Chen et al. 2022,
the 12-layer, 640-dimensional RNA language model) and embeds a list of RNA sequences
(siRNA guides of 19 to 23 nt and 3'UTR seed-match windows, each well under the 1024-token
limit). For each sequence it returns the 640-d per-nucleotide embeddings (one vector per
base, the special BOS / EOS tokens stripped) and the mean-pooled 640-d sequence vector,
as numpy arrays, and can write them to a JSON sidecar plus a .npy file.

REAL DATA ONLY: there is NO synthetic fallback. A missing, empty, or non-RNA sequence
RAISES; the model is never substituted by random vectors. torch and the ``fm`` (rna-fm)
package are required to CALL the extractor and are imported lazily, so importing this
module stays light (the lightweight numpy import path never pulls in torch / fm).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
RNAFM_DIM = 640  # RNA-FM rna_fm_t12 hidden dimension
RNAFM_MAX_SEQ_LEN = 1022  # 1024-token limit minus the BOS and EOS tokens
RNAFM_INSTALL = "pip install rna-fm torch  (the 'fm' package; see github.com/ml4bio/RNA-FM)"
RNAFM_CHECKPOINT_DIR = REPO / "external" / "RNA-FM" / "pretrained"  # cuhkaih/rnafm weights
_REPR_LAYER = 12  # final layer of rna_fm_t12

_MODEL_CACHE: dict = {}


def confirm_checkpoint(checkpoint_dir: str | Path = RNAFM_CHECKPOINT_DIR) -> str:
    """Confirm the RNA-FM checkpoint is present and loads via fm.pretrained.rna_fm_t12().

    Returns the resolved checkpoint path. RAISES (hard failure, not a silent pending) when
    the weights are missing or fail to load, PRINTING the expected path so it can be placed
    correctly. Requires the 'fm' package (ImportError if absent).
    """
    require_rnafm()
    import fm

    cdir = Path(checkpoint_dir)
    pths = sorted(cdir.glob("*.pth")) if cdir.exists() else []
    if not pths:
        expected = cdir / "RNA-FM_pretrained.pth"
        print(
            f"[rnafm] RNA-FM checkpoint MISSING. Place the cuhkaih/rnafm weights at: {expected}\n"
            f"[rnafm]   huggingface-cli download cuhkaih/rnafm --local-dir {cdir}"
        )
        raise FileNotFoundError(
            f"RNA-FM checkpoint not found under {cdir}; expected a .pth (e.g. {expected}). "
            "No silent fallback: place the cuhkaih/rnafm weights there."
        )
    try:
        fm.pretrained.rna_fm_t12()
    except Exception as exc:  # checkpoint present but unreadable / wrong: hard fail with the path
        print(
            f"[rnafm] RNA-FM checkpoint at {pths[0]} failed to load via fm.pretrained.rna_fm_t12()"
        )
        raise FileNotFoundError(
            f"RNA-FM checkpoint under {cdir} did not load via fm.pretrained.rna_fm_t12() ({exc}); "
            f"confirm the cuhkaih/rnafm weights at {pths[0]} are the rna_fm_t12 checkpoint."
        ) from exc
    return str(pths[0])


def require_rnafm() -> None:
    """Raise ImportError (with the install command) if torch or RNA-FM are unavailable."""
    try:
        import fm  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"RNA-FM embeddings need torch and the 'fm' (rna-fm) package, which are not "
            f"installed. {RNAFM_INSTALL}"
        ) from exc


def _clean_sequences(sequences: list[str]) -> list[str]:
    """Validate and normalize RNA sequences; RAISE on any missing / invalid one."""
    if sequences is None or len(sequences) == 0:
        raise ValueError("no sequences provided")
    clean: list[str] = []
    for i, s in enumerate(sequences):
        if s is None or not str(s).strip():
            raise ValueError(f"sequence {i} is missing or empty; REAL DATA ONLY, no fallback")
        seq = str(s).strip().upper().replace("T", "U")
        bad = sorted({c for c in seq if c not in "ACGUN"})
        if bad:
            raise ValueError(f"sequence {i} has non-RNA characters {bad}: {seq!r}")
        if len(seq) > RNAFM_MAX_SEQ_LEN:
            raise ValueError(
                f"sequence {i} length {len(seq)} exceeds the RNA-FM limit {RNAFM_MAX_SEQ_LEN}"
            )
        clean.append(seq)
    return clean


def load_model():
    """Load (and cache) RNA-FM rna_fm_t12: returns (model, alphabet, batch_converter).

    Confirms the checkpoint first, so a missing / unreadable checkpoint is a hard failure
    (FileNotFoundError with the expected path) rather than a silent download or fallback.
    The model is loaded in float32 and never half precision.
    """
    require_rnafm()
    if "rna_fm_t12" not in _MODEL_CACHE:
        import fm

        confirm_checkpoint()
        model, alphabet = fm.pretrained.rna_fm_t12()
        model = model.float().eval()  # force float32; never fp16
        _MODEL_CACHE["rna_fm_t12"] = (model, alphabet, alphabet.get_batch_converter())
    return _MODEL_CACHE["rna_fm_t12"]


def resolve_device(device=None):
    """Resolve the torch device: explicit, else cuda if available else cpu."""
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def extract_embeddings(
    sequences: list[str],
    labels: list[str] | None = None,
    batch_size: int = 8,
    device=None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Return (per_nucleotide, mean_pooled) RNA-FM embeddings for ``sequences``.

    per_nucleotide is a list of float32 arrays, one (L_i, 640) array per sequence (the
    special tokens stripped); mean_pooled is a float32 (n, 640) array. Both the model and
    the tokenized inputs are moved to ``device`` (default: cuda if available else cpu); the
    embeddings stay float32 (no fp16 autocast). Raises on any missing / invalid sequence
    (no synthetic fallback).
    """
    import torch

    clean = _clean_sequences(sequences)
    if labels is None:
        labels = [f"seq{i}" for i in range(len(clean))]
    elif len(labels) != len(clean):
        raise ValueError(f"got {len(labels)} labels for {len(clean)} sequences")
    dev = resolve_device(device)
    model, _alphabet, batch_converter = load_model()
    model = model.float().to(dev).eval()  # float32 on the chosen device; never fp16

    per_nt: list[np.ndarray] = []
    pooled: list[np.ndarray] = []
    for start in range(0, len(clean), batch_size):
        chunk = clean[start : start + batch_size]
        chunk_labels = labels[start : start + batch_size]
        _, _, tokens = batch_converter(list(zip(chunk_labels, chunk, strict=True)))
        tokens = tokens.to(dev)
        # No autocast: run the forward in plain float32 so the device move cannot change values.
        with torch.no_grad():
            reps = model(tokens, repr_layers=[_REPR_LAYER])["representations"][_REPR_LAYER]
        if reps.dtype != torch.float32:
            raise ValueError(
                f"RNA-FM embeddings must stay float32; got {reps.dtype} (no fp16 / autocast allowed)"
            )
        for j, seq in enumerate(chunk):
            # tokens are [BOS, base_1 ... base_L, EOS, pad ...]; keep the L base positions.
            emb = reps[j, 1 : len(seq) + 1, :].detach().to("cpu").numpy().astype(np.float32)
            if emb.shape != (len(seq), RNAFM_DIM):
                raise ValueError(
                    f"unexpected RNA-FM output {emb.shape} for a {len(seq)} nt sequence"
                )
            per_nt.append(emb)
            pooled.append(emb.mean(axis=0))
    return per_nt, np.stack(pooled).astype(np.float32)


def write_embeddings(
    sequences: list[str], out_prefix: str | Path, labels: list[str] | None = None
) -> tuple[list[np.ndarray], np.ndarray]:
    """Extract and write the embeddings to ``out_prefix`` + .npy and + .json.

    The .npy holds the per-nucleotide embeddings as a length-n object array (one (L_i, 640)
    array per sequence); the JSON sidecar holds the metadata and the mean-pooled vectors.
    Returns (per_nucleotide, mean_pooled).
    """
    out_prefix = Path(out_prefix)
    clean = _clean_sequences(sequences)
    if labels is None:
        labels = [f"seq{i}" for i in range(len(clean))]
    per_nt, pooled = extract_embeddings(clean, labels=labels)

    obj = np.empty(len(per_nt), dtype=object)  # ragged-safe 1-D object array
    for i, emb in enumerate(per_nt):
        obj[i] = emb
    npy_path = out_prefix.with_suffix(".npy")
    np.save(npy_path, obj, allow_pickle=True)

    sidecar = {
        "model": "RNA-FM rna_fm_t12",
        "embedding_dim": RNAFM_DIM,
        "n_sequences": len(clean),
        "labels": labels,
        "sequences": clean,
        "lengths": [int(emb.shape[0]) for emb in per_nt],
        "mean_pooled": pooled.tolist(),
        "per_nucleotide_npy": npy_path.name,
        "note": "per-nucleotide 640-d embeddings are in the .npy (object array, one (L,640) "
        "per sequence); the mean-pooled 640-d vectors are listed here.",
    }
    out_prefix.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return per_nt, pooled
