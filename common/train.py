"""
train.py — SMILES GPT for molecular generation.
Character-level autoregressive transformer trained on SMILES strings.

Hypothesis: including non-drug-like molecules in pretraining improves
generation diversity without hurting validity/uniqueness/novelty.

Usage:
    uv run train.py                       # train on data/chembl.csv
    uv run train.py --data custom.csv     # custom dataset
    uv run train.py --epochs 1            # run for 1 epoch
"""

import json
import math
import time
import random
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from rdkit.Chem import QED as rdkit_QED
from rdkit.Chem.Scaffolds import MurckoScaffold

from breakit_tokenizer import tokenize as _breakit_tokenize, build_vocab as _build_breakit_vocab, PAD, BOS, EOS, UNK

# Note: RDKit здесь только для evaluate(). Подготовка данных — в prepare_data.py

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

DATA_PATH    = "data/chembl.csv"
MAX_LEN      = 128   # max SMILES tokens; covers >99% of ChEMBL
N_GEN        = 5000  # molecules to generate per evaluation

N_LAYER      = 4
N_EMBD       = 256
N_HEAD       = 4     # head_dim = 64

EPOCHS       = 20

BATCH_SIZE   = 512
LR           = 3e-4
WEIGHT_DECAY = 0.1
BETAS        = (0.9, 0.95)
WARMUP_RATIO = 0.05  # fraction of total steps

# ---------------------------------------------------------------------------
# SMILES Tokenizer (BreakIt — fixed vocabulary)
# ---------------------------------------------------------------------------

class SmilesTokenizer:
    PAD, BOS, EOS, UNK = PAD, BOS, EOS, UNK

    def __init__(self):
        self.stoi, self.itos = _build_breakit_vocab()

    def build_vocab(self, smiles_list: list[str] | None = None) -> None:
        """No-op: vocabulary is fixed from breakit_tokenizer."""
        pass

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, smiles: str) -> list[int]:
        return [self.stoi.get(t, self.UNK) for t in _breakit_tokenize(smiles)]

    def decode(self, ids: list[int]) -> str:
        skip = {self.PAD, self.BOS, self.EOS, self.UNK}
        return "".join(self.itos[i] for i in ids if i not in skip and i in self.itos)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_smiles(path: str) -> list[str]:
    """Read pre-filtered SMILES from a CSV with a 'smiles' column."""
    df = pd.read_csv(path)
    col = next(c for c in df.columns if c.lower() == "smiles")
    mols = df[col].dropna().astype(str).tolist()
    print(f"Loaded {len(mols):,} molecules from {path}")
    return mols


def encode_dataset(smiles: list[str], tok: SmilesTokenizer) -> list[list[int]]:
    seqs = []
    for smi in smiles:
        ids = [tok.BOS] + tok.encode(smi) + [tok.EOS]
        seqs.append(ids[: MAX_LEN + 1])  # +1 because we shift into x/y
    return seqs


def make_batch(seqs: list[list[int]], device: torch.device):
    L = max(len(s) for s in seqs)
    padded = torch.tensor(
        [s + [SmilesTokenizer.PAD] * (L - len(s)) for s in seqs],
        dtype=torch.long, device=device,
    )
    x = padded[:, :-1]
    y = padded[:, 1:].clone()
    y[y == SmilesTokenizer.PAD] = -1  # ignored by cross_entropy
    return x, y


def iter_batches(seqs: list[list[int]], batch_size: int, device: torch.device):
    """One epoch over shuffled data."""
    idx = list(range(len(seqs)))
    random.shuffle(idx)
    for start in range(0, len(idx) - batch_size + 1, batch_size):
        yield make_batch([seqs[i] for i in idx[start : start + batch_size]], device)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    vocab_size: int
    max_seq_len: int = MAX_LEN
    n_layer:     int = N_LAYER
    n_embd:      int = N_EMBD
    n_head:      int = N_HEAD


def _rms_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head   = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv  = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd,     bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        shape = (B, T, self.n_head, self.head_dim)
        q, k, v = q.view(shape), k.view(shape), v.view(shape)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        q, k = _rms_norm(q), _rms_norm(k)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc   = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.relu(self.fc(x)).square())  # ReLU²


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn = Attention(cfg)
        self.mlp  = MLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(_rms_norm(x), cos, sin)
        x = x + self.mlp(_rms_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg     = cfg
        head_dim     = cfg.n_embd // cfg.n_head
        self.wte     = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks  = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs    = torch.outer(torch.arange(cfg.max_seq_len).float(), inv_freq)
        self.register_buffer("rope_cos", freqs.cos()[None, :, None, :])
        self.register_buffer("rope_sin", freqs.sin()[None, :, None, :])

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        x   = _rms_norm(self.wte(idx))
        cos = self.rope_cos[:, :T]
        sin = self.rope_sin[:, :T]
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.lm_head(_rms_norm(x))
        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits

    @torch.no_grad()
    def generate(self, n: int, bos_id: int, eos_id: int,
                 temperature: float = 1.0) -> list[list[int]]:
        device  = next(self.parameters()).device
        idx     = torch.full((n, 1), bos_id, dtype=torch.long, device=device)
        done    = torch.zeros(n, dtype=torch.bool, device=device)
        seqs: list[list[int]] = [[] for _ in range(n)]
        for _ in range(self.cfg.max_seq_len):
            logits    = self(idx)[:, -1, :] / temperature
            next_tok  = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx       = torch.cat([idx, next_tok], dim=1)
            for i in range(n):
                if done[i]:
                    continue
                t = next_tok[i].item()
                if t == eos_id:
                    done[i] = True
                else:
                    seqs[i].append(t)
            if done.all():
                break
        return seqs

    @classmethod
    def load(cls, path: str | Path, device: torch.device | None = None):
        path = Path(path)
        cfg  = GPTConfig(**json.load((path / "config.json").open()))
        tok  = SmilesTokenizer()
        td   = json.load((path / "tokenizer.json").open())
        tok.stoi = td["stoi"]
        tok.itos = {int(k): v for k, v in td["itos"].items()}
        # Use actual vocab size from tokenizer (may differ if vocab was extended during finetuning)
        cfg.vocab_size = len(tok.stoi)
        model = cls(cfg).to(device or "cpu")
        state = torch.load(path / "model.pt", map_location=device or "cpu", weights_only=True)
        # Strip torch.compile "_orig_mod." prefix if present
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        model.eval()
        return model, tok

# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def _morgan_fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def evaluate(
    model: GPT,
    tok: SmilesTokenizer,
    train_set: set[str],
    n: int = N_GEN,
    temperature: float = 1.0,
    eval_seed: int = 0,
) -> dict:
    # Fix seed for reproducible generation — save/restore so training shuffle is unaffected
    py_state  = random.getstate()
    th_state  = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    random.seed(eval_seed)
    torch.manual_seed(eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(eval_seed)

    model.eval()
    seqs    = model.generate(n, tok.BOS, tok.EOS, temperature)
    decoded = [tok.decode(s) for s in seqs]

    # Restore RNG state so training data shuffling is not affected by eval seed
    random.setstate(py_state)
    torch.set_rng_state(th_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state)

    # --- validity ---
    valid_pairs: list[tuple[str, object]] = []
    for smi in decoded:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_pairs.append((Chem.MolToSmiles(mol), mol))

    validity   = len(valid_pairs) / n
    if not valid_pairs:
        return dict(validity=0.0, uniqueness=0.0, novelty=0.0,
                    int_div=0.0, mean_tanimoto=0.0, scaffold_entropy=0.0, snn=0.0,
                    mean_qed=0.0, pct_lipinski=0.0,
                    w1_logp=float("nan"), w1_qed=float("nan"), w1_mw=float("nan"))

    canonical  = [smi for smi, _ in valid_pairs]
    mols       = [mol for _, mol in valid_pairs]

    # --- uniqueness ---
    unique_set = set(canonical)
    uniqueness = len(unique_set) / len(valid_pairs)

    # --- novelty ---
    novelty = sum(1 for s in unique_set if s not in train_set) / len(unique_set)

    # --- internal diversity (Tanimoto on Morgan FP, capped at 500 mols) ---
    fps    = [_morgan_fp(m) for m in mols[:500]]
    cap    = min(len(fps), 200)
    sims   = [
        DataStructs.TanimotoSimilarity(fps[i], fps[j])
        for i in range(cap) for j in range(i + 1, cap)
    ]
    mean_tanimoto = (sum(sims) / len(sims)) if sims else 0.0
    int_div       = 1.0 - mean_tanimoto

    # --- scaffold entropy (Bemis-Murcko) ---
    scaffold_counts: Counter = Counter()
    for mol in mols:
        try:
            s = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            scaffold_counts[s] += 1
        except Exception:
            pass
    total_scaffolded = sum(scaffold_counts.values())
    if total_scaffolded > 1:
        probs = [c / total_scaffolded for c in scaffold_counts.values()]
        raw_entropy = -sum(p * math.log(p) for p in probs)
        scaffold_entropy = raw_entropy / math.log(len(scaffold_counts))  # normalize to [0, 1]
    else:
        scaffold_entropy = 0.0

    # --- parse training reference sample (shared for SNN + property distances) ---
    _r          = random.Random(eval_seed)
    train_smpl  = _r.sample(sorted(train_set), min(2000, len(train_set)))
    train_ref   = [Chem.MolFromSmiles(s) for s in train_smpl]
    train_ref   = [m for m in train_ref if m is not None]

    # --- SNN: mean max-Tanimoto similarity of generated mols to training set ---
    train_fps   = [_morgan_fp(m) for m in train_ref[:1000]]
    snn_scores  = [max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)) for fp in fps]
    snn         = float(np.mean(snn_scores)) if snn_scores else 0.0

    # --- Wasserstein-1 on property distributions ---
    cap_p = min(len(mols), 2000)

    def _safe_qed(m):
        try:
            return rdkit_QED.qed(m)
        except Exception:
            return float("nan")

    gen_logp = [Descriptors.MolLogP(m) for m in mols[:cap_p]]
    gen_qed  = [_safe_qed(m)           for m in mols[:cap_p]]
    gen_mw   = [Descriptors.MolWt(m)   for m in mols[:cap_p]]

    ref_logp = [Descriptors.MolLogP(m) for m in train_ref]
    ref_qed  = [_safe_qed(m)           for m in train_ref]
    ref_mw   = [Descriptors.MolWt(m)   for m in train_ref]

    # --- drug-likeness metrics ---
    valid_qed = [v for v in gen_qed if not math.isnan(v)]
    mean_qed  = float(np.mean(valid_qed)) if valid_qed else 0.0

    lipinski_pass = sum(
        1 for m, mw, lp in zip(mols[:cap_p], gen_mw, gen_logp)
        if mw <= 500 and lp <= 5
        and Descriptors.NumHDonors(m) <= 5
        and Descriptors.NumHAcceptors(m) <= 10
    )
    pct_lipinski = lipinski_pass / len(mols[:cap_p]) if mols else 0.0

    score = validity * scaffold_entropy * int_div # decision metric: higher = better

    return dict(
        score=score,
        validity=validity,
        uniqueness=uniqueness,
        novelty=novelty,
        int_div=int_div,
        mean_tanimoto=mean_tanimoto,
        scaffold_entropy=scaffold_entropy,
        snn=snn,
        mean_qed=mean_qed,
        pct_lipinski=pct_lipinski,
        w1_logp=_wasserstein1d(gen_logp, ref_logp),
        w1_qed=_wasserstein1d(gen_qed,   ref_qed),
        w1_mw=_wasserstein1d(gen_mw,     ref_mw),
    )


def _wasserstein1d(p: list[float], q: list[float]) -> float:
    """Earth mover's distance for 1D distributions via sorted CDF interpolation."""
    if not p or not q:
        return float("nan")
    ps = np.sort(p)
    qs = np.sort(q)
    n  = max(len(ps), len(qs))
    t  = np.linspace(0, 1, n)
    pi = np.interp(t, np.linspace(0, 1, len(ps)), ps)
    qi = np.interp(t, np.linspace(0, 1, len(qs)), qs)
    return float(np.mean(np.abs(pi - qi)))

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def get_lr(step: int, total_steps: int) -> float:
    warmup = int(WARMUP_RATIO * total_steps)
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine decay


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default=DATA_PATH, help="Path to CSV with 'smiles' column")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--save",   default=None, help="Directory to save best checkpoint")
    parser.add_argument("--load",   default=None, help="Load checkpoint directory and generate molecules")
    parser.add_argument("--gen",    type=int, default=N_GEN, help="Number of molecules to generate")
    parser.add_argument("--temp",   type=float, default=1.0, help="Sampling temperature")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load checkpoint and generate ---
    if args.load:
        model, tok = GPT.load(args.load, device)
        print(f"Loaded {args.load} | {count_params(model) / 1e6:.1f}M params")
        # Fix seed for reproducible generation
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        seqs = model.generate(args.gen, tok.BOS, tok.EOS, args.temp)
        decoded = [tok.decode(s) for s in seqs]
        out = Path(args.load) / "generated.txt"
        out.write_text("\n".join(decoded) + "\n")
        print(f"Generated {args.gen} molecules -> {out}")
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"Device: {device}")

    # --- data ---
    smiles   = load_smiles(args.data)
    train_set = set(smiles)

    tok = SmilesTokenizer()
    tok.build_vocab(smiles)
    print(f"Vocab size: {tok.vocab_size}")

    seqs = encode_dataset(smiles, tok)
    steps_per_epoch = len(seqs) // BATCH_SIZE
    total_steps     = steps_per_epoch * args.epochs
    print(f"Steps per epoch: {steps_per_epoch:,}  |  Total: {total_steps:,}")

    # --- model ---
    cfg   = GPTConfig(vocab_size=tok.vocab_size)
    model = GPT(cfg).to(device)
    model = torch.compile(model)
    print(f"Parameters: {count_params(model) / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=BETAS
    )

    # --- training loop ---
    step      = 0
    best_score = -1.0
    t0         = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_t0   = time.time()

        for x, y in iter_batches(seqs, BATCH_SIZE, device):
            lr = LR * get_lr(step, total_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item()
            step += 1

        avg_loss    = epoch_loss / steps_per_epoch
        epoch_time  = time.time() - epoch_t0
        elapsed     = time.time() - t0

        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}  |  loss: {avg_loss:.4f}  |  "
              f"time: {epoch_time:.0f}s  |  elapsed: {elapsed/60:.1f}min")

        metrics = evaluate(model, tok, train_set)
        print(
            f"  score:            {metrics['score']:.4f}  (validity × scaffold_entropy x int_div)\n"
            f"  ---\n"
            f"  validity:         {metrics['validity']:.3f}\n"
            f"  uniqueness:       {metrics['uniqueness']:.3f}\n"
            f"  novelty:          {metrics['novelty']:.3f}\n"
            f"  int_div:          {metrics['int_div']:.3f}\n"
            f"  mean_tanimoto:    {metrics['mean_tanimoto']:.3f}\n"
            f"  scaffold_entropy: {metrics['scaffold_entropy']:.3f}\n"

            f"  snn:              {metrics['snn']:.3f}\n"
            f"  mean_qed:         {metrics['mean_qed']:.3f}\n"
            f"  pct_lipinski:     {metrics['pct_lipinski']:.3f}\n"
            f"  w1_logp:          {metrics['w1_logp']:.3f}\n"
            f"  w1_qed:           {metrics['w1_qed']:.3f}\n"
            f"  w1_mw:            {metrics['w1_mw']:.1f}"
        )

        if metrics['score'] > best_score:
            best_score = metrics['score']
            if args.save:
                save_dir = Path(args.save)
                save_dir.mkdir(parents=True, exist_ok=True)
                state = {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}
                torch.save(state, save_dir / "model.pt")
                json.dump(asdict(cfg), (save_dir / "config.json").open("w"))
                json.dump({
                    "stoi": tok.stoi,
                    "itos": {int(k): v for k, v in tok.itos.items()},
                }, (save_dir / "tokenizer.json").open("w"))
                json.dump(metrics, (save_dir / "metrics.json").open("w"), indent=2)
                print(f"  ** saved best (score={best_score:.4f}) to {save_dir}")


if __name__ == "__main__":
    main()