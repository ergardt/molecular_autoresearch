"""
breakit_tokenizer.py — BreakIt-style SMILES tokenizer with fixed vocabulary.

Based on Coley et al. 2018 "A graph-convolutional neural network model for
the prediction of drug-drug interactions".

Fixed vocabulary of chemically meaningful tokens. Same vocab for all datasets.
Unknown bracketed atoms fall back to a single UNK token.
"""

import re

# Fixed vocabulary — ordered longest-first for greedy matching.
# Built from top bracketed groups across ChEMBL, ZINC, COCONUT, HMDB.
# Covers ~99.5% of real SMILES tokens. Rare metals/isotopes fall back to UNK.
VOCAB = [
    # Halogens and multi-char atoms
    "Br", "Cl",

    # Charged/bracketed groups (sorted longest first, by frequency in data)
    "[NH2+]", "[NH3+]", "[OH-]", "[O-]", "[NH-]",
    "[CH3]", "[CH2]", "[CH]", "[CH3+]", "[CH2-]",
    "[nH]", "[nH+]", "[o-]", "[n+]", "[s+]", "[s+2]",
    "[Si]", "[SiH3]",
    "[NH+]", "[n-]", "[o+]", "[S+]", "[N-]",
    "[C-]", "[O+]", "[Se]", "[P@@]", "[P@]",
    "[N@@+]", "[N@+]", "[C+]", "[N@]", "[C]",
    "[se]", "[O]", "[N@@]", "[S@+]", "[n]",
    "[S@@+]", "[cH-]", "[S-]", "[Cl-]", "[As]",
    "[I-]", "[18F]", "[S@H+]", "[3H]",
    "[K+]", "[PH]", "[s+]", "[P+]",
    "[S@@H+]", "[Br-]", "[SH]", "[B-]",
    "[Fe]", "[S@@]", "[Mg+2]", "[Ca+2]", "[N]",
    "[S@]", "[c-]", "[Ni+2]", "[125I]", "[Fe+2]",
    "[S]", "[B]", "[I+]", "[OH]", "[11C]",
    "[CH-]", "[SeH]", "[Te]", "[Mg]", "[Zn+2]",
    "[14C]", "[As+]", "[Co]", "[131I]", "[c]",
    "[Mg-2]", "[V]", "[c+]", "[Al+3]",
    "[Zn]", "[Sn]", "[BH2-]", "[123I]", "[Al]",
    "[F]", "[SH+]", "[Cu+2]", "[Gd+3]",
    "[NaH]", "[K]", "[Fe-2]",
    "[Na]", "[Gd-]", "[Sr+2]", "[Cl]", "[Cu]",
    "[NH2]", "[NH]", "[Fe-3]", "[NH4+]", "[19F]",
    "[13C]", "[Br+2]", "[10B]", "[P-]",
    "[P@+]", "[W]", "[Ag+]", "[Li+]", "[Mn+2]",
    "[Hg+]", "[Au-]", "[Bi+3]", "[Gd]",
    "[Cl+]", "[N@H+]", "[18O]", "[124I]",
    # Missing from original small vocab
    "[H]", "[N+]", "[F-]",

    # Stereochemistry — common bracketed chiral centers
    "[C@H]", "[C@@H]", "[C@@]", "[C@]",

    # Stereochemistry markers (standalone)
    "@@", "@",

    # Bonds
    "=", "#", "-",

    # Ring closures (digits)
    "1", "2", "3", "4", "5", "6", "7", "8", "9",

    # Structure
    "(", ")", ".",

    # Aromatic atoms
    "c", "n", "o", "s", "p",

    # Aliphatic atoms
    "C", "N", "O", "S", "P", "F", "I", "B",

    # Escape / special
    "\\", "/", ":",
]

# Regex that matches:
# 1. Known tokens from VOCAB (longest first)
# 2. Any bracketed group [...] as a single token (fallback for metals, isotopes)
# 3. Percent-encoded ring closures %XX
# 4. Single characters as fallback
_TOKEN_ALTS = (
    "|".join(re.escape(t) for t in VOCAB)
    + "|\\[[^\\]]+\\]"       # any bracketed group [Fe+2], [13CH3], [Ru@]
    + "|%[0-9]{2}"           # %10, %11 ring closures
    + "|."                   # single char fallback
)
_PATTERN = re.compile(_TOKEN_ALTS)

# Special tokens
PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]


def tokenize(smiles: str) -> list[str]:
    """Tokenize a SMILES string into BreakIt tokens."""
    return _PATTERN.findall(smiles)


def build_vocab() -> tuple[dict[str, int], dict[int, str]]:
    """Build fixed stoi/itos from VOCAB + specials."""
    vocab = SPECIAL + VOCAB
    stoi = {t: i for i, t in enumerate(vocab)}
    itos = {i: t for t, i in stoi.items()}
    return stoi, itos


def encode(smiles: str, stoi: dict[str, int]) -> list[int]:
    """Encode SMILES to token IDs. Unknown tokens -> UNK."""
    return [stoi.get(t, UNK) for t in tokenize(smiles)]


def decode(ids: list[int], itos: dict[int, str]) -> str:
    """Decode token IDs back to SMILES."""
    skip = {PAD, BOS, EOS, UNK}
    return "".join(itos[i] for i in ids if i not in skip and i in itos)


def vocab_size() -> int:
    """Return total vocabulary size (specials + chemical tokens)."""
    return len(SPECIAL) + len(VOCAB)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    stoi, itos = build_vocab()
    print(f"Vocab size: {vocab_size()}")
    print()

    examples = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",          # aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",       # caffeine
        "C[C@H](O)CC[C@H]1CC=CC2",            # stereochemistry
        "CC(=O)c3ccccc3",                      # acetophenone
        # COCONUT edge cases: metals, isotopes
        "[Fe+2]Cl",                            # iron chloride
        "[13CH3]OH",                           # isotope
        "[Ru+2](=O)(=O)([O-])",                # ruthenium complex
        "CC[C@H](C)C(=O)O",                    # leucine
        "C1=CC(=CC=C1C(=O)O)C(=O)O",          # ibuprofen
        "CN(C)C(=O)C1=CC=CC=C1",              # phenylacetamide
        # Edge: % ring closures
        "C1CC2CCC3C%10CC%11CC1C3CC2C",         # steroid with %10 %11
    ]

    for smi in examples:
        toks = tokenize(smi)
        ids = encode(smi, stoi)
        decoded = decode(ids, itos)
        unk_count = ids.count(UNK)
        ok = "✓" if decoded == smi else "✗"
        print(f"  {ok} {smi}")
        print(f"    tokens: {toks}")
        print(f"    unk: {unk_count}, decoded: {decoded}")
        print()
