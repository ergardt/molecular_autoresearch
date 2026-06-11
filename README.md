# Autonomous Discovery of Molecular Training Distributions

Molecular generative models are usually trained on fixed, human-curated datasets.  
In this project, the dataset itself becomes the object of optimization.

We let an autonomous agent discover what the model should learn from.

---

## The Idea

Instead of hand-designing training data, we:

- Define a parameterized space of molecular training distributions.
- Give an agent a small, fast GPT training setup.
- Let it iteratively:
  - Modify the training distribution
  - Train for a fixed time budget
  - Evaluate diversity, collapse, and controllability
  - Keep or discard the change

You don’t edit the model manually.  
You edit the research instructions in `program.md`.

---

## Philosophy

Traditional setup:

> Optimize model parameters θ given fixed data D.

This project:

> Optimize training distribution D given fixed model class.

We treat the dataset as a controllable variable.

---

## Repository Structure

Only three components matter:

- `prepare.py` — data prep and evaluation utilities (fixed).
- `train.py` — molecular GPT training loop (agent-modifiable).
- `program.md` — research specification for the agent (human-edited).

All experimental logic and objectives live in `program.md`.

---

## Evaluation

Each run:

- Fixed compute budget (short wall-clock training).
- Generate molecules.
- Evaluate:
  - Scaffold diversity
  - Internal diversity
  - Mode collapse index
  - Validity
  - *(Optional)* downstream fine-tune proxy

Models are compared under identical compute constraints.

---

## Goal

Discover non-trivial training distributions that:

- Reduce mode collapse
- Increase structural expressivity
- Improve downstream adaptability

without increasing model size or compute.

---

For detailed objectives, constraints, and reward definitions, see `program.md`.
