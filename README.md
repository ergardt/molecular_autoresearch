# Autonomous discovery of training distributions for controllable molecular exploration

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

> Optimize training distribution 
