# LLM Memorization & Data Extraction Research

Research conducted at Berkeley Artificial Intelligence Research (BAIR) on measuring and characterizing training data memorization in large language models.

## Overview

This project implements the **(n, p)-discoverable extraction** framework to systematically measure how much training data LLMs memorize and can reproduce. Given a 50-token prefix from a known training document, we generate `n` continuations and measure whether at least fraction `p` exactly reproduce the original next 50 tokens.

We extend the standard exact-match metric with **edit similarity** (character-level SequenceMatcher) and **semantic similarity** (sentence-transformer embeddings) to capture approximate memorization that exact token matching misses.

## Experiments

| File | Description |
|------|-------------|
| `sampling_strategy.py` | Compares extraction rates across greedy, top-k, top-p, and temperature sampling |
| `np_parameter_sweep.py` | Systematic sweep across n (10–10,000) and p (0.1–0.999) thresholds |
| `model_comparison.py` | Compares extraction rates across model sizes (Pythia 1B, GPT-Neo 1.3B) |
| `train_vs_test.py` | Validates metric by comparing extraction on training data (Enron) vs held-out data (TREC) |
| `extended_memorization_metrics.py` | Full pipeline with exact, edit, and semantic similarity metrics |
| `fine_tune_epoch_1.py` | Continued training of OLMo-7B on synthetic data (1 epoch) |
| `fine_tune_epoch_3.py` | Continued training of OLMo-7B on synthetic data (3 epochs) |

## Core Utilities

`utils.py` contains shared functions for model loading, dataset preparation, sampling configuration, and the core discoverable extraction implementation.

## Models Evaluated

- EleutherAI/Pythia (1B, 2.8B, 6.9B, 12B)
- EleutherAI/GPT-Neo 1.3B
- AllenAI/OLMo-7B (fine-tuned)

## Datasets

- **Enron emails** — known to be in Pythia's training corpus (primary evaluation set)
- **TREC** — held-out dataset for negative control
- **Synthetic data** — for continued training / fine-tuning experiments

## Infrastructure

Experiments run on distributed GPU infrastructure using HuggingFace Accelerate with bfloat16 precision, gradient checkpointing, and gradient accumulation.

## Key Dependencies

- PyTorch, Transformers, Accelerate
- sentence-transformers (for semantic similarity)
- FAISS (for embedding indexing)
