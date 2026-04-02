"""
Model Size Comparison
Compares (n,p)-discoverable extraction rates across different model sizes
(Pythia 1B, GPT-Neo 1.3B) with full (n, p) parameter sweeps.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import setup_device, load_enron_emails, prepare_prefix_suffix_pairs
from utils import np_discoverable_extraction, get_sampling_fn

device = setup_device()

# Load dataset once, prepare pairs with default tokenizer
base_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b")
emails = load_enron_emails(n=10000)
prefixes, suffixes = prepare_prefix_suffix_pairs(emails, base_tokenizer)

models = {
    "Pythia 1B": "EleutherAI/pythia-1b",
    "GPT-Neo 1.3B": "EleutherAI/gpt-neo-1.3B",
}

p_values = [0.1, 0.9]
n_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
            200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000,
            6000, 7000, 8000, 9000, 10000]

sampling_fn = get_sampling_fn("top_k")
extraction_results = {name: {p: [] for p in p_values} for name in models}

for model_name, model_path in models.items():
    print(f"\nLoading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for p in p_values:
        for n in n_values:
            rate = np.mean([
                np_discoverable_extraction(model, tokenizer, prefix, suffix, sampling_fn, n, p)
                for prefix, suffix in zip(prefixes, suffixes)
            ])
            extraction_results[model_name][p].append(rate)
            print(f"  {model_name}, n={n}, p={p}: {rate:.4f}")

# Save results
results_data = []
for model_name in models:
    for p in p_values:
        for n, rate in zip(n_values, extraction_results[model_name][p]):
            results_data.append((model_name, n, p, rate))

df = pd.DataFrame(results_data, columns=["Model", "n", "p", "Extraction Rate"])
print("\nResults:")
print(df)

# Plot
plt.figure(figsize=(10, 6))
for model_name in models:
    for p in p_values:
        plt.plot(n_values, extraction_results[model_name][p], label=f"{model_name}, p={p}")
plt.xscale("log")
plt.xlabel("Number of Trials (n)")
plt.ylabel("Extraction Rate")
plt.title("(n, p)-Discoverable Extraction Rates by Model Size")
plt.legend()
plt.grid(True)
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
