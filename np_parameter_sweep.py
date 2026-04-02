"""
(n, p) Parameter Sweep
Systematic sweep across n (number of generation attempts) and p (success threshold)
to characterize extraction rates on Pythia-2.8B with Enron emails.
"""

from utils import setup_device, load_model, load_enron_emails, prepare_prefix_suffix_pairs
from utils import np_discoverable_extraction, get_sampling_fn

device = setup_device()
model, tokenizer = load_model("EleutherAI/pythia-2.8b", device=device)
emails = load_enron_emails(n=10000)
prefixes, suffixes = prepare_prefix_suffix_pairs(emails, tokenizer)

sampling_fn = get_sampling_fn("top_k")

n_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
            200, 300, 400, 500, 600, 700, 800, 900, 1000,
            2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
p_values = [0.1, 0.5, 0.9, 0.99, 0.999]

results_data = []
for n in n_values:
    for p_threshold in p_values:
        results = [
            np_discoverable_extraction(model, tokenizer, prefix, suffix, sampling_fn, n, p_threshold, device)
            for prefix, suffix in zip(prefixes, suffixes)
        ]
        extraction_rate = sum(results) / len(results)
        results_data.append((n, p_threshold, extraction_rate))
        print(f"n={n}, p={p_threshold}: extraction rate = {extraction_rate:.4f}")

print("\nAll results:", results_data)
