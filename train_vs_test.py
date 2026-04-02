"""
Training vs Test Data Extraction
Compares extraction rates on Enron emails (training data) vs TREC dataset (test data)
to validate that memorization metrics detect training data specifically.
"""

from datasets import load_dataset
from utils import setup_device, load_model, load_enron_emails, prepare_prefix_suffix_pairs
from utils import np_discoverable_extraction, get_sampling_fn

device = setup_device()
model, tokenizer = load_model("EleutherAI/pythia-2.8b", device=device)

# Prepare training data (Enron - in Pythia's training set)
emails = load_enron_emails(n=10000)
prefixes, suffixes = prepare_prefix_suffix_pairs(emails, tokenizer)

# Prepare test data (TREC - not in training set)
trec_dataset = load_dataset("trec", split="test", trust_remote_code=True)
trec_texts = trec_dataset["text"][:10000]

sampling_fn = get_sampling_fn("top_k")

n_values = list(range(10000, 100001, 10000)) + list(range(100000, 1000001, 100000))
p_values = [0.1, 0.5, 0.9]

results_data = []
for dataset_label, data in [("Enron", list(zip(prefixes, suffixes))),
                             ("TREC", [(text, text) for text in trec_texts])]:
    for n in n_values:
        for p_threshold in p_values:
            results = [
                np_discoverable_extraction(model, tokenizer, prefix, suffix, sampling_fn, n, p_threshold, device)
                for prefix, suffix in data
            ]
            extraction_rate = sum(results) / len(results)
            results_data.append({
                "dataset": dataset_label,
                "n_trials": n,
                "p_threshold": p_threshold,
                "extraction_rate": extraction_rate
            })
            print(f"{dataset_label}, n={n}, p={p_threshold}: {extraction_rate:.4f}")

print("\nAll results:", results_data)
