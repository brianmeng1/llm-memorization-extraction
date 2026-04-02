"""
Sampling Strategy Comparison
Measures (n,p)-discoverable extraction rates across greedy, top-k, top-p, 
and temperature sampling on Pythia-2.8B with 10,000 Enron emails.
"""

from utils import setup_device, load_model, load_enron_emails, prepare_prefix_suffix_pairs
from utils import np_discoverable_extraction, get_sampling_fn

device = setup_device()
model, tokenizer = load_model("EleutherAI/pythia-2.8b", device=device)
emails = load_enron_emails(n=10000, dataset_name="snoop2head/enron_aeslc_emails")
prefixes, suffixes = prepare_prefix_suffix_pairs(emails, tokenizer)

n_trials = 100
p_threshold = 0.9

for strategy in ["greedy", "top_k", "top_p", "temperature"]:
    print(f"\nStarting extraction using '{strategy}' strategy...")
    sampling_fn = get_sampling_fn(strategy=strategy)
    results = []
    for i, (prefix, suffix) in enumerate(zip(prefixes, suffixes), start=1):
        result = np_discoverable_extraction(model, tokenizer, prefix, suffix, sampling_fn, n_trials, p_threshold, device)
        results.append(result)
        if i % 100 == 0 or i == len(prefixes):
            print(f"  Processed {i}/{len(prefixes)} email pairs")
    extraction_rate = sum(results) / len(results)
    print(f"Extraction rate for {strategy}: {extraction_rate:.4f}")
