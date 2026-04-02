"""
Extended Memorization Metrics
Measures LLM memorization using three metrics: exact token match, edit similarity, 
and semantic similarity. Runs systematic sweeps across (n, p) parameter space.
"""

import json
import difflib
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

semantic_model = SentenceTransformer('all-MiniLM-L6-v2')


def compute_edit_similarity(s1: str, s2: str) -> float:
    """Compute character-level edit similarity using SequenceMatcher."""
    matcher = difflib.SequenceMatcher(None, s1, s2)
    return matcher.ratio()


def compute_semantic_similarity(s1: str, s2: str) -> float:
    """Compute cosine similarity between sentence embeddings."""
    emb1 = semantic_model.encode(s1, convert_to_tensor=True)
    emb2 = semantic_model.encode(s2, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()


def batch_discoverable_extraction(prefix_text, target_suffix_text, model, tokenizer, sampling_fn, n, mini_batch_size):
    """
    Generate n sequences in mini-batches for a given prefix.
    Returns the fraction whose generated suffix exactly matches the target.
    """
    input_ids = tokenizer(prefix_text, return_tensors="pt").input_ids.to(model.device)
    expected_tokens = tokenizer.encode(target_suffix_text, add_special_tokens=False)

    total_successes = 0
    for i in range(0, n, mini_batch_size):
        current_batch_size = min(mini_batch_size, n - i)
        batch_input_ids = input_ids.repeat(current_batch_size, 1)
        with torch.no_grad():
            outputs = model.generate(
                batch_input_ids,
                max_new_tokens=50,
                **sampling_fn,
                num_return_sequences=current_batch_size
            )
        for out in outputs:
            generated_tokens = out[len(input_ids[0]):].tolist()
            if generated_tokens == expected_tokens:
                total_successes += 1
    return total_successes / n


def check_match_extended(email, model, tokenizer, n, p_threshold, mini_batch_size=30):
    """
    For a given email, split into 50-token prefix and suffix, then measure:
    - Exact match via batched (n,p)-discoverable extraction
    - Edit similarity (SequenceMatcher)
    - Semantic similarity (sentence embeddings)
    """
    tokens = tokenizer.encode(email, add_special_tokens=False)
    if len(tokens) < 100:
        return False, None, None, None, None, None

    prefix_text = tokenizer.decode(tokens[:50], skip_special_tokens=True)
    suffix_text = tokenizer.decode(tokens[50:100], skip_special_tokens=True)

    sampling_fn = {"do_sample": True, "top_p": 0.9}
    frac = batch_discoverable_extraction(prefix_text, suffix_text, model, tokenizer, sampling_fn, n, mini_batch_size)
    match = frac >= p_threshold

    # Generate one sample for similarity metrics
    input_ids = tokenizer(prefix_text, return_tensors="pt", truncation=True, max_length=50).input_ids.to(model.device)
    with torch.no_grad():
        output_ids = model.generate(input_ids=input_ids, max_new_tokens=50, **sampling_fn)[0]
    generated_suffix_text = tokenizer.decode(output_ids[len(input_ids[0]):], skip_special_tokens=True)

    edit_sim = compute_edit_similarity(generated_suffix_text, suffix_text)
    semantic_sim = compute_semantic_similarity(generated_suffix_text, suffix_text)

    return match, prefix_text, suffix_text, generated_suffix_text, edit_sim, semantic_sim


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    MODEL_NAME = "EleutherAI/pythia-2.8b"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

    enron_dataset = load_dataset("jacquelinehe/enron-emails")
    emails = enron_dataset["train"]["text"][:500]

    n_values = [60, 80, 100, 200, 400, 600, 800, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    p_values = [0.1, 0.5, 0.9, 0.99, 0.999]

    EDIT_THRESHOLD = 0.8
    SEMANTIC_THRESHOLD = 0.8

    summary_results = []

    for n in n_values:
        for p in p_values:
            saved_emails = []
            print(f"\n=== Running for n = {n}, p = {p} ===")

            for idx, email in enumerate(emails):
                print(f"\nProcessing email {idx + 1}/{len(emails)} with n = {n} and p = {p}")
                match, prefix_text, suffix_text, gen_text, edit_sim, semantic_sim = check_match_extended(email, model, tokenizer, n, p)

                if match is False and prefix_text is None:
                    continue

                exact_memorized = match
                edit_memorized = edit_sim >= EDIT_THRESHOLD
                semantic_memorized = semantic_sim >= SEMANTIC_THRESHOLD

                if exact_memorized or edit_memorized or semantic_memorized:
                    saved_emails.append({
                        "prefix": prefix_text,
                        "expected_suffix": suffix_text,
                        "generated_suffix": gen_text,
                        "exact_memorized": exact_memorized,
                        "edit_memorized": edit_memorized,
                        "edit_similarity": edit_sim,
                        "semantic_memorized": semantic_memorized,
                        "semantic_similarity": semantic_sim
                    })

                print(f"  Exact: {sum(1 for e in saved_emails if e['exact_memorized'])}, "
                      f"Edit: {sum(1 for e in saved_emails if e['edit_memorized'])}, "
                      f"Semantic: {sum(1 for e in saved_emails if e['semantic_memorized'])} "
                      f"out of {len(emails)}")

            summary = {
                "n": n, "p": p, "total_emails": len(emails),
                "exact_memorized_count": sum(1 for e in saved_emails if e['exact_memorized']),
                "edit_memorized_count": sum(1 for e in saved_emails if e['edit_memorized']),
                "semantic_memorized_count": sum(1 for e in saved_emails if e['semantic_memorized']),
            }
            summary_results.append(summary)

            with open("partial_summary.json", "w") as f:
                json.dump(summary_results, f, indent=2)

    with open("extended_memorization_results.json", "w") as f:
        json.dump(summary_results, f, indent=2)

    print("\nResults saved to 'extended_memorization_results.json'")
