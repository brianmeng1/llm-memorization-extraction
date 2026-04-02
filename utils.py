import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def setup_device():
    """Initialize and return the compute device."""
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU count:", torch.cuda.device_count())
        print("GPU name:", torch.cuda.get_device_name(0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load_model(model_name="EleutherAI/pythia-2.8b", device=None, device_map=None):
    """Load a HuggingFace model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device_map:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
    elif device:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    return model, tokenizer


def prepare_prefix_suffix_pairs(emails, tokenizer, prefix_len=50, suffix_len=50):
    """Split emails into prefix/suffix token pairs for extraction testing."""
    total_len = prefix_len + suffix_len
    prefixes = []
    suffixes = []
    for email in emails:
        tokens = tokenizer.encode(email, truncation=True, max_length=total_len)
        if len(tokens) >= total_len:
            prefixes.append(tokenizer.decode(tokens[:prefix_len]))
            suffixes.append(tokenizer.decode(tokens[prefix_len:total_len]))
    print(f"Prepared {len(prefixes)} prefix/suffix pairs from {len(emails)} emails.")
    return prefixes, suffixes


def load_enron_emails(n=10000, dataset_name="jacquelinehe/enron-emails"):
    """Load Enron email dataset."""
    dataset = load_dataset(dataset_name)
    emails = dataset["train"]["text"][:n]
    print(f"Loaded {len(emails)} emails from {dataset_name}.")
    return emails


def get_sampling_fn(strategy="top_k"):
    """Return sampling parameters for a given decoding strategy."""
    strategies = {
        "greedy": {},
        "top_k": {"do_sample": True, "top_k": 40},
        "top_p": {"do_sample": True, "top_p": 0.9},
        "temperature": {"do_sample": True, "temperature": 1.0},
    }
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(strategies.keys())}")
    return strategies[strategy]


def discoverable_extraction(model, tokenizer, prefix, target_suffix, sampling_fn, device=None):
    """
    Generate a suffix from a prefix and check if it matches the target.
    Returns True if the target suffix appears in the generated text.
    """
    inputs = tokenizer(prefix, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    if device:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=len(input_ids[0]) + 50,
            pad_token_id=tokenizer.eos_token_id,
            **sampling_fn
        )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return target_suffix in generated_text


def np_discoverable_extraction(model, tokenizer, prefix, target_suffix, sampling_fn, n, p, device=None):
    """
    (n, p)-discoverable extraction as defined in the literature.
    Returns True if at least fraction p of n generation attempts reproduce the target suffix.
    """
    successes = 0
    for _ in range(n):
        if discoverable_extraction(model, tokenizer, prefix, target_suffix, sampling_fn, device):
            successes += 1
        if successes / n >= p:
            return True
    return False
