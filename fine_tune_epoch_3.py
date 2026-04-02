import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW  # Use PyTorch's AdamW
from accelerate import Accelerator  # Critical for memory management
from tqdm import tqdm

# CONTINUE TRAINING ON BASE OLMo-7B MODEL WITH SYNTHETIC DATASET, 3 EPOCH

class SyntheticDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.block_size,
            padding="max_length",
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in tokenized.items()}  # Remove batch dim

def main():
    # Load data
    data_df = pd.read_csv("raw_text_data.csv", header=None, names=['text'])
    texts = data_df['text'].astype(str).str.strip().tolist()

    # Initialize accelerator first
    accelerator = Accelerator(gradient_accumulation_steps=4)
    
    # Load model with memory optimizations
    model = AutoModelForCausalLM.from_pretrained(
        "allenai/OLMo-7B",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16  # Use bfloat16 for better stability
    )
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Critical optimizations
    model.gradient_checkpointing_enable()
    model.tie_weights()  # Resolves "weights not tied" warning

    # Configuration
    block_size = 500
    batch_size = 1  # Must use 1 with device_map="auto"
    learning_rate = 1e-5

    # Prepare components with accelerate
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    dataset = SyntheticDataset(texts, tokenizer, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

    # Training loop
    model.train()
    for epoch in range(3):
        accelerator.print(f"Epoch {epoch+1}/1")
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1} progress", leave=False)):
            with accelerator.accumulate(model):
                inputs = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch.get("attention_mask"),
                    "labels": batch["input_ids"]
                        }
                outputs = model(**inputs)
                loss = outputs.loss
                accelerator.backward(loss)
                
                # Optimizer step with gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad()

            # Memory cleanup
            if step % 10 == 0:
                torch.cuda.empty_cache()
                accelerator.print(f"Step {step} | Loss: {loss.item():.4f}")

    # Save model
    accelerator.wait_for_everyone()
    accelerator.save_model(model, "./fine_tuned_olmo7B_epoch_3", safe_serialization=True)

if __name__ == "__main__":
    # Set critical environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    main()

