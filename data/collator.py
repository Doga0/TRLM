import torch

class TinyDiffusionRecursiveDataCollator:
    def __init__(self, tokenizer, mask_prob_range=(0.1, 1.0)):
        self.tokenizer = tokenizer
        self.mask_prob_range = mask_prob_range
        
    def __call__(self, batch):
        input_ids = torch.stack([b['input_ids'] for b in batch])
        labels = input_ids.clone()
        batch_size, seq_len = input_ids.shape

        # Sample a random masking ratio for each item in the batch
        # This teaches the model to handle both minor edits and full generation
        ratios = torch.empty(batch_size, 1).uniform_(*self.mask_prob_range)
        
        # Create a boolean mask matrix
        rand_matrix = torch.rand(batch_size, seq_len)
        masked_indices = rand_matrix < ratios
        
        # Apply the [MASK] token to the input
        input_ids[masked_indices] = self.tokenizer.mask_token_id
        
        # We only want to compute loss on the masked tokens
        # Set unmasked tokens to -100 so PyTorch CrossEntropy ignores them
        labels[~masked_indices] = -100 

        # This tells the model that all tokens in the sequence are part of the "response"
        response_mask = torch.ones_like(input_ids, dtype=torch.bool)
        
        return {
            "input_ids": input_ids,
            "targets": labels,  # The original unmasked tokens
            "attention_mask": torch.ones_like(input_ids), # bidirectional
            "response_mask": response_mask
        }