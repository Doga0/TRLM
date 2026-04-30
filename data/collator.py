import torch

class MaskingCollator:
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def __call__(self, batch):
        if isinstance(batch[0], dict) and "input_ids" in batch[0]:
            input_ids = [torch.as_tensor(item["input_ids"], dtype=torch.long) for item in batch]
        else:
            input_ids = [torch.as_tensor(item, dtype=torch.long) for item in batch]

        # Pad the sequences to the maximum length in this batch
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        labels = input_ids.clone()
        
        # Create a probability matrix for masking
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        # Prevent masking of padding tokens
        padding_mask = (labels == self.tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
        
        # Sample which tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Labels for unmasked tokens should be -100
        labels[~masked_indices] = -100 
        
        # Replace the chosen tokens in input_ids with the [MASK] token
        input_ids[masked_indices] = self.tokenizer.mask_token_id

        return {
            "masked_input_ids": input_ids, 
            "labels": labels
        }