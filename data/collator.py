import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class MaskingCollator:
    tokenizer: Any

    mlm_probability: float = 0.20

    random_prob: float = 0.50
    span_prob: float = 0.30
    continuation_prob: float = 0.20

    mean_span_length: int = 4
    max_span_length: int = 12

    mask_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None

    def __post_init__(self):
        if self.mask_token_id is None:
            self.mask_token_id = self.tokenizer.mask_token_id

        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.pad_token_id

        if self.mask_token_id is None:
            raise ValueError("Tokenizer must have a mask_token_id.")

        if self.pad_token_id is None:
            raise ValueError("Tokenizer must have a pad_token_id.")

        total = self.random_prob + self.span_prob + self.continuation_prob

        self.random_prob /= total
        self.span_prob /= total
        self.continuation_prob /= total

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        features = []

        for ex in examples:
            input_ids = ex["input_ids"]

            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()

            features.append({"input_ids": input_ids})

        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", input_ids.ne(self.pad_token_id).long())

        labels = torch.full_like(input_ids, fill_value=-100)

        for i in range(input_ids.size(0)):
            valid_positions = self._get_valid_positions(input_ids[i], attention_mask[i])

            if valid_positions.numel() == 0:
                continue

            mode = self._sample_mode()

            if mode == "random":
                mask_positions = self._random_mask_positions(valid_positions)

            elif mode == "span":
                mask_positions = self._span_mask_positions(valid_positions)

            else:
                mask_positions = self._continuation_mask_positions(valid_positions)

            if mask_positions.numel() == 0:
                continue

            labels[i, mask_positions] = input_ids[i, mask_positions]
            input_ids[i, mask_positions] = self.mask_token_id

        batch["masked_input_ids"] = input_ids
        batch["labels"] = labels
        batch["attention_mask"] = attention_mask

        return batch

    def _sample_mode(self) -> str:
        r = random.random()

        if r < self.random_prob:
            return "random"

        if r < self.random_prob + self.span_prob:
            return "span"

        return "continuation"

    def _get_valid_positions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = attention_mask.bool()

        special_ids = {
            self.pad_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.bos_token_id,
            self.tokenizer.mask_token_id,
            self.tokenizer.unk_token_id,
        }

        for token_id in special_ids:
            if token_id is not None:
                valid &= input_ids.ne(int(token_id))

        return torch.nonzero(valid, as_tuple=False).squeeze(-1)

    def _target_count(self, valid_positions: torch.Tensor) -> int:
        n_valid = valid_positions.numel()

        if n_valid <= 0:
            return 0

        count = int(round(n_valid * self.mlm_probability))
        count = max(1, count)
        count = min(count, n_valid)

        return count

    def _random_mask_positions(self, valid_positions: torch.Tensor) -> torch.Tensor:
        n_to_mask = self._target_count(valid_positions)

        if n_to_mask <= 0:
            return valid_positions[:0]

        perm = torch.randperm(valid_positions.numel(), device=valid_positions.device)
        selected = valid_positions[perm[:n_to_mask]]

        return selected.sort().values

    def _span_mask_positions(self, valid_positions: torch.Tensor) -> torch.Tensor:
        n_to_mask = self._target_count(valid_positions)

        if n_to_mask <= 0:
            return valid_positions[:0]

        selected = set()
        valid_list = valid_positions.tolist()

        attempts = 0
        max_attempts = 100

        while len(selected) < n_to_mask and attempts < max_attempts:
            attempts += 1

            start_idx = random.randint(0, len(valid_list) - 1)
            span_len = random.randint(1, self.max_span_length)

            for j in range(start_idx, min(start_idx + span_len, len(valid_list))):
                selected.add(valid_list[j])

                if len(selected) >= n_to_mask:
                    break

        if not selected:
            return valid_positions[:0]

        selected_tensor = torch.tensor(
            sorted(selected),
            dtype=torch.long,
            device=valid_positions.device,
        )

        return selected_tensor

    def _continuation_mask_positions(self, valid_positions: torch.Tensor) -> torch.Tensor:
        n_to_mask = self._target_count(valid_positions)

        if n_to_mask <= 0:
            return valid_positions[:0]

        # Half suffix continuation, half prefix infilling.
        if random.random() < 0.5:
            # Suffix-style: keep prefix, mask the last n tokens.
            selected = valid_positions[-n_to_mask:]
        else:
            # Prefix-style: keep suffix, mask the first n tokens.
            selected = valid_positions[:n_to_mask]

        return selected.sort().values