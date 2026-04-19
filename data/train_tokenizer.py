import logging
import os
import sys
from typing import Iterator, List, Optional

from datasets import load_dataset
from tokenizers import (
    Tokenizer, 
    decoders, 
    models, 
    pre_tokenizers, 
    processors, 
    trainers
)
from tokenizers.pre_tokenizers import ByteLevel, Digits, Sequence
from transformers import PreTrainedTokenizerFast, AutoTokenizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger("TRLM-Tokenizer")


class TRLMTokenizerManager:
    def __init__(self, vocab_size: int = 32768, repo_id: str = "DogaOytc/TRLM"):
        self.vocab_size = vocab_size
        self.repo_id = repo_id
        self.raw_path = "./tokenizer_data/trlm_tokenizer.json"
        self.hf_save_dir = "./trlm_tokenizer_hf"
        self.special_tokens = ["<|endoftext|>", "<|pad|>", "<|mask|>"]
        
        self.tokenizer = Tokenizer(models.BPE())
        self._setup_pipeline()

    def _setup_pipeline(self):
        """Defines the normalization and pre-tokenization rules."""

        self.tokenizer.pre_tokenizer = Sequence([
            ByteLevel(add_prefix_space=False),
            Digits(individual_digits=True)
        ])

        self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
        self.tokenizer.decoder = decoders.ByteLevel()

    def _get_batch_iterator(self, samples_per_dataset: int) -> Iterator[str]:
        """Generator for training data."""

        dataset_configs = [
            ("HuggingFaceFW/fineweb-edu", "sample-10BT", "FineWeb-Edu"),
            ("JeanKaddour/minipile", None, "MiniPile"),
            ("HuggingFaceTB/finemath", "finemath-4plus", "FineMath"),
            ("Skylion007/openwebtext", None, "OpenWebText")
        ]

        total_yielded = 0

        for path, subset, name in dataset_configs:
            logger.info(f"⬇️  Connecting to {name}...")

            try:
                ds = load_dataset(path, name=subset, split="train", streaming=True)

                this_ds_count = 0
                for item in ds:
                    text = item.get("text", "")
                    if len(text) > 10:
                        yield text
                        this_ds_count += 1
                        total_yielded += 1

                    if this_ds_count >= samples_per_dataset:
                        break

                logger.info(f"   ✅ Fetched {this_ds_count} examples from {name}")

            except Exception as e:
                logger.error(f"   ❌ FAILED to load {name}. Skipping... Error: {e}")
                continue 

        if total_yielded == 0:
            raise RuntimeError("CRITICAL: No data was loaded from ANY dataset.")
        else:
            logger.info(f"✨ Total examples ready for training: {total_yielded}")

    def train(self, samples_per_source: int = 50000):
        """Trains the BPE model on the specified datasets."""

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
        
        logger.info("Starting training...")
        self.tokenizer.train_from_iterator(
            self._get_batch_iterator(samples_per_source), 
            trainer=trainer
        )
        
        os.makedirs(os.path.dirname(self.raw_path), exist_ok=True)
        self.tokenizer.save(self.raw_path)
        logger.info(f"Raw tokenizer saved to {self.raw_path}")

    def convert_and_upload(self, push: bool = True):
        """Wraps the raw tokenizer for Transformers and pushes to Hub."""

        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer,
            bos_token="<|endoftext|>",
            eos_token="<|endoftext|>",
            unk_token="<|endoftext|>",
            pad_token="<|pad|>",
            mask_token="<|mask|>"
        )
        
        fast_tokenizer.save_pretrained(self.hf_save_dir)

        if push:
            fast_tokenizer.push_to_hub(self.repo_id)
            logger.info(f"Pushed to {self.repo_id}")

    @staticmethod
    def run_tests(tokenizer_id: str):

        print(f"--- Loading Tokenizer from: {tokenizer_id} ---")
    
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

        sample_text = "Solve for x: 14592 + 873 = 15465."
        
        inputs = tokenizer(sample_text, add_special_tokens=True)
        
        print(f"\n[Test 1]")
        print(f"Input Text:  {sample_text}")
        print(f"Token IDs:   {inputs['input_ids']}")
        print(f"Tokens:      {tokenizer.convert_ids_to_tokens(inputs['input_ids'])}")
        
        decoded = tokenizer.decode(inputs['input_ids'])
        print(f"Decoded:     {decoded}")

        print(f"\n[Test 2: Special Tokens]")
        print(f"PAD token:  {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        print(f"EOS token:  {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
        print(f"MASK token: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")

        print(f"\n[Test 3: Batch Padding]")
        batch_sentences = [
            "Short sentence.",
            "A much longer sentence to test if padding works correctly."
        ]
        
        batch_encoded = tokenizer(
            batch_sentences, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        
        print(f"Padded Input IDs:\n{batch_encoded['input_ids']}")
        print(f"Attention Mask:\n{batch_encoded['attention_mask']}")

if __name__ == "__main__":
    
    repo_id = "DogaOytc/TRLM"

    manager = TRLMTokenizerManager(repo_id=repo_id)
    
    manager.train(samples_per_source=50000)
    
    manager.convert_and_upload(push=True)
    
    manager.run_tests(repo_id)
