import dotenv
import os
import torch
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import login

from configs.config import TRLMConfig
from models.trlm import TRLM
from training.trainer import TRLMTrainer
from data.collator import MaskingCollator
from data.dataset import (
    OpenWebTextDataset,
    OpenWebTextConfig,
    FineWebEduDataset,
    FineWebEduConfig,
    MixedPretrainDataset
)

dotenv.load_dotenv()
token = os.environ.get('HF_TOKEN')
login(token=token)

tokenizer = AutoTokenizer.from_pretrained("DogaOytc/TRLM-tokenizer")

if tokenizer.mask_token is None:
    tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

config = TRLMConfig()

config.vocab_size = len(tokenizer)
config.mask_token_id = tokenizer.mask_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TRLM(config=config)
model = model.to(device)
print(f"Model moved to: {device}")

owt_config = OpenWebTextConfig(max_seq_len=config.max_seq_len)
fwe_config = FineWebEduConfig(max_seq_len=config.max_seq_len)

print("Initializing datasets...")

# TRAIN DATASETS 
owt_train = OpenWebTextDataset(
    tokenizer=tokenizer, config=owt_config, 
    is_eval=False, 
    eval_size=config.eval_samples_per_step
)
fwe_train = FineWebEduDataset(
    tokenizer=tokenizer, 
    config=fwe_config, 
    is_eval=False, 
    eval_size=config.eval_samples_per_step
)


# EVAL DATASETS
owt_eval = OpenWebTextDataset(
    tokenizer=tokenizer, 
    config=owt_config, 
    is_eval=True, 
    eval_size=config.eval_samples_per_step
)
fwe_eval = FineWebEduDataset(
    tokenizer=tokenizer, 
    config=fwe_config, 
    is_eval=True, 
    eval_size=config.eval_samples_per_step
)


dataset_probs = [
        0.50, # OpenWebText (General knowledge)
        0.50, # FineWebEdu (High-quality reasoning)
    ]

mixed_train_dataset = MixedPretrainDataset(
    datasets=[owt_train, fwe_train],
    probabilities=dataset_probs
)

# Mix Eval Datasets
mixed_eval_dataset = MixedPretrainDataset(
    datasets=[owt_eval, fwe_eval],
    probabilities=dataset_probs
)

collator = MaskingCollator(
    tokenizer=tokenizer,
    mlm_probability=config.mlm_probability,
    random_prob=config.mask_random_prob,
    span_prob=config.mask_span_prob,
    continuation_prob=config.mask_continuation_prob,
    mean_span_length=4,
    max_span_length=12,
)

train_loader = StatefulDataLoader(
    mixed_train_dataset,
    batch_size=16,
    collate_fn=collator,
    num_workers=2,
    pin_memory=True
)

eval_loader = StatefulDataLoader(
    mixed_eval_dataset,
    batch_size=16,
    collate_fn=collator,
    num_workers=2,
    pin_memory=True
)

trainer = TRLMTrainer(
    model=model,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    config=config
)

print("Starting TRLM Pretraining...")
trainer.train()

if config.push_to_hub:
    export_dir = os.path.join(config.checkpoint_dir, "hf_export")
    final_ckpt = os.path.join(config.checkpoint_dir, "checkpoint_final.pt")

    trainer.export_for_hf(
        export_dir=export_dir, 
        checkpoint_path=final_ckpt, 
        tokenizer=tokenizer
    )
    trainer.push_to_hf(
        export_dir=export_dir, 
        repo_id=config.hf_repo_id
    )

print("\nFINISHED!!!")