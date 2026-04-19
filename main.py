import dotenv
import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import login

from configs.config import TRLMConfig
from models.trlm import TRLM
from training.trainer import TRLMTrainer
from data.collator import TinyDiffusionRecursiveDataCollator
from data.dataset import (
    OpenWebTextDataset,
    OpenWebTextConfig,
    FineMathDataset,
    FineMathConfig,
    FineWebEduDataset,
    FineWebEduConfig,
    MiniPileDataset,
    MiniPileConfig,
    MixedPretrainDataset
)

dotenv.load_dotenv()
token = os.environ.get('HF_TOKEN')
login(token=token)

tokenizer = AutoTokenizer.from_pretrained("DogaOytc/TRLM")

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
mp_config = MiniPileConfig(max_seq_len=config.max_seq_len)
fm_config = FineMathConfig(max_seq_len=config.max_seq_len)

print("Initializing dataset streams...")
EVAL_SAMPLES_PER_DATASET = 2000

# TRAIN DATASETS 
owt_train = OpenWebTextDataset(tokenizer, owt_config, is_eval=False, eval_size=EVAL_SAMPLES_PER_DATASET)
fwe_train = FineWebEduDataset(tokenizer, fwe_config, is_eval=False, eval_size=EVAL_SAMPLES_PER_DATASET)
mp_train = MiniPileDataset(tokenizer, mp_config, is_eval=False, eval_size=EVAL_SAMPLES_PER_DATASET)
fm_train = FineMathDataset(tokenizer, fm_config, is_eval=False, eval_size=EVAL_SAMPLES_PER_DATASET)

# EVAL DATASETS
owt_eval = OpenWebTextDataset(tokenizer, owt_config, is_eval=True, eval_size=EVAL_SAMPLES_PER_DATASET)
fwe_eval = FineWebEduDataset(tokenizer, fwe_config, is_eval=True, eval_size=EVAL_SAMPLES_PER_DATASET)
mp_eval = MiniPileDataset(tokenizer, mp_config, is_eval=True, eval_size=EVAL_SAMPLES_PER_DATASET)
fm_eval = FineMathDataset(tokenizer, fm_config, is_eval=True, eval_size=EVAL_SAMPLES_PER_DATASET)

dataset_probs = [
        0.30, # OpenWebText (General knowledge)
        0.40, # FineWebEdu (High-quality reasoning)
        0.15, # MiniPile (Diverse domains)
        0.15  # FineMath (Algorithmic & logical reasoning)
    ]

mixed_train_dataset = MixedPretrainDataset(
    datasets=[owt_train, fwe_train, mp_train, fm_train],
    probabilities=dataset_probs
)

# Mix Eval Datasets
mixed_eval_dataset = MixedPretrainDataset(
    datasets=[owt_eval, fwe_eval, mp_eval, fm_eval],
    probabilities=dataset_probs
)

collator = TinyDiffusionRecursiveDataCollator(
        tokenizer=tokenizer,
    )

train_loader = DataLoader(
    mixed_train_dataset,
    batch_size=16,
    collate_fn=collator,
    num_workers=2,
    pin_memory=True
)

eval_loader = DataLoader(
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
    config=config,
    epochs=5,
    warmup_steps=3,
    gradient_accumulation_steps=16
)

print("Starting TRDM Pretraining...")
trainer.train()