from dataclasses import dataclass

@dataclass
class TRLMConfig:
    # model params
    # vocab_size: int = 50257
    hidden_dim: int = 512
    num_heads: int = 8
    num_groups: int = 2
    max_seq_len: int = 1024
    num_layers: int = 12
    rms_norm_eps: float = 1e-6
    attn: str = "gqa"
    attn_dropout: float = 0.0
    transformer_dropout: float = 0.0
    step_dropout: float = 0.1

    mask_token_id: int = 2

    # MLP (SwiGLU)
    swiglu_intermediate_size: int = 2048
    swiglu_hidden_act: str = "silu"

    # TRM recursion params
    thinking_steps: int = 4
    max_recursions: int = 16
    min_recursions: int = 1

    # logging and saving
    use_tensorboard: bool = True
    save_total_limit: int = 5
    log_dir: str = "/content/drive/MyDrive/TRM/TRLM/exp4/logs"
    logging_steps: int = 1
    checkpoint_dir: str = "/content/drive/MyDrive/TRM/TRLM/exp4/checkpoints"
    push_to_hub: bool = True
    hf_repo_id: str = "DogaOytc/TRLM"

    # evaluation
    eval_generation: bool = True
    eval_interval: int = 50
    eval_steps: int = 500
    save_steps: int = 50
    eval_samples_per_step: int = 4000

    # optimizer
    muon_lr: float = 0.001
    muon_momentum: float = 0.95 
    muon_beta2: float = 0.95 
    muon_ns_steps: int = 5
    learning_rate: float = 3e-4
    adamw_betas: tuple[float, float] = (0.9, 0.95)
    adamw_eps: float = 1e-5
    weight_decay: float = 0.01 

    # train
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 20
    save_every_epoch: bool = True
    max_steps_per_epoch: int = 100000 
    gradient_accumulation_steps: int = 16
    warmup_steps: int = None
    warmup_ratio: float = 0.1
    epochs: int = 1

    refinement_probability: float = 0.5
    refinement_loss_weight: float = 0.5
    eval_refinement_remask_ratio: float = 0.3

    # ema
    ema_decay: float = 0.999

    # token-budget training
    max_train_tokens: int = 1_000_000_000
    warmup_ratio: float = 0.03

    log_token_interval: int = 100_000
    eval_token_interval: int = 10_000_000
    save_token_interval: int = 250_000_000

    # masking
    mlm_probability: float = 0.20
    mask_random_prob: float = 0.50
    mask_span_prob: float = 0.30
    mask_continuation_prob: float = 0.20

    def get_warmup_steps(self, actual_steps_per_epoch: int = None):
        steps_per_epoch = actual_steps_per_epoch or self.max_steps_per_epoch
        total_opt_steps = (self.epochs * steps_per_epoch) // self.gradient_accumulation_steps
        return int(total_opt_steps * self.warmup_ratio)
