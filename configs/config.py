from dataclasses import dataclass

@dataclass
class TRLMConfig:
    # model params
    vocab_size: int = 32000
    hidden_dim: int = 512
    num_heads: int = 8
    num_groups: int = 2
    max_seq_len: int = 1024
    num_layers: int = 2
    rms_norm_eps: float = 1e-6
    attn: str = "gqa"
    attn_dropout: float = 0.1
    transformer_dropout: float = 0.1

    mask_token_id: int = -1

    # MLP (SwiGLU)
    swiglu_intermediate_size: int = 2048
    swiglu_hidden_act: str = "silu"

    # TRM recursion params
    max_supervision_steps: int = 16
    thinking_steps: int = 2
    max_act_steps: int = 16
    n_sup: int = 8
    halt_threshold: float = 0.7

    # logging and saving
    use_tensorboard: bool = True
    save_total_limit: int = 5
    log_dir: str = "/content/drive/MyDrive/TRM/TRLM/logs"
    logging_steps: int = 1
    checkpoint_dir: str = "/content/drive/MyDrive/TRM/TRLM/checkpoints"
    push_to_hub: bool = True
    hf_repo_id: str = "DogaOytc/TRLM"

    # evaluation
    lambda_halt: float = 1.0
    eval_generation: bool = True
    eval_interval: int = 50
    eval_steps: int = 500
    save_steps: int = 50
    eval_samples_per_step: int = 2000

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
    max_steps_per_epoch: int = 10000 
    gradient_accumulation_steps: int = 16
    q_loss_weight: float = 0.5
    warmup_steps: int = None
    warmup_ratio: int = 0.1
    epochs: int = 1

    # ema
    ema_decay: float = 0.999

    def get_warmup_steps(self, actual_steps_per_epoch: int = None):
        steps_per_epoch = actual_steps_per_epoch or self.max_steps_per_epoch
        total_opt_steps = (self.epochs * steps_per_epoch) // self.gradient_accumulation_steps
        return int(total_opt_steps * self.warmup_ratio)
