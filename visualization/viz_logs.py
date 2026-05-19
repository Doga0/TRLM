import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_log(file_path):
    train_data = []
    eval_data = []

    # Updated regular expressions to capture the new metrics and account for missing spaces
    train_pattern = re.compile(
        r"Step (\d+) \| Loss: ([\d.]+) \| Lm_loss: ([\d.]+) \| Ppl: ([\d.]+) \| Grad norm: ([\d.]+) \| Depth: ([\d.]+) \| Tokens: ([\d.]+)"
    )
    
    # \s* is used to safely handle cases where the log string has no spaces (e.g. "...0.0001Consistency...")
    eval_pattern = re.compile(
        r"Eval Step (\d+) \| Loss: ([\d.]+) \| Masked PPL: ([\d.]+) \| Acc_Long: ([\d.]+) \| Acc_Short: ([\d.]+) \| Acc_Diff: ([\d.]+)\s*Consistency:\s*([\d.]+)\s*Depth:\s*([\d.]+)"
    )

    with open(file_path, 'r') as f:
        for line in f:
            # Parse Training Steps
            train_match = train_pattern.search(line)
            if train_match:
                train_data.append({
                    'Step': int(train_match.group(1)),
                    'Loss': float(train_match.group(2)),
                    'Lm_loss': float(train_match.group(3)),
                    'Ppl': float(train_match.group(4)),
                    'Grad_norm': float(train_match.group(5)),
                    'Depth': float(train_match.group(6)),
                    'Tokens': float(train_match.group(7))
                })
            
            # Parse Evaluation Steps
            eval_match = eval_pattern.search(line)
            if eval_match:
                eval_data.append({
                    'Step': int(eval_match.group(1)),
                    'Eval_Loss': float(eval_match.group(2)),
                    'Eval_PPL': float(eval_match.group(3)),
                    'Eval_Acc_Long': float(eval_match.group(4)),
                    'Eval_Acc_Short': float(eval_match.group(5)),
                    'Eval_Acc_Diff': float(eval_match.group(6)),
                    'Eval_Consistency': float(eval_match.group(7)),
                    'Eval_Depth': float(eval_match.group(8))
                })

    return pd.DataFrame(train_data), pd.DataFrame(eval_data)

def plot_metrics(df_train, df_eval):
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle('TRDM Pretraining Metrics', fontsize=20, fontweight='bold')

    if not df_train.empty:
        # 1. Training Loss & LM Loss
        axes[0, 0].plot(df_train['Step'], df_train['Loss'], label='Total Loss', color='blue')
        axes[0, 0].plot(df_train['Step'], df_train['Lm_loss'], label='LM Loss', color='cyan', linestyle='--')
        axes[0, 0].set_title('Training Losses over Steps')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 2. Training Perplexity
        axes[0, 1].plot(df_train['Step'], df_train['Ppl'], label='Train PPL', color='purple')
        axes[0, 1].set_title('Training Perplexity over Steps')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Perplexity')
        axes[0, 1].grid(True)

        # 3. Training Tokens Processed (Replaced Q_loss)
        axes[1, 0].plot(df_train['Step'], df_train['Tokens'], label='Tokens', color='orange')
        axes[1, 0].set_title('Tokens Processed over Steps')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Tokens')
        axes[1, 0].grid(True)

        # 4. Training Gradient Norm
        axes[1, 1].plot(df_train['Step'], df_train['Grad_norm'], label='Grad Norm', color='red')
        axes[1, 1].set_title('Gradient Norm over Steps')
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Grad Norm')
        axes[1, 1].grid(True)

    if not df_eval.empty:
        # 5. Evaluation Loss
        axes[2, 0].plot(df_eval['Step'], df_eval['Eval_Loss'], label='Eval Loss', color='darkblue', marker='o')
        axes[2, 0].set_title('Evaluation Loss over Steps')
        axes[2, 0].set_xlabel('Steps')
        axes[2, 0].set_ylabel('Eval Loss')
        axes[2, 0].legend()
        axes[2, 0].grid(True)

        # 6. Evaluation Perplexity
        axes[2, 1].plot(df_eval['Step'], df_eval['Eval_PPL'], label='Masked PPL', color='magenta', marker='o')
        axes[2, 1].set_title('Evaluation Masked PPL over Steps')
        axes[2, 1].set_xlabel('Steps')
        axes[2, 1].set_ylabel('Eval PPL')
        axes[2, 1].legend()
        axes[2, 1].grid(True)

        # 7. Evaluation Accuracy (Long vs Short)
        axes[3, 0].plot(df_eval['Step'], df_eval['Eval_Acc_Long'], label='Acc Long', color='green', marker='o')
        axes[3, 0].plot(df_eval['Step'], df_eval['Eval_Acc_Short'], label='Acc Short', color='limegreen', marker='x', linestyle='--')
        axes[3, 0].set_title('Evaluation Accuracy over Steps')
        axes[3, 0].set_xlabel('Steps')
        axes[3, 0].set_ylabel('Accuracy')
        axes[3, 0].legend()
        axes[3, 0].grid(True)

        # 8. Evaluation Consistency (Replaced Halt)
        axes[3, 1].plot(df_eval['Step'], df_eval['Eval_Consistency'], label='Eval Consistency', color='brown', marker='o')
        axes[3, 1].set_title('Evaluation Consistency over Steps')
        axes[3, 1].set_xlabel('Steps')
        axes[3, 1].set_ylabel('Consistency')
        axes[3, 1].legend()
        axes[3, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    try:
        df_train, df_eval = parse_log('C:/Users/USER/Desktop/DOA/projects/bitirme/TRDLM/logs/2/training.log')
        plot_metrics(df_train, df_eval)
    except FileNotFoundError:
        print("Error: 'training.log' not found. Please save the log output to a file named 'training.log'.")