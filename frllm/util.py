# ----------------------------------------------------------------------
# Load configuration from file
# ----------------------------------------------------------------------
import configparser
import ast
import os
import math
#import matplotlib.pyplot as plt
import torch
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, split, token_dir):
        assert split in ["train", "valid"]
        self.B = B
        self.T = T
        self.split = split
        self.shards = []
        self.token_dir = token_dir
        self.update_shard_list()
        self.reset()

    def update_shard_list(self):
        self.shards = sorted([os.path.join(self.token_dir, f) for f in os.listdir(self.token_dir) if f.endswith(".npy")])

        if self.split == "train":
            # remove the last shard
            if len(self.shards) > 1:
                # last shard may not be full
                self.shards.pop()

        print(f"found {len(self.shards)} shards for split {self.split}")

    def get_state(self):
        return {
            "shard_index": self.current_shard_index,
            "token_index": self.current_token_index,
        }

    def set_state(self, state):
        self.reset()
        self.current_shard_index = state["shard_index"]
        self.current_token_index = state["token_index"]

    def reset(self):
        self.current_shard_index = 0
        # each process has a different offset in the shard
        # so that they don't overlap
        self.current_token_index = self.B * self.T 
        self.tokens = load_tokens(self.shards[self.current_shard_index])

    def next_batch(self):
        """Returns 2 batches of tokens of shape (B, T) - input batch and target batch"""
        # get B*T tokens + 1 because we need to predict the next token
        buffer = self.tokens[self.current_token_index: self.current_token_index + self.B * self.T+1]
        # get all tokens except the last one
        x = (buffer[:-1]).view(self.B, self.T)
        # target tokens are the ones that follow the input tokens
        # shift the tokens by 1 to the left
        y = (buffer[1:]).view(self.B, self.T)

        # advance index
        self.current_token_index += self.B * self.T 
        # check if we need to load the next shard
        if self.current_token_index + (self.B * self.T+ 1) > len(self.tokens):
            # cycle through the shards, enables to continue get batches for more than one epoch
            self.current_shard_index = (self.current_shard_index + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard_index])
            # each process has a different offset in the shard
            # so that they don't overlap
            self.current_token_index = self.B * self.T 
        
        return x, y

def load_config(config_file="config.txt"):
    config = configparser.ConfigParser()
    config.read(config_file)
    # Convert string values to appropriate Python types
    def parse_value(value):
        try:
            # Try to evaluate as literal (for boolean, None, etc)
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # If it fails, return as string
            return value

    # Create configuration dictionaries
    GPT_CONFIG = {
        key: parse_value(value)
        for key, value in config['model'].items()
    }
    
    HYPERS = {
        key: parse_value(value)
        for key, value in config['hypers'].items()
    }
    
    #
    FILES = {
        key: parse_value(value)
        for key, value in config['files'].items()
    }

    TRAINING = {
        key: parse_value(value)
        for key, value in config['training'].items()
    }
    
    return GPT_CONFIG, HYPERS, FILES, TRAINING

class LogPrinter:
    def __init__(self, log_file):
        # check if parent directory exists
        parent_dir = os.path.dirname(log_file)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        self.log_file = open(log_file,"a")

    def log_print(self, msg):
        print(msg)
        self.log_file.write(msg + "\n")
        self.log_file.flush()

    def close(self):
        self.log_file.close()

def get_lr(step, epoch_steps):
    """Get learning rate"""
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warm_up = epoch_steps // 20 # 5%

    # slow learning rate after one epoch
    if step > epoch_steps:
        return min_lr
    
    # go lineary from min_lr to max_lr
    if step < warm_up:
        return min_lr + (max_lr - min_lr) * step / warm_up
    
    # go from max_lr to min_lr using a cosine function to smooth out the learning rate
    decay_ratio = (step - warm_up) / (epoch_steps - warm_up)
    coefficient = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + (max_lr - min_lr) * coefficient


def plot_loss(stat_file):
    """Plot loss from stat file"""
    with open(stat_file, "r") as f:
        lines = f.readlines()
    
    train_losses = []
    train_steps = []
    validation_losses = []
    validation_steps = []
    
    for line in lines[1:]:  # Skip header line
        epoch, step, loss, lr, is_validation = line.strip().split(",")
        step = int(step)
        loss = float(loss)
        
        if is_validation == "1":
            validation_losses.append(loss)
            validation_steps.append(step)
        else:
            train_losses.append(loss)
            train_steps.append(step)
    
    plt.plot(train_steps, train_losses, label="train")
    plt.plot(validation_steps, validation_losses, label="validation")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    #plt.show()
    plt.savefig("loss-epoch-0.png")

def plot_lr(stat_file):
    """Plot learning rate """
    epoch_steps = 4000
    x = []
    y = []
    for step in range(epoch_steps + 1):
        if step % 10 == 0:
            x.append(step)
            y.append(get_lr(step, epoch_steps))
    import matplotlib.pyplot as plt
    plt.plot(x, y)
    plt.savefig("learning-rate.png")
    # print first 20 values of y
    print(y[:20])

if __name__ == "__main__":
    
    plot_loss("docs/concepts/training/e0-stats.txt")

