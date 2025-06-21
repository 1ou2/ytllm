# ----------------------------------------------------------------------
# Load configuration from file
# ----------------------------------------------------------------------
import configparser
import ast
import os
import math
import matplotlib.pyplot as plt

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

