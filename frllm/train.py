"""Train an LLM model"""
import os
from util import load_config, DataLoaderLite, LogPrinter
from model import GPTModel
import torch
import random
from datetime import datetime
# ------------------
# device selection
# ------------------
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"using device: {device}")

device_name = device
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)  # 0 is the GPU index

torch.manual_seed(4321) # seed for CPU
random.seed(4321) # seed for Python random module
if torch.cuda.is_available():
    torch.cuda.manual_seed(4321) # seed for GPU


GPT_CONFIG, HYPERS, FILES, TRAINING = load_config()
logger = LogPrinter(FILES["log_dir"] + FILES["log_file"])
stats_file = open(FILES["log_dir"] + FILES["stat_file"], "a")
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logger.log_print(f"--------------------------------------------")
logger.log_print(f"Training started at {current_time}")
B = TRAINING["micro_batch_size"]
T = GPT_CONFIG["context_length"]


nb_shards = TRAINING["n_shards"]
# each step we process a fixed sized batch
batch_size = TRAINING["batch_size"]

grad_accum_steps = batch_size // (B * T)
# we are sample +1 token for y
# tokens_per_shard = 1048577 = 2^20+1
epoch_train_steps = ((TRAINING["tokens_per_shard"]-1) // (batch_size)) * nb_shards
warmup_steps = epoch_train_steps // 20 # 5% of one epoch

logger.log_print(f"B: {B} | T: {T} | B*T : {B * T } tokens | Batch per shards: {(TRAINING["tokens_per_shard"]-1)// TRAINING["batch_size"]}")
# Big optimization here !
# change format from float32 to tf32 (tensor float32)
# this causes a little loss on precision but 8x througput on GPU processing and x3 in memory
# -> x3 performance increase
torch.set_float32_matmul_precision('high')


# load tokenizer
from tokenizers import ByteLevelBPETokenizer
special_tokens = ["<|endoftext|>", "<|user|>", "<|bot|>", "<|sys|>","<|gab1|>", "<|gab2|>", "<|gab3|>","<|gab4|>", "<|gab5|>"]

# Load the trained tokenizer
tokenizer = ByteLevelBPETokenizer(
    FILES["tokenizer_dir"] + "gabgpt-vocab.json",
    FILES["tokenizer_dir"] + "gabgpt-merges.txt"
)

# Add special tokens to the loaded tokenizer
tokenizer.add_special_tokens(special_tokens)