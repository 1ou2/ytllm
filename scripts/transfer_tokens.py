import os 
import sys
from dotenv import load_dotenv
token_root_dir = "data/tokenized/"
token_dirs = ["gutenberg", "news", "wikipedia"]
target_dir = "data/shards/"

def cp_shards(token_dirs, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    # tokens are stored in shard files (*.npy)
    # each shard file contains a list of tokens
    # shards can have the same name between dirs
    # copy shards to target dir, but make sure to rename them
    # to avoid overwriting
    shard_id = 0
    for token_dir in token_dirs:
        for shard in sorted(os.listdir(token_root_dir + token_dir)):
            shard_path = token_root_dir + token_dir + "/" + shard
            # shard_000000.npy
            target_path = target_dir + "shard_"+ str(shard_id).zfill(6) + ".npy"
            print("copying {} to {}".format(shard_path, target_path))
            os.system("cp {} {}".format(shard_path, target_path))
            shard_id += 1

def package_shards():
    # create a tarball of the target dir
    os.system("tar -cvf data/shards.tar data/shards/")

    # split the tarball into 10 shards
    os.system("split -b 200M data/shards.tar tokens_part")

def download_shards():
    load_dotenv()
    remote_dir = os.getenv("AWS_DIR")
    # raise error if remote_dir not set
    if remote_dir is None or not remote_dir:
        raise ValueError("remote dir AWS_DIR not found")

    for suffix in ["aa", "ab", "ac", "ad", "ae", "af", "ag", "ah", "ai", "aj", "ak", "al", "am", "an", "ao", "ap", "aq"]:
        remote_path = f"https://s3.us-west-2.amazonaws.com/{remote_dir}/tokens_part{suffix}"
        # download shards from s3
        os.system(f"wget {remote_path}")

if __name__ == "__main__":
    #cp_shards(token_dirs, target_dir)
    #package_shards()
    download_shards()


