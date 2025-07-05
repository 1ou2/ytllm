import os 
import sys
from dotenv import load_dotenv
import random
import numpy as np
import glob
from tqdm import tqdm

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

def download_shards(max_nb_shards=1000):
    """Download shards from AWS
    Args
    max_nb_shards: maximum number of shards to download"""
    load_dotenv()
    remote_dir = os.getenv("AWS_DIR")
    # raise error if remote_dir not set
    if remote_dir is None or not remote_dir:
        raise ValueError("remote dir AWS_DIR not found")


    for suffix in ["aa", "ab", "ac", "ad", "ae", "af", "ag", "ah", "ai", "aj", "ak", "al", "am", "an", "ao", "ap", "aq", "ar", "as"]:
        remote_path = f"https://s3.us-west-2.amazonaws.com/{remote_dir}/shards_{suffix}"
        # download shards from s3
        os.system(f"wget {remote_path}")
        max_nb_shards -= 1
        if max_nb_shards == 0:
            break

def combine_shards(input_dir:str=".", file_prefix:str="shards_"):
    """Combine shards into a single file"""
    # get all shards
    shards = os.listdir(input_dir)
    # sort shards
    shards.sort()
    # combine shards
    output_file = input_dir + "/shards.tar"
    # use cat system command, as shards were created using split
    all_shards = " ".join([input_dir + "/" + shard for shard in shards])
    print(f"Combining files : {all_shards}")
    os.system(f"cat {all_shards} > {output_file}")
    print(f"Output: {output_file}")

def shuffle_shards(shard_dir="data/shards/train"):
    # get all shards
    shards = os.listdir(shard_dir)
    # shuffle shards
    random.shuffle(shards)

    # First pass: rename to temporary names
    for i, shard in enumerate(shards):
        temp_name = f"temp_{i:06d}.npy"
        os.rename(os.path.join(shard_dir, shard), os.path.join(shard_dir, temp_name))
    
    # Second pass: rename to final names
    for i in range(len(shards)):
        temp_name = f"temp_{i:06d}.npy"
        final_name = f"shard_{i:06d}.npy"
        os.rename(os.path.join(shard_dir, temp_name), os.path.join(shard_dir, final_name))

def dummy_shards():
    # create 100 shards containing a single number
    os.makedirs("data/shards/test", exist_ok=True)
    for i in range(100):
        with open(f"data/shards/test/shard_{i:06d}.npy", "w") as f:
            f.write(str(i))


def concatenate_npy_shards(input_dir, output_file, dtype=np.uint16):
    """
    Concatenate multiple .npy files into a single binary file.
    
    Args:
        input_dir: Directory containing .npy files
        output_file: Output binary file path
        dtype: Data type for output (np.uint16 for nanoGPT)
    """
    
    # Find all .npy files
    print(os.path.join(input_dir))
    npy_files = glob.glob(os.path.join(input_dir, "*.npy"))
    npy_files.sort()  # Ensure consistent ordering
    
    print(f"Found {len(npy_files)} .npy files")
    
    if len(npy_files) == 0:
        raise ValueError(f"No .npy files found in {input_dir}")
    
    # Check first file to understand the data
    first_array = np.load(npy_files[0])
    print(f"First file shape: {first_array.shape}")
    print(f"First file dtype: {first_array.dtype}")
    print(f"Expected total tokens: {len(npy_files) * len(first_array):,}")
    
    # Memory-efficient concatenation using memory mapping
    total_tokens = 0
    
    # First pass: count total tokens
    print("Counting total tokens...")
    for npy_file in tqdm(npy_files):
        arr = np.load(npy_file)
        total_tokens += len(arr)
    
    print(f"Total tokens: {total_tokens:,}")
    
    # Create memory-mapped output file
    print(f"Creating output file: {output_file}")
    output_array = np.memmap(output_file, dtype=dtype, mode='w+', shape=(total_tokens,))
    
    # Second pass: copy data
    print("Copying data...")
    offset = 0
    for npy_file in tqdm(npy_files):
        arr = np.load(npy_file)
        
        # Convert to target dtype if needed
        if arr.dtype != dtype:
            arr = arr.astype(dtype)
        
        # Copy to output
        output_array[offset:offset + len(arr)] = arr
        offset += len(arr)
    
    # Ensure data is written to disk
    del output_array
    
    print(f"Successfully created {output_file}")
    print(f"Final size: {total_tokens:,} tokens ({total_tokens * np.dtype(dtype).itemsize / 1e9:.2f} GB)")


if __name__ == "__main__":

    # argument parsing
    if len(sys.argv) > 1:
        if sys.argv[1] == "package":
            package_shards()
        elif sys.argv[1] == "download":
            max_shards = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
            download_shards(max_shards)
        elif sys.argv[1] == "shuffle":
            shuffle_shards("data/shards/test")
        elif sys.argv[1] == "concatenate":
            input_dir = sys.argv[2] if len(sys.argv) > 2 else "data/shards/test"
            output_file = sys.argv[3] if len(sys.argv) > 3 else "data/shards/test.bin"
            concatenate_npy_shards(input_dir, output_file)
        elif sys.argv[1] == "combine":
            input_dir = sys.argv[2] if len(sys.argv) > 2 else "."
            file_prefix = sys.argv[3] if len(sys.argv) > 3 else "shards_"
            combine_shards(input_dir,file_prefix)
        elif sys.argv[1] == "dummy":
            dummy_shards()
        else:
            print("Unknown argument")
            sys.exit(1)
        sys.exit(0)
    else:
        print("No argument provided.")
        print("Usage: python transfer_tokens.py <package|download|shuffle|combine|concatenate>")
        # add usage with additional args for download and concatenate
        print("\nAdditional args: ")
        print(" - download max_nb_shards")
        print(" - combine input_dir file_prefix")
        print(" - concatenate input_dir output_file")
        print("\nExamples:")
        print(" - python transfer_tokens.py package")
        print(" - python transfer_tokens.py download 3")
        print(" - python transfer_tokens.py concatenate data/shards/test data/shards/test.bin")
        sys.exit(1)



