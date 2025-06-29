
import os
import numpy as np
import torch

def load_tokens(filename):
    """Load tokens saved as a numpy array.
    Retuns a torch tensor of type long"""
    nptokens = np.load(filename)
    nptokens = nptokens.astype(np.uint16)
    return torch.tensor(nptokens,dtype=torch.long)

class DataLoaderLite:
    def __init__(self, B, T, split, token_dir, process_rank =0, num_processes=1):
        assert split in ["train", "valid"]
        self.B = B
        self.T = T
        self.split = split
        self.shards = []
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.token_dir = token_dir # DATA_TOKENIZED_DIR = "data/tokenized/wikipedia_fr/"
        self.update_shard_list()

    def update_shard_list(self):
        self.shards = sorted([os.path.join(self.token_dir, f) for f in os.listdir(self.token_dir) if f.endswith(".npy")])

        if self.split == "train":
            # remove the last shard
            if len(self.shards) > 1:
                # last shard may not be full
                self.shards.pop()


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
        self.current_token_index = self.B * self.T * self.process_rank
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
        self.current_token_index += self.B * self.T * self.num_processes
        # check if we need to load the next shard
        if self.current_token_index + (self.B * self.T * self.num_processes + 1) > len(self.tokens):
            # check if we ran out of shards
            if self.current_shard_index + 1 >= len(self.shards):
                # try checking if a new shard is available
                # for optimization reasons, we might still be sending new shards to a remote GPU server
                self.update_shard_list()

            # cycle through the shards, enables to continue get batches for more than one epoch
            self.current_shard_index = (self.current_shard_index + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard_index])
            # each process has a different offset in the shard
            # so that they don't overlap
            self.current_token_index = self.B * self.T * self.process_rank
        
        return x, y
    
class IndexedDataLoader:
    def __init__(self, B, T, split, nb_shards, token_dir, process_rank =0, num_processes=1):
        assert split in ["train", "valid"]
        self.B = B
        self.T = T
        self.total_shards = nb_shards
        self.split = split
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.token_dir = token_dir

        self.current_shard_index = -1
        self.processed_shards = []
        self.shard_pool = []
        self.update_shard_pool()
        self.reset()

    def set_total_shards(self, total_shards):
        self.total_shards = total_shards

    def set_processed_shards(self, processed_shards):
        self.processed_shards = processed_shards
        self.update_shard_pool()
        self.reset()

    def update_shard_pool(self):
        # all available shards
        shard_pool = sorted([self.get_index(f) for f in os.listdir(self.token_dir) if f.endswith(".npy")])
        if self.split == "valid":
            self.shard_pool = shard_pool
        else:
            # remove already processed shards
            self.shard_pool = [s for s in shard_pool if s not in self.processed_shards]

    def next_shard(self):
        """Load next shard
        Returns the next shard index or None if no more shards available
        """
        # Add current shard to processed shards if it exists
        if self.current_shard_index != -1:
            self.processed_shards.append(self.current_shard_index)
            # Remove the current shard from the pool
            if self.current_shard_index in self.shard_pool:
                self.shard_pool.remove(self.current_shard_index)

        # Check if we have any more shards to process
        if len(self.shard_pool) == 0:
            self.current_shard_index = -1
            return None
            
        # Get the next shard index (now at position 0 after pop)
        shard_index = self.shard_pool[0]
        
        # Update current shard and load new tokens
        self.current_shard_index = shard_index
        self.current_token_index = self.B * self.T * self.process_rank
        self.tokens = load_tokens(self.get_shard_name(self.current_shard_index))
        
        return shard_index

    def get_state(self):
        return {
            "shard_index": self.current_shard_index,
            "token_index": self.current_token_index,
            "processed_shards": self.processed_shards
        }
    
    def get_shard_index(self)->int:
        """Returns the current shard index"""
        return self.current_shard_index

    def get_shard_name(self, shard_index)->str:
        index = f"{shard_index:06d}"
        sname = f"shard_{index}.npy"
        return os.path.join(self.token_dir, sname)

    def get_index(self, shard_name)->int:
        index = shard_name.split("_")[1].split(".")[0]
        return int(index)

    def __set_state(self, state,fill_processed=False):
        """ Set the state of the dataloader
        Used to resume after a checkpoint
        state : dict of the state 
        fill_processed : if the processed shards we not saved, assume we processed all shards up until current shard index
        """
        self.reset()
        # if key exists
        if "shard_index" in state:
            self.current_shard_index = state["shard_index"]
        if "token_index" in state:
            self.current_token_index = state["token_index"]
        # the ids of the processed shards we saved
        if "processed_shards" in state:
            self.processed_shards = state["processed_shards"]
        # assume shards were processed in order
        else:
            if fill_processed and self.current_shard_index > 0:
                self.processed_shards = list(range(self.current_shard_index))


    def set_state(self, state, fill_processed=False):
        """ Set the state of the dataloader
        Used to resume after a checkpoint
        state : dict of the state 
        fill_processed : if the processed shards we not saved, assume we processed all shards up until current shard index
        """
        # Set processed shards first
        if "processed_shards" in state:
            self.processed_shards = state["processed_shards"]
        elif fill_processed and "shard_index" in state and state["shard_index"] > 0:
            # assume shards were processed in order
            self.processed_shards = list(range(state["shard_index"]))
        
        # Update shard pool based on processed shards
        self.update_shard_pool()
        
        # Set the current shard and token index
        if "shard_index" in state:
            self.current_shard_index = state["shard_index"]
            # CRITICAL: Load the correct shard's tokens
            if self.current_shard_index != -1:
                self.tokens = load_tokens(self.get_shard_name(self.current_shard_index))
        else:
            # If no shard index provided, start from first available shard
            if len(self.shard_pool) > 0:
                self.current_shard_index = self.shard_pool[0]
                self.tokens = load_tokens(self.get_shard_name(self.current_shard_index))
            else:
                self.current_shard_index = -1
        
        # Set token index
        if "token_index" in state:
            self.current_token_index = state["token_index"]
        else:
            # Default offset for this process
            self.current_token_index = self.B * self.T * self.process_rank

    def reset(self):
        if len(self.shard_pool) == 0:
            return False
        self.current_shard_index = self.shard_pool[0]

        # each process has a different offset in the shard
        # so that they don't overlap
        self.current_token_index = self.B * self.T * self.process_rank
        self.tokens = load_tokens(self.get_shard_name(self.current_shard_index))
        return True

    def next_batch(self):
        """Returns 2 batches of tokens of shape (B, T) - input batch and target batch"""
        # check if we need to load the next shard
        if self.current_token_index + (self.B * self.T * self.num_processes + 1) > len(self.tokens):
            pool_available = self.next_shard()
            # shard pool exhausted
            if not pool_available:
                # try refreshing the shard pool
                self.update_shard_pool()
                # if still empty, we are done
                if len(self.shard_pool) == 0:
                    return None, None
                # otherwise, we have a new shard
                else:
                    pool_available = self.next_shard()
                    assert pool_available, "shard pool should not be empty"


        # get B*T tokens + 1 because we need to predict the next token
        buffer = self.tokens[self.current_token_index: self.current_token_index + self.B * self.T+1]
        # get all tokens except the last one
        x = (buffer[:-1]).view(self.B, self.T)
        # target tokens are the ones that follow the input tokens
        # shift the tokens by 1 to the left
        y = (buffer[1:]).view(self.B, self.T)

        # advance index
        self.current_token_index += self.B * self.T * self.num_processes
        
        return x, y
    
if __name__ == "__main__":
    train_loader = IndexedDataLoader(64, 1024, "train", "data/tokenized/tmp/train")
    train_loader.set_processed_shards(list(range(3)))
    while True:
        x, y = train_loader.next_batch()
        if x is None:
            break
    print("done")
