import json
from datetime import datetime
import matplotlib.pyplot as plt
import os

class TrainingStats:
    def __init__(self):
        # Main data structure: dictionary with steps as keys
        self.stats = {}
        self.metadata = {
            'start_time': datetime.now().isoformat(),
            'last_update': None
        }
    
    def update(self, step, loss=None, lr=None, generated_text=None, 
              tokens_processed=None, shard_index=None):
        """
        Update statistics for a specific step.
        Creates the step entry if it doesn't exist, updates it if it does.
        """
        # Initialize step data if it doesn't exist
        if step not in self.stats:
            self.stats[step] = {}
        
        # Update timestamp
        self.stats[step]['timestamp'] = datetime.now().isoformat()
        
        # Update metrics if provided
        if loss is not None:
            self.stats[step]['loss'] = float(loss)
        if lr is not None:
            self.stats[step]['learning_rate'] = float(lr)
        if generated_text is not None:
            self.stats[step]['generated_text'] = generated_text
        if tokens_processed is not None:
            self.stats[step]['tokens_processed'] = tokens_processed
        if shard_index is not None:
            self.stats[step]['shard_index'] = int(shard_index)
        
        self.metadata['last_update'] = datetime.now().isoformat()
    
    def save_stats(self, save_dir):
        """Save stats to a JSON file"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"{save_dir}/training_stats_{timestamp}.json"
        
        data = {
            'metadata': self.metadata,
            'stats': self.stats
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def load_stats(self, filepath):
        """Load stats from a JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.stats = data['stats']
            self.metadata = data.get('metadata', {})
    
    def get_all_steps(self):
        """Return sorted list of all steps"""
        return sorted([int(step) for step in self.stats.keys()])
    
    def get_metric_history(self, metric):
        """
        Get the history of a specific metric across all steps
        Returns: (steps, values) tuple
        """
        steps = []
        values = []
        for step in sorted([int(s) for s in self.stats.keys()]):
            if metric in self.stats[str(step)]:
                steps.append(step)
                values.append(self.stats[str(step)][metric])
        return steps, values
    
    def plot_loss_curve(self, save_path=None):
        """Plot the loss curve"""
        steps, losses = self.get_metric_history('loss')
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses)
        plt.title('Training Loss Over Time')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def get_generations(self):
        """Get all text generations with their corresponding steps"""
        generations = []
        for step in self.get_all_steps():
            if 'generated_text' in self.stats[str(step)]:
                generations.append({
                    'step': step,
                    'text': self.stats[str(step)]['generated_text'],
                    'loss': self.stats[str(step)].get('loss'),
                    'timestamp': self.stats[str(step)]['timestamp']
                })
        return generations
    
    def get_latest_stats(self):
        """Get the most recent statistics"""
        if not self.stats:
            return None
        
        latest_step = str(max(int(step) for step in self.stats.keys()))
        latest = self.stats[latest_step]
        
        return {
            'step': int(latest_step),
            'loss': latest.get('loss'),
            'learning_rate': latest.get('learning_rate'),
            'shard_index': latest.get('shard_index'),
            'timestamp': latest.get('timestamp')
        }
