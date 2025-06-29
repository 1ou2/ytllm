

# Datasets
https://www.kaggle.com/datasets/wikimedia-foundation/wikipedia-structured-contents
Download frwiki_namespace_0_0.jsonl
Le texte des articles se trouve dans dans un champ `value` qui est de ```"type": "paragraph"```
Exemple
```
"type": "paragraph", "value": "Esochí (grec moderne : Εσοχή) est une localité située dans le dème d'Arrianá, dans le district régional de Rhodope, dans la periphérie de Macédoine-Orientale-et-Thrace, en Grèce."
```
Les paragraphes sont dans des sections.
```
"sections": [{"type": "section", "name": "Abstract", "has_parts": ["type": "paragraph", "value": "Esochí etc.. ", "links": [{"url": ""}]]}]
```

Sur le premier fichier jsonl 222815 article dont 219378 qui contiennent du texte
Processing frwiki_namespace_0_1.jsonl
228626it [00:52, 4319.86it/s] 
Processing articles: 100%|██████████████████████████████████████████████████████████████| 228626/228626 [00:28<00:00, 8028.42it/s]
Total articles processed: 225138
Processing frwiki_namespace_0_2.jsonl
222434it [02:03, 1793.83it/s] 
Processing articles: 100%|█████████████████████████████████████████████████████████████| 222434/222434 [00:15<00:00, 13946.37it/s]
Total articles processed: 219107
Processing frwiki_namespace_0_3.jsonl
229869it [00:47, 4867.66it/s] 
Processing articles: 100%|█████████████████████████████████████████████████████████████| 229869/229869 [00:17<00:00, 13266.65it/s]
Total articles processed: 226240
Processing frwiki_namespace_0_4.jsonl
228082it [03:54, 972.76it/s]  
Processing articles: 100%|█████████████████████████████████████████████████████████████| 228082/228082 [00:16<00:00, 14142.49it/s]
Total articles processed: 224689
# OSCAR

# install
uv init

# deepseek mixed precision

1.Precision Distribution:
FP8 precision is used for:
- Feed-forward network (FFN) computations
- Key (K) and Value (V) projections in attention
- Attention output projections
Higher precision (bfloat16) is maintained for:
- Query (Q) projections
- Layer normalization
- Embedding layers
- Final classifier layer
2. Implementation Details:
Split the attention projection into separate Q, K, V projections to control precision
Wrapped FP8-specific computations in selective_precision_autocast
Kept layer normalization operations in higher precision
Used FP8 for all FFN computations
3.Important Notes:
The actual FP8 implementation requires NVIDIA's Transformer Engine or a similar library that provides FP8 tensor operations
The selective_precision_autocast context manager is currently a placeholder that needs to be integrated with the actual FP8 implementation
The DeepSeek paper's approach helps maintain model quality while reducing memory usage and increasing training speed
4.Memory and Performance Benefits:
Using FP8 for the specified layers reduces memory usage
The mixed precision approach maintains model quality by keeping critical operations in higher precision
The attention mechanism's key and value operations benefit from FP8 without significant accuracy loss

To fully implement this in practice, you would need to:
Install and integrate NVIDIA's Transformer Engine or a similar library that provides FP8 support


pip install transformer-engine

The key changes I've made to implement FP8 training are:
Added NVIDIA's Transformer Engine imports for FP8 support
Implemented a proper selective_precision_autocast context manager that uses Transformer Engine's FP8 autocast
Replaced standard PyTorch Linear layers with Transformer Engine layers in specific components:
Key (K) and Value (V) projections in attention
Output projection in attention
All layers in the Feed Forward Network
The implementation now uses:
FP8 format e-4m3 (4-bit exponent, 3-bit mantissa) for forward pass
FP8 format e-5m2 (5-bit exponent, 2-bit mantissa) for backward pass
Regular precision (bfloat16) for Query (Q) projections and layer norms
To use this implementation on your RTX 4090:
Make sure you have the latest NVIDIA drivers installed
The code will automatically use FP8 for the specified layers when running on CUDA
For layers not using FP8, it will fall back to bfloat16 as configured
The hybrid precision approach follows the DeepSeek paper's methodology:
Critical operations (Q projections, layer norms) stay in higher precision
Compute-intensive operations (K/V projections, FFN) use FP8
Automatic fallback to bfloat16 for non-CUDA devices


# Flash attention
pip install flash-attn --no-build-isolation

Flash Attention expects (Batch, Time, Heads, Head_size)
Regular attention expects (Batch, Heads, Time, Head_size)


# hugging face


# Dataset Card for French Wikipedia Text Corpus
## Dataset Description
The French Wikipedia Text Corpus is a comprehensive dataset derived from French Wikipedia articles. It is specifically designed for training language models (LLMs). The dataset contains the text of paragraphs from Wikipedia articles, with sections, footnotes, and titles removed to provide a clean and continuous text stream.
Dataset Details
Features

    text: A single attribute containing the full text of Wikipedia articles as a continuous chunk of text.

## Languages
The dataset is in French.

## Dataset statistics:
Train set: 2539940 examples
Validation set: 133682 examples

## Dataset Creation
The dataset was created by extracting and processing text from French Wikipedia articles. The processing involved removing article sections, footnotes, and titles to ensure that the text is in a format suitable for training language models.

## Intended Uses
This dataset is intended for use in training and evaluating language models. The clean and continuous text format makes it particularly suitable for models that require large volumes of coherent text data.

## Example Usage
```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset('1ou2/fr_wiki_paragraphs')

# Access the training data
train_data = dataset['train']

# Print an example
print(train_data[0]['text'])
```

## Citation Information

If you use this dataset in your research, please cite it as follows:

@misc{french_wikipedia_text_corpus,
  author = {Gabriel Pastor},
  title = {French Wikipedia Text Corpus},
  year = {2025},
  publisher = {Hugging Face},
  journal = {Hugging Face Datasets},
  howpublished = {\url{https://huggingface.co/datasets/1ou2/fr_wiki_paragraphs}}
}

## License

This dataset is licensed under the CC BY-SA 3.0 license, in accordance with Wikipedia's content license.


# gutenberg
https://dev.gutenberg.org/browse/scores/top-fr.php

pour arsène lupin séparateur est ------

Filtre auteur FR
{'Mérimée', 'Zola', 'Musset', 'Leroux', 'Verne', 'Leblanc', 'Diderot', 'Dumas', 'Proust', 'Voltaire', 'Gautier', 'Stendhal', 'France', 'Balzac', 'Montesquieu', 'Maupassant', 'Daudet'}

# datasources
books : 18
news : 459
wikipedia : 1352

# taille du modèle

Pour 250M
vocab_size = 32768
context_length = 1024
embed_dim = 1024
n_layers = 16
n_heads = 16


# Datasets
Fineweb2 https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
https://arxiv.org/pdf/2506.20920

# Optimisation
NeoBert : https://arxiv.org/pdf/2502.19587
