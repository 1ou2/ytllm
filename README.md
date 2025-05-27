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

OSCAR

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
