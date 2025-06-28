def calculate_model_parameters(config):
    """
    Calculate the total number of parameters in the GPT model
    """
    
    # Extract config values
    vocab_size = config.vocab_size  # 32768
    n_embd = config.n_embd         # 768
    n_head = config.n_head         # 12
    num_kv_groups = config.num_kv_groups  # 4
    n_layer = config.n_layer       # 12
    
    # Derived values
    head_dim = n_embd // n_head    # 768 / 12 = 64
    
    print(f"Model Configuration:")
    print(f"  vocab_size: {vocab_size:,}")
    print(f"  n_embd: {n_embd:,}")
    print(f"  n_head: {n_head}")
    print(f"  num_kv_groups: {num_kv_groups}")
    print(f"  n_layer: {n_layer}")
    print(f"  head_dim: {head_dim}")
    print()
    
    # 1. Token Embeddings (shared with output layer)
    embedding_params = vocab_size * n_embd
    print(f"1. Token Embeddings (wte): {embedding_params:,}")
    
    # 2. Per-layer parameters
    print(f"\n2. Per-layer parameters:")
    
    # GroupQueryAttention parameters
    q_proj_params = n_embd * n_embd  # Q projection: full dimension
    k_proj_params = n_embd * (num_kv_groups * head_dim)  # K projection: reduced
    v_proj_params = n_embd * (num_kv_groups * head_dim)  # V projection: reduced
    c_proj_params = n_embd * n_embd  # Output projection
    
    attention_params = q_proj_params + k_proj_params + v_proj_params + c_proj_params
    print(f"  GroupQueryAttention:")
    print(f"    q_proj: {n_embd} × {n_embd} = {q_proj_params:,}")
    print(f"    k_proj: {n_embd} × {num_kv_groups * head_dim} = {k_proj_params:,}")
    print(f"    v_proj: {n_embd} × {num_kv_groups * head_dim} = {v_proj_params:,}")
    print(f"    c_proj: {n_embd} × {n_embd} = {c_proj_params:,}")
    print(f"    Total attention: {attention_params:,}")
    
    # RMSNorm parameters (2 per layer: pre-attention + pre-feedforward)
    norm_params_per_layer = 2 * n_embd
    print(f"  RMSNorm (2 per layer): 2 × {n_embd} = {norm_params_per_layer:,}")
    
    # FeedForward parameters
    intermediate_size = int(8 * n_embd / 3)  # SwiGLU intermediate size
    ff_c_fc1_params = n_embd * intermediate_size
    ff_c_fc2_params = n_embd * intermediate_size
    ff_c_proj_params = intermediate_size * n_embd
    
    feedforward_params = ff_c_fc1_params + ff_c_fc2_params + ff_c_proj_params
    print(f"  FeedForward (SwiGLU, intermediate_size={intermediate_size:,}):")
    print(f"    c_fc1: {n_embd} × {intermediate_size} = {ff_c_fc1_params:,}")
    print(f"    c_fc2: {n_embd} × {intermediate_size} = {ff_c_fc2_params:,}")
    print(f"    c_proj: {intermediate_size} × {n_embd} = {ff_c_proj_params:,}")
    print(f"    Total feedforward: {feedforward_params:,}")
    
    # Total per layer
    params_per_layer = attention_params + norm_params_per_layer + feedforward_params
    print(f"  Total per layer: {params_per_layer:,}")
    
    # 3. All transformer layers
    all_layers_params = n_layer * params_per_layer
    print(f"\n3. All {n_layer} transformer layers: {all_layers_params:,}")
    
    # 4. Final layer norm
    final_norm_params = n_embd
    print(f"\n4. Final RMSNorm: {final_norm_params:,}")
    
    # 5. Language model head (shared with embeddings, so no additional params)
    lm_head_params = 0  # Shared with wte
    print(f"\n5. Language model head: {lm_head_params:,} (shared with embeddings)")
    
    # Total parameters
    total_params = embedding_params + all_layers_params + final_norm_params + lm_head_params
    
    print(f"\n" + "="*50)
    print(f"TOTAL PARAMETERS: {total_params:,}")
    print(f"TOTAL PARAMETERS: {total_params/1_000_000:.1f}M")
    print(f"="*50)
    
    return total_params

# Your model configuration
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 32768
    n_layer: int = 14
    n_head: int = 16
    num_kv_groups: int = 4
    n_embd: int = 896
    dropout: float = 0.1

def model_sizing():
    
    # Calculate parameters
    config = GPTConfig()
    total_params = calculate_model_parameters(config)

    # Additional memory estimates
    print(f"\nMemory Estimates (FP32):")
    print(f"  Model weights: {total_params * 4 / (1024**3):.2f} GB")
    print(f"  Model weights (FP16): {total_params * 2 / (1024**3):.2f} GB")

    # Compare to standard models
    print(f"\nComparison to standard models:")
    print(f"  Your model: {total_params/1_000_000:.1f}M parameters")
    print(f"  GPT-2 Small: ~124M parameters")
    print(f"  GPT-2 Medium: ~355M parameters")
    print(f"  GPT-2 Large: ~774M parameters")

def analyze_scaling_strategies():
    """
    Analyze different strategies to scale the model to ~150M parameters
    """
    
    # Current configuration
    current_config = {
        'vocab_size': 32768,
        'n_embd': 896,
        'n_head': 14,
        'num_kv_groups': 4,
        'n_layer': 12
    }
    
    current_params = 100_682_496
    target_params = 150_000_000
    additional_needed = target_params - current_params
    
    print(f"Current parameters: {current_params:,}")
    print(f"Target parameters: {target_params:,}")
    print(f"Additional needed: {additional_needed:,}")
    print("="*60)
    
    strategies = []
    
    # Strategy 1: Increase embedding dimension
    print("STRATEGY 1: Increase Embedding Dimension")
    print("-" * 40)
    
    for new_n_embd in [896, 1024, 1152]:
        new_config = current_config.copy()
        new_config['n_embd'] = new_n_embd
        new_config['n_head'] = 16 if new_n_embd >= 1024 else 14  # Keep head_dim = 64
        
        params = calculate_params_quick(new_config)
        print(f"  n_embd={new_n_embd}, n_head={new_config['n_head']}: {params:,} ({params/1_000_000:.1f}M)")
        strategies.append(('Increase n_embd', new_config, params))
    
    print()
    
    # Strategy 2: Add more layers
    print("STRATEGY 2: Add More Layers")
    print("-" * 30)
    
    for new_layers in [14, 16, 18]:
        new_config = current_config.copy()
        new_config['n_layer'] = new_layers
        
        params = calculate_params_quick(new_config)
        print(f"  n_layer={new_layers}: {params:,} ({params/1_000_000:.1f}M)")
        strategies.append(('Add layers', new_config, params))
    
    print()
    
    # Strategy 3: Reduce GQA groups (more KV heads)
    print("STRATEGY 3: Reduce GQA Groups (More KV Heads)")
    print("-" * 45)
    
    for new_kv_groups in [6, 12]:  # 12 = full MHA
        new_config = current_config.copy()
        new_config['num_kv_groups'] = new_kv_groups
        
        params = calculate_params_quick(new_config)
        mha_status = " (Full MHA)" if new_kv_groups == 12 else ""
        print(f"  num_kv_groups={new_kv_groups}{mha_status}: {params:,} ({params/1_000_000:.1f}M)")
        strategies.append(('Reduce GQA', new_config, params))
    
    print()
    
    # Strategy 4: Hybrid approaches
    print("STRATEGY 4: Hybrid Approaches")
    print("-" * 30)
    
    # Hybrid 1: Slightly increase n_embd + add layers
    hybrid1 = current_config.copy()
    hybrid1['n_embd'] = 896
    hybrid1['n_head'] = 14
    hybrid1['n_layer'] = 14
    params1 = calculate_params_quick(hybrid1)
    print(f"  n_embd=896, n_head=14, n_layer=14: {params1:,} ({params1/1_000_000:.1f}M)")
    strategies.append(('Hybrid 1', hybrid1, params1))
    
    # Hybrid 2: Moderate n_embd increase + less GQA
    hybrid2 = current_config.copy()
    hybrid2['n_embd'] = 896
    hybrid2['n_head'] = 14
    hybrid2['num_kv_groups'] = 7
    params2 = calculate_params_quick(hybrid2)
    print(f"  n_embd=896, n_head=14, num_kv_groups=7: {params2:,} ({params2/1_000_000:.1f}M)")
    strategies.append(('Hybrid 2', hybrid2, params2))
    
    print()
    
    # Find closest to target
    print("RECOMMENDATIONS:")
    print("="*50)
    
    # Sort by distance to target
    strategies.sort(key=lambda x: abs(x[2] - target_params))
    
    print("Best options (closest to 150M):")
    for i, (name, config, params) in enumerate(strategies[:3]):
        distance = abs(params - target_params)
        print(f"{i+1}. {name}: {params:,} ({params/1_000_000:.1f}M)")
        print(f"   Distance from target: {distance:,}")
        print(f"   Config: {config}")
        print()
    
    return strategies

def calculate_params_quick(config):
    """Quick parameter calculation"""
    vocab_size = config['vocab_size']
    n_embd = config['n_embd']
    n_head = config['n_head']
    num_kv_groups = config['num_kv_groups']
    n_layer = config['n_layer']
    
    head_dim = n_embd // n_head
    
    # Embeddings (shared)
    embedding_params = vocab_size * n_embd
    
    # Per layer
    # Attention
    q_proj = n_embd * n_embd
    k_proj = n_embd * (num_kv_groups * head_dim)
    v_proj = n_embd * (num_kv_groups * head_dim)
    c_proj = n_embd * n_embd
    attention_params = q_proj + k_proj + v_proj + c_proj
    
    # LayerNorms
    norm_params = 2 * n_embd
    
    # FeedForward (SwiGLU)
    intermediate_size = int(8 * n_embd / 3)
    ff_params = 3 * n_embd * intermediate_size
    
    params_per_layer = attention_params + norm_params + ff_params
    all_layers = n_layer * params_per_layer
    
    # Final norm
    final_norm = n_embd
    
    total = embedding_params + all_layers + final_norm
    return total

def test_scaling_strategies():
    # Run the analysis
    strategies = analyze_scaling_strategies()

    print("\nDETAILED RECOMMENDATIONS:")
    print("="*60)
    print("1. BEST CHOICE: n_embd=896, n_head=14, n_layer=14")
    print("   - Balanced scaling across width and depth")
    print("   - Maintains efficient head_dim=64")
    print("   - Good compute/memory tradeoff")
    print()
    print("2. ALTERNATIVE: n_embd=1024, n_head=16 (keep 12 layers)")
    print("   - Standard power-of-2 embedding dimension")
    print("   - Easier to optimize and more common")
    print("   - Slightly larger than target but reasonable")
    print()
    print("3. CONSERVATIVE: Add 4 layers (n_layer=16)")
    print("   - Minimal changes to existing config")
    print("   - Depth scaling often works well")
    print("   - Stays close to current architecture")


model_sizing()