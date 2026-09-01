"""
Measure tokenization statistics: <unk> rates and tokens-per-sequence
"""

import sys
sys.path.insert(0, 'src')

from pathlib import Path
from models.bpe import BPETokenizer
from dataset import load_packed_split
import json


def analyze_tokenization(
    splits_dir: str = "data.nosync/splits_packed",
    tokenizers_dir: str = "tokenizers"
):
    """
    Analyze tokenization statistics across all splits.
    
    Measures:
    - Tokens per sequence (min, max, mean, median)
    - Vocabulary coverage (% of tokens that are in-vocab)
    - Token frequency distribution
    """
    
    print("="*70)
    print("TOKENIZATION ANALYSIS")
    print("="*70)
    
    # Load tokenizers
    print("\nLoading tokenizers...")
    encoder = BPETokenizer.load(str(Path(tokenizers_dir) / "encoder.json"))
    decoder = BPETokenizer.load(str(Path(tokenizers_dir) / "decoder.json"))
    
    print(f"✓ Encoder: vocab_size={len(encoder.vocab)}, merges={len(encoder.merges)}")
    print(f"✓ Decoder: vocab_size={len(decoder.vocab)}, merges={len(decoder.merges)}")
    
    results = {}
    
    # Analyze each split
    for split_name in ['train', 'val', 'test']:
        print(f"\n{'='*70}")
        print(f"ANALYZING {split_name.upper()} SPLIT")
        print(f"{'='*70}")
        
        # Load split
        cipher_bytes, plain_text = load_packed_split(
            splits_dir=splits_dir,
            split_name=split_name
        )
        
        print(f"Loaded {len(cipher_bytes)} pairs\n")
        
        # Analyze encoder (cipher)
        print(f"ENCODER TOKENIZER (cipher bytes):")
        print("-" * 70)
        
        encoder_stats = analyze_tokenizer(
            tokenizer=encoder,
            data=[bytes_obj.decode('latin-1') for bytes_obj in cipher_bytes],
            data_name="cipher",
            vocab_size=len(encoder.vocab)
        )
        
        # Analyze decoder (plaintext)
        print(f"\nDECODER TOKENIZER (plaintext):")
        print("-" * 70)
        
        decoder_stats = analyze_tokenizer(
            tokenizer=decoder,
            data=plain_text,
            data_name="plaintext",
            vocab_size=len(decoder.vocab)
        )
        
        results[split_name] = {
            'encoder': encoder_stats,
            'decoder': decoder_stats
        }
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print_summary_table(results)
    
    # Save results
    results_path = Path("tokenization_stats.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {results_path}")
    
    return results


def analyze_tokenizer(tokenizer, data, data_name, vocab_size):
    """
    Analyze a single tokenizer on a dataset.
    
    Returns:
        dict with statistics
    """
    
    token_counts = []
    all_tokens = []
    oov_count = 0
    total_tokens = 0
    
    for item in data:
        tokens = tokenizer.encode(item)
        token_counts.append(len(tokens))
        all_tokens.extend(tokens)
        
        # Count OOV tokens (tokens >= vocab_size)
        oov = sum(1 for t in tokens if t >= vocab_size)
        oov_count += oov
        total_tokens += len(tokens)
    
    # Calculate statistics
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)
    mean_tokens = sum(token_counts) / len(token_counts)
    median_tokens = sorted(token_counts)[len(token_counts) // 2]
    
    oov_rate = 100 * oov_count / total_tokens if total_tokens > 0 else 0
    
    # Token frequency analysis
    token_freq = {}
    for token in all_tokens:
        token_freq[token] = token_freq.get(token, 0) + 1
    
    most_common_tokens = sorted(token_freq.items(), key=lambda x: -x[1])[:5]
    
    print(f"\n  Tokens per sequence:")
    print(f"    Min:    {min_tokens}")
    print(f"    Max:    {max_tokens}")
    print(f"    Mean:   {mean_tokens:.2f}")
    print(f"    Median: {median_tokens}")
    print(f"    Std:    {calculate_std(token_counts):.2f}")
    
    print(f"\n  Token vocabulary coverage:")
    print(f"    Total tokens in corpus: {total_tokens:,}")
    print(f"    Unique tokens used: {len(token_freq)}")
    print(f"    Vocab size: {vocab_size}")
    print(f"    Coverage: {100 * len(token_freq) / vocab_size:.1f}%")
    print(f"    <unk> tokens: {oov_count:,} ({oov_rate:.2f}%)")
    
    print(f"\n  Most frequent tokens:")
    for token_id, freq in most_common_tokens:
        token_bytes = tokenizer.vocab.get(token_id, b'<unknown>')
        freq_pct = 100 * freq / total_tokens
        print(f"    Token {token_id:4d}: {freq:8,} times ({freq_pct:5.2f}%) → {token_bytes!r}")
    
    # Compute percentiles
    sorted_counts = sorted(token_counts)
    p10 = sorted_counts[int(len(sorted_counts) * 0.1)]
    p25 = sorted_counts[int(len(sorted_counts) * 0.25)]
    p75 = sorted_counts[int(len(sorted_counts) * 0.75)]
    p90 = sorted_counts[int(len(sorted_counts) * 0.9)]
    
    print(f"\n  Percentiles (tokens per sequence):")
    print(f"    P10: {p10}")
    print(f"    P25: {p25}")
    print(f"    P75: {p75}")
    print(f"    P90: {p90}")
    
    return {
        'tokens_per_sequence': {
            'min': min_tokens,
            'max': max_tokens,
            'mean': mean_tokens,
            'median': median_tokens,
            'std': calculate_std(token_counts),
            'p10': p10,
            'p25': p25,
            'p75': p75,
            'p90': p90
        },
        'vocabulary': {
            'total_tokens': total_tokens,
            'unique_tokens': len(token_freq),
            'vocab_size': vocab_size,
            'coverage_pct': 100 * len(token_freq) / vocab_size,
            'oov_count': oov_count,
            'oov_rate_pct': oov_rate
        },
        'most_frequent_tokens': [
            {'token_id': tid, 'frequency': freq, 'frequency_pct': 100*freq/total_tokens}
            for tid, freq in most_common_tokens
        ]
    }


def calculate_std(values):
    """Calculate standard deviation"""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


def print_summary_table(results):
    """Print a nice summary table"""
    
    print("\nTOKENS PER SEQUENCE (summary):")
    print("-" * 100)
    print(f"{'Split':<10} {'Component':<10} {'Min':<8} {'Mean':<10} {'Median':<10} {'Max':<8} {'Std':<10}")
    print("-" * 100)
    
    for split_name in ['train', 'val', 'test']:
        split_data = results[split_name]
        
        for component in ['encoder', 'decoder']:
            comp_data = split_data[component]
            tokens = comp_data['tokens_per_sequence']
            
            print(f"{split_name:<10} {component:<10} {tokens['min']:<8} {tokens['mean']:<10.2f} "
                  f"{tokens['median']:<10} {tokens['max']:<8} {tokens['std']:<10.2f}")
    
    print("\n\n<UNK> TOKEN RATES (summary):")
    print("-" * 100)
    print(f"{'Split':<10} {'Component':<10} {'Total Tokens':<15} {'<unk> Count':<15} {'<unk> Rate %':<15}")
    print("-" * 100)
    
    for split_name in ['train', 'val', 'test']:
        split_data = results[split_name]
        
        for component in ['encoder', 'decoder']:
            comp_data = split_data[component]
            vocab = comp_data['vocabulary']
            
            print(f"{split_name:<10} {component:<10} {vocab['total_tokens']:<15,} "
                  f"{vocab['oov_count']:<15,} {vocab['oov_rate_pct']:<15.4f}")
    
    print("\n\nVOCABULARY COVERAGE (summary):")
    print("-" * 100)
    print(f"{'Split':<10} {'Component':<10} {'Vocab Size':<15} {'Unique Used':<15} {'Coverage %':<15}")
    print("-" * 100)
    
    for split_name in ['train', 'val', 'test']:
        split_data = results[split_name]
        
        for component in ['encoder', 'decoder']:
            comp_data = split_data[component]
            vocab = comp_data['vocabulary']
            
            print(f"{split_name:<10} {component:<10} {vocab['vocab_size']:<15} "
                  f"{vocab['unique_tokens']:<15} {vocab['coverage_pct']:<15.2f}")


if __name__ == "__main__":
    results = analyze_tokenization(
        splits_dir="data.nosync/splits_packed",
        tokenizers_dir="tokenizers"
    )
