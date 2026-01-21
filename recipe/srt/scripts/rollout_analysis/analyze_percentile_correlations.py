#!/usr/bin/env python3
"""
Analyze Spearman correlations using different percentile aggregations (p25, p50, p75)
for entropy within observation windows.
"""

import json
import argparse
from pathlib import Path
import numpy as np
from scipy import stats

def analyze_correlations(input_path: Path):
    """Analyze correlations using different percentile aggregations."""
    
    with open(input_path) as f:
        data = json.load(f)
    
    results = data['per_sample_results']
    print(f"Loaded {len(results)} samples")
    
    # Check available fields
    sample_keys = list(results[0].keys())
    print(f"Available fields: {sample_keys}")
    
    lengths = [r['response_length'] for r in results]
    
    # Windows to analyze
    windows = [10, 20, 50, 100]
    
    # Aggregation methods
    agg_methods = ['mean', 'median', 'p25', 'p75']
    
    print("\n" + "=" * 80)
    print("SPEARMAN CORRELATIONS: Entropy Aggregation Method vs Output Length")
    print("=" * 80)
    
    # Results table
    header = f"{'Window':<12}" + "".join([f"{m:<12}" for m in agg_methods])
    print(f"\n{header}")
    print("-" * (12 + 12 * len(agg_methods)))
    
    correlation_data = {}
    
    for window in windows:
        row = f"First {window:<5}"
        correlation_data[window] = {}
        
        for method in agg_methods:
            key = f'early_{window}_{method}'
            if key in sample_keys:
                values = [r[key] for r in results]
                rho, p = stats.spearmanr(values, lengths)
                row += f"{rho:>10.4f}  "
                correlation_data[window][method] = {'rho': rho, 'p': p}
            else:
                row += f"{'N/A':>10}  "
                correlation_data[window][method] = None
        
        print(row)
    
    # Also include full response stats
    print("\nFull response:")
    full_methods = ['mean_entropy', 'median_entropy']
    for method in full_methods:
        if method in sample_keys:
            values = [r[method] for r in results]
            rho, p = stats.spearmanr(values, lengths)
            print(f"  {method}: ρ = {rho:.4f}")
    
    # Best method analysis
    print("\n" + "-" * 60)
    print("BEST AGGREGATION METHOD BY WINDOW")
    print("-" * 60)
    
    for window in windows:
        if window in correlation_data:
            best_method = None
            best_rho = 0
            for method, corr in correlation_data[window].items():
                if corr and abs(corr['rho']) > abs(best_rho):
                    best_rho = corr['rho']
                    best_method = method
            print(f"First {window} tokens: Best = {best_method} (ρ = {best_rho:.4f})")
    
    return correlation_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True)
    args = parser.parse_args()
    
    analyze_correlations(Path(args.input_json))


if __name__ == "__main__":
    main()
