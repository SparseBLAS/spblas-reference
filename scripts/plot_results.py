#!/usr/bin/env python3
"""
Performance plotting script for SPMM benchmark results.
Parses MKL and SYCL experiment results and creates comparison plots.
"""

import re
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

@dataclass
class ExperimentResult:
    """Container for experiment parameters and results."""
    m: int
    k: int 
    n: int
    nnz_row: int
    gb_per_s: float
    gflops: float
    method: Optional[str] = None  # Only for SYCL
    wg_size: Optional[int] = None  # Only for SYCL

def parse_mkl_results(filename: str) -> List[ExperimentResult]:
    """Parse MKL benchmark results from output file."""
    results = []
    current_params = None
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Parse command line to extract parameters
            if line.startswith('./spmm_benchmark'):
                parts = line.split()
                if len(parts) >= 5:
                    m = int(parts[1])
                    k = int(parts[2])
                    n = int(parts[3])
                    nnz_row = int(parts[4])
                    current_params = (m, k, n, nnz_row)
            
            # Parse median results
            elif line.startswith('Median duration') and current_params:
                # Extract GB/s from line like: "Median duration 0.00017339 (80.74285714285715 GB/s) 17.70% of peak"
                gb_match = re.search(r'\(([0-9.]+) GB/s\)', line)
                if gb_match:
                    gb_per_s = float(gb_match.group(1))
                    
                    # Look for the corresponding GFLOPs line (should be next)
                    continue
            
            elif line.startswith('Median achieved') and current_params:
                # Extract GFLOPs from line like: "Median achieved 18.455504931080224 GFLOPs"
                gflops_match = re.search(r'Median achieved ([0-9.]+) GFLOPs', line)
                if gflops_match and gb_per_s:
                    gflops = float(gflops_match.group(1))
                    
                    m, k, n, nnz_row = current_params
                    results.append(ExperimentResult(
                        m=m, k=k, n=n, nnz_row=nnz_row,
                        gb_per_s=gb_per_s, gflops=gflops
                    ))
                    current_params = None  # Reset for next experiment
    
    return results

def parse_sycl_results(filename: str) -> List[ExperimentResult]:
    """Parse SYCL benchmark results from output file."""
    results = []
    current_params = None
    gb_per_s = None
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Parse command line to extract parameters
            if line.startswith('./sycl_spmm'):
                parts = line.split()
                if len(parts) >= 7:
                    m = int(parts[1])
                    k = int(parts[2])
                    n = int(parts[3])
                    nnz_row = int(parts[4])
                    method = parts[5]
                    wg_size = int(parts[6])
                    current_params = (m, k, n, nnz_row, method, wg_size)
            
            # Parse median results
            elif line.startswith('Median duration') and current_params:
                # Extract GB/s from line like: "Median duration 0.000161946 (86.44859397576971 GB/s) 18.95802499468634% of peak."
                gb_match = re.search(r'\(([0-9.]+) GB/s\)', line)
                if gb_match:
                    gb_per_s = float(gb_match.group(1))
            
            elif line.startswith('Median achieved') and current_params and gb_per_s:
                # Extract GFLOPs from line like: "Median achieved 19.759672977412226 GFLOPs"
                gflops_match = re.search(r'Median achieved ([0-9.]+) GFLOPs', line)
                if gflops_match:
                    gflops = float(gflops_match.group(1))
                    
                    m, k, n, nnz_row, method, wg_size = current_params
                    results.append(ExperimentResult(
                        m=m, k=k, n=n, nnz_row=nnz_row,
                        gb_per_s=gb_per_s, gflops=gflops,
                        method=method, wg_size=wg_size
                    ))
                    current_params = None  # Reset for next experiment
                    gb_per_s = None
    
    return results

def find_best_sycl_performance(sycl_results: List[ExperimentResult]) -> Dict[Tuple[int, int, int, int], Tuple[ExperimentResult, str]]:
    """
    Find the best SYCL performance for each (m, k, n, nnz_row) combination.
    Returns dict mapping parameters to (best_result, "method-wg_size" label).
    """
    # Group by parameters
    param_groups = defaultdict(list)
    for result in sycl_results:
        key = (result.m, result.k, result.n, result.nnz_row)
        param_groups[key].append(result)
    
    # Find best performance for each parameter combination
    best_results = {}
    for params, results in param_groups.items():
        best_result = max(results, key=lambda r: r.gb_per_s)
        label = f"{best_result.method}-{best_result.wg_size}"
        best_results[params] = (best_result, label)
    
    return best_results

def create_plots(mkl_results: List[ExperimentResult], sycl_results: List[ExperimentResult]):
    """Create performance plots for each n value."""
    # Get all unique n values
    all_n_values = set()
    for result in mkl_results + sycl_results:
        all_n_values.add(result.n)
    
    # Find best SYCL performance for each parameter combination
    best_sycl = find_best_sycl_performance(sycl_results)
    
    # Create plots for each n value
    for n in sorted(all_n_values):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Filter results for this n value
        mkl_n_results = [r for r in mkl_results if r.n == n]
        sycl_n_results = [(params, result, label) for params, (result, label) in best_sycl.items() if params[2] == n]
        
        if not mkl_n_results and not sycl_n_results:
            continue
            
        # Prepare MKL data
        mkl_nnz_rows = [r.nnz_row for r in mkl_n_results]
        mkl_gb_per_s = [r.gb_per_s for r in mkl_n_results]
        
        # Prepare SYCL data
        sycl_nnz_rows = [params[3] for params, result, label in sycl_n_results]
        sycl_gb_per_s = [result.gb_per_s for params, result, label in sycl_n_results]
        sycl_labels = [label for params, result, label in sycl_n_results]
        
        # Plot lines
        if mkl_nnz_rows:
            # Sort MKL data by nnz_row for proper line plotting
            mkl_sorted = sorted(zip(mkl_nnz_rows, mkl_gb_per_s))
            mkl_nnz_rows_sorted, mkl_gb_per_s_sorted = zip(*mkl_sorted)
            ax.plot(mkl_nnz_rows_sorted, mkl_gb_per_s_sorted, '-o', label='MKL', markerfacecolor='white')
        
        if sycl_nnz_rows:
            # Sort SYCL data by nnz_row for proper line plotting
            sycl_sorted = sorted(zip(sycl_nnz_rows, sycl_gb_per_s, sycl_labels))
            sycl_nnz_rows_sorted, sycl_gb_per_s_sorted, sycl_labels_sorted = zip(*sycl_sorted)
            ax.plot(sycl_nnz_rows_sorted, sycl_gb_per_s_sorted, '-s', label='SYCL', markerfacecolor='white')
            
            # Add annotations for SYCL points
            for x, y, label in zip(sycl_nnz_rows_sorted, sycl_gb_per_s_sorted, sycl_labels_sorted):
                ax.annotate(label, (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        
        # Customize plot
        ax.set_xlabel('nnz_row')
        ax.set_ylabel('GB/s')
        ax.set_title(f'SPMM Performance Comparison (n={n})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set x-axis to log scale if there's a wide range
        if mkl_nnz_rows or sycl_nnz_rows:
            all_nnz = (mkl_nnz_rows or []) + (sycl_nnz_rows or [])
            if max(all_nnz) / min(all_nnz) > 10:
                ax.set_xscale('log')
        
        plt.tight_layout()
        
        # Save plot
        output_filename = f'spmm_performance_n{n}.png'
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {output_filename}")
        
        plt.show()

def main():
    """Main function to parse data and create plots."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    mkl_file = os.path.join(script_dir, 'spmm_experiments_mkl.out')
    sycl_file = os.path.join(script_dir, 'spmm_experiments_sycl.out')
    
    # Check if files exist
    if not os.path.exists(mkl_file):
        print(f"Error: MKL results file not found: {mkl_file}")
        return
    
    if not os.path.exists(sycl_file):
        print(f"Error: SYCL results file not found: {sycl_file}")
        return
    
    # Parse results
    print("Parsing MKL results...")
    mkl_results = parse_mkl_results(mkl_file)
    print(f"Found {len(mkl_results)} MKL results")
    
    print("Parsing SYCL results...")
    sycl_results = parse_sycl_results(sycl_file)
    print(f"Found {len(sycl_results)} SYCL results")
    
    # Create plots
    print("Creating plots...")
    create_plots(mkl_results, sycl_results)
    
    print("Done!")

if __name__ == "__main__":
    main()