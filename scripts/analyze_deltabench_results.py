#!/usr/bin/env python3
"""
Analyze DeltaBench results for consistency and performance metrics.
Creates visualization showing how performance varies with problem position.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def load_and_validate_data(file_path):
    """Load and validate the JSONL file."""
    print(f"Loading data from {file_path}...")
    
    data = []
    required_fields = ['predicted_sections', 'true_sections', 'precision', 'recall', 'f1_score', 'judge', 'parsing_success']
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                data.append(entry)
                
                # Validate required fields
                missing_fields = [field for field in required_fields if field not in entry]
                if missing_fields:
                    print(f"Warning: Line {line_num} missing fields: {missing_fields}")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    print(f"Successfully loaded {len(data)} entries")
    return data

def calculate_metrics(data):
    """Calculate micro and macro averaged metrics."""
    print("\nCalculating metrics...")
    
    # Individual metrics for macro averaging
    precisions = []
    recalls = []
    f1_scores = []
    
    # Aggregated counts for micro averaging
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    parsing_success_count = 0
    judge_success_count = 0
    
    for entry in data:
        if 'precision' in entry and entry['precision'] is not None:
            precisions.append(entry['precision'])
        if 'recall' in entry and entry['recall'] is not None:
            recalls.append(entry['recall'])
        if 'f1_score' in entry and entry['f1_score'] is not None:
            f1_scores.append(entry['f1_score'])
            
        # Aggregate TP, FP, FN for micro averaging
        if 'tp_step' in entry and entry['tp_step'] is not None:
            total_tp += entry['tp_step']
        if 'fp_step' in entry and entry['fp_step'] is not None:
            total_fp += entry['fp_step']
        if 'fn_step' in entry and entry['fn_step'] is not None:
            total_fn += entry['fn_step']
            
        if 'parsing_success' in entry and entry['parsing_success'] == 1:
            parsing_success_count += 1
        if 'judge' in entry and entry['judge'] == 1:
            judge_success_count += 1
    
    # Macro averages
    macro_precision = np.mean(precisions) if precisions else 0
    macro_recall = np.mean(recalls) if recalls else 0
    macro_f1 = np.mean(f1_scores) if f1_scores else 0
    
    # Micro averages
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    return {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'parsing_success_rate': parsing_success_count / len(data),
        'judge_success_rate': judge_success_count / len(data),
        'total_entries': len(data)
    }

def create_position_bins(data, min_bin_size=50):
    """Create bins based on the first error position with greedy approach for consistent positions."""
    print(f"\nCreating position bins with minimum size {min_bin_size}...")
    
    # Extract first error position for each entry
    position_data = []
    for entry in data:
        if 'true_sections' in entry and entry['true_sections']:
            # Get the first (smallest) error position
            first_position = min(entry['true_sections'])
            position_data.append({
                'position': first_position,
                'precision': entry.get('precision', 0),
                'recall': entry.get('recall', 0),
                'f1_score': entry.get('f1_score', 0),
                'predicted_sections': entry.get('predicted_sections', []),
                'true_sections': entry.get('true_sections', [])
            })
    
    # Group by position
    position_groups = defaultdict(list)
    for item in position_data:
        position_groups[item['position']].append(item)
    
    # Sort positions
    sorted_positions = sorted(position_groups.keys())
    
    # Create bins greedily - keep same positions together
    bins = []
    bin_labels = []
    current_bin = []
    current_positions = []
    
    for pos in sorted_positions:
        group = position_groups[pos]
        
        # If adding this group would exceed reasonable size and we have enough items
        if len(current_bin) >= min_bin_size and len(current_bin) + len(group) > min_bin_size * 2:
            # Finalize current bin
            bins.append(current_bin)
            if len(current_positions) == 1:
                bin_labels.append(f"{current_positions[0]}")
            else:
                bin_labels.append(f"{min(current_positions)}-{max(current_positions)}")
            
            current_bin = []
            current_positions = []
        
        # Add current group to bin
        current_bin.extend(group)
        current_positions.append(pos)
    
    # Add final bin if not empty
    if current_bin:
        bins.append(current_bin)
        if len(current_positions) == 1:
            bin_labels.append(f"{current_positions[0]}")
        else:
            bin_labels.append(f"{min(current_positions)}-{max(current_positions)}")
    
    # If we have bins that are too small, merge them with adjacent bins
    merged_bins = []
    merged_labels = []
    
    i = 0
    while i < len(bins):
        current_bin = bins[i]
        current_label = bin_labels[i]
        
        # If current bin is too small and we have more bins, merge with next
        while len(current_bin) < min_bin_size and i + 1 < len(bins):
            next_bin = bins[i + 1]
            next_label = bin_labels[i + 1]
            
            current_bin.extend(next_bin)
            
            # Update label for merged bin
            current_positions = [item['position'] for item in current_bin]
            min_pos = min(current_positions)
            max_pos = max(current_positions)
            
            if min_pos == max_pos:
                current_label = f"{min_pos}"
            else:
                current_label = f"{min_pos}-{max_pos}"
            
            i += 1
        
        merged_bins.append(current_bin)
        merged_labels.append(current_label)
        i += 1
    
    return merged_bins, merged_labels

def calculate_bin_metrics(bins, approach='macro'):
    """Calculate metrics for each bin using either macro or micro approach."""
    bin_metrics = []
    bin_sizes = []
    
    for bin_data in bins:
        if not bin_data:
            bin_metrics.append({'precision': 0, 'recall': 0, 'f1_score': 0})
            bin_sizes.append(0)
            continue
        
        bin_sizes.append(len(bin_data))
        
        if approach == 'macro':
            # Macro: average individual metrics
            precisions = [item['precision'] for item in bin_data if item['precision'] is not None]
            recalls = [item['recall'] for item in bin_data if item['recall'] is not None]
            f1_scores = [item['f1_score'] for item in bin_data if item['f1_score'] is not None]
            
            bin_metrics.append({
                'precision': np.mean(precisions) if precisions else 0,
                'recall': np.mean(recalls) if recalls else 0,
                'f1_score': np.mean(f1_scores) if f1_scores else 0
            })
        else:  # micro
            # Micro: aggregate TP, FP, FN then calculate metrics
            # Extract TP, FP, FN from the loaded data entries
            total_tp = 0
            total_fp = 0
            total_fn = 0
            
            for item in bin_data:
                # Find the original entry in the data to get TP/FP/FN
                # Since we stored the metrics directly, we need to calculate from predicted/true sections
                predicted = set(item.get('predicted_sections', []))
                true = set(item.get('true_sections', []))
                
                tp = len(predicted & true)
                fp = len(predicted - true)
                fn = len(true - predicted)
                
                total_tp += tp
                total_fp += fp
                total_fn += fn
            
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            bin_metrics.append({
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            })
    
    return bin_metrics, bin_sizes

def create_visualization(bins, bin_labels, overall_metrics):
    """Create visualizations with both macro and micro approaches."""
    print("\nCreating visualizations...")
    
    # Calculate metrics for both approaches
    macro_metrics, bin_sizes = calculate_bin_metrics(bins, 'macro')
    micro_metrics, _ = calculate_bin_metrics(bins, 'micro')
    
    # Create both plots
    for approach, metrics in [('Macro', macro_metrics), ('Micro', micro_metrics)]:
        fig = plt.figure(figsize=(16, 10))
        
        # Create a custom layout with shared x-axis
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        
        # Top plot: Metrics by position (line plots)
        x = np.arange(len(bin_labels))
        
        precisions = [m['precision'] for m in metrics]
        recalls = [m['recall'] for m in metrics]
        f1_scores = [m['f1_score'] for m in metrics]
        
        ax1.plot(x, precisions, marker='o', linewidth=3, markersize=8, label='Precision', color='#1f77b4')
        ax1.plot(x, recalls, marker='s', linewidth=3, markersize=8, label='Recall', color='#ff7f0e')
        ax1.plot(x, f1_scores, marker='^', linewidth=3, markersize=8, label='F1-Score', color='#2ca02c')
        
        # Add value labels on points (only for key points to avoid clutter)
        for i, (p, r, f) in enumerate(zip(precisions, recalls, f1_scores)):
            if i % 2 == 0 or i == len(precisions) - 1:  # Show every other point and last point
                ax1.text(i, p + 0.03, f'{p:.3f}', ha='center', va='bottom', fontsize=9, color='#1f77b4', weight='bold')
                ax1.text(i, r + 0.03, f'{r:.3f}', ha='center', va='bottom', fontsize=9, color='#ff7f0e', weight='bold')
                ax1.text(i, f + 0.03, f'{f:.3f}', ha='center', va='bottom', fontsize=9, color='#2ca02c', weight='bold')
        
        ax1.set_ylabel('Score', fontsize=12, weight='bold')
        ax1.set_title(f'{approach} Performance Metrics by First Error Position', fontsize=14, weight='bold', pad=20)
        ax1.legend(fontsize=11, loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='both', labelsize=10)
        
        # Remove x-axis labels from top plot
        ax1.set_xticklabels([])
        
        # Bottom plot: Bin sizes (histogram)
        bars = ax2.bar(x, bin_sizes, alpha=0.8, color='#d62728', width=0.7, edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Error Position Bins', fontsize=12, weight='bold')
        ax2.set_ylabel('# Problems', fontsize=12, weight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=10)
        ax2.grid(axis='y', alpha=0.3)
        ax2.tick_params(axis='both', labelsize=10)
        
        # Add value labels on bars
        for i, (bar, size) in enumerate(zip(bars, bin_sizes)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(bin_sizes) * 0.01, 
                    str(size), ha='center', va='bottom', fontsize=9, weight='bold')
        
        # Add overall statistics as text box
        overall_text = f"Overall {approach} Metrics:\n"
        overall_text += f"Precision: {overall_metrics[f'{approach.lower()}_precision']:.3f}\n"
        overall_text += f"Recall: {overall_metrics[f'{approach.lower()}_recall']:.3f}\n"
        overall_text += f"F1-Score: {overall_metrics[f'{approach.lower()}_f1']:.3f}"
        
        ax1.text(0.02, 0.98, overall_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.subplots_adjust(hspace=0.1)
        
        # Save with approach-specific filename
        filename = f'/Users/nikitakaragodin/google/visualizations/deltabench_performance_analysis_{approach.lower()}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        print(f"Saved {approach} visualization to: {filename}")
    
    return macro_metrics, micro_metrics, bin_sizes

def print_summary_table(overall_metrics, macro_metrics, micro_metrics, bin_labels, bin_sizes):
    """Print summary tables."""
    print("\n" + "="*80)
    print("OVERALL PERFORMANCE METRICS")
    print("="*80)
    
    print(f"{'Metric':<20} {'Macro':<12} {'Micro':<12}")
    print("-" * 44)
    print(f"{'Precision':<20} {overall_metrics['macro_precision']:<12.4f} {overall_metrics['micro_precision']:<12.4f}")
    print(f"{'Recall':<20} {overall_metrics['macro_recall']:<12.4f} {overall_metrics['micro_recall']:<12.4f}")
    print(f"{'F1-Score':<20} {overall_metrics['macro_f1']:<12.4f} {overall_metrics['micro_f1']:<12.4f}")
    print()
    print(f"{'Parsing Success Rate':<20} {overall_metrics['parsing_success_rate']:<12.4f}")
    print(f"{'Judge Success Rate':<20} {overall_metrics['judge_success_rate']:<12.4f}")
    print(f"{'Total Entries':<20} {overall_metrics['total_entries']:<12}")
    
    print("\n" + "="*80)
    print("PERFORMANCE BY ERROR POSITION BINS - MACRO APPROACH")
    print("="*80)
    
    print(f"{'Position':<12} {'Size':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 68)
    
    for i, (label, size, metrics) in enumerate(zip(bin_labels, bin_sizes, macro_metrics)):
        print(f"{label:<12} {size:<8} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f}")
    
    print("\n" + "="*80)
    print("PERFORMANCE BY ERROR POSITION BINS - MICRO APPROACH")
    print("="*80)
    
    print(f"{'Position':<12} {'Size':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 68)
    
    for i, (label, size, metrics) in enumerate(zip(bin_labels, bin_sizes, micro_metrics)):
        print(f"{label:<12} {size:<8} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f}")

def main():
    """Main analysis function."""
    file_path = '/Users/nikitakaragodin/google/results/Deltabench_v1_gpt-4o-mini_deltabench.jsonl'
    
    # Load and validate data
    data = load_and_validate_data(file_path)
    
    # Calculate overall metrics
    overall_metrics = calculate_metrics(data)
    
    # Create position bins
    bins, bin_labels = create_position_bins(data, min_bin_size=50)
    
    # Create visualizations
    macro_metrics, micro_metrics, bin_sizes = create_visualization(bins, bin_labels, overall_metrics)
    
    # Print summary
    print_summary_table(overall_metrics, macro_metrics, micro_metrics, bin_labels, bin_sizes)

if __name__ == "__main__":
    main()