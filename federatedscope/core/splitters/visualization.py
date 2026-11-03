"""
Visualization utilities for data partition analysis in federated learning.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple


class DataPartitionVisualizer:
    """
    Visualizes data partitions for federated learning splitters.
    """
    
    def __init__(self, save_dir: str = "exp/plots", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            save_dir: Directory to save plots
            figsize: Figure size for plots
        """
        self.save_dir = save_dir
        self.figsize = figsize
        os.makedirs(save_dir, exist_ok=True)
        
        # Set up color palette
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
        self.color_map = {}
    
    def _get_label_counts(self, dataset, indices_list: List[List[int]]) -> Tuple[List[Counter], List[int], Dict]:
        """
        Get label counts for each client's data partition.
        
        Args:
            dataset: The dataset object
            indices_list: List of indices for each client
            
        Returns:
            Tuple of (label_counts_per_client, total_samples_per_client, label_to_name_mapping)
        """
        # Convert dataset to list format
        tmp_dataset = [ds for ds in dataset]
        
        if isinstance(tmp_dataset[0], tuple):
            labels = [y for x, y in tmp_dataset]
        elif isinstance(tmp_dataset[0], dict):
            if 'categories' in tmp_dataset[0]:
                labels = [x['categories'] for x in tmp_dataset]
            elif 'label' in tmp_dataset[0]:
                labels = [x['label'] for x in tmp_dataset]
            else:
                raise ValueError('Cannot find label or categories in dataset')
        else:
            raise TypeError(f'Unsupported data formats {type(tmp_dataset[0])}')
        
        # Create mapping from label values to display names
        unique_labels = sorted(list(set(labels)))
        label_to_name_mapping = {}
        for i, label in enumerate(unique_labels):
            if isinstance(label, str):
                # Use the actual category name
                label_to_name_mapping[label] = label
            else:
                # For numeric labels, try to get meaningful names
                label_to_name_mapping[label] = f'Category {label}'
        
        # Count labels for each client
        label_counts_per_client = []
        total_samples_per_client = []
        
        for client_indices in indices_list:
            client_labels = [labels[idx] for idx in client_indices]
            label_counts = Counter(client_labels)
            label_counts_per_client.append(label_counts)
            total_samples_per_client.append(len(client_indices))
        
        return label_counts_per_client, total_samples_per_client, label_to_name_mapping
    
    def _get_color_for_label(self, label: Any) -> str:
        """
        Get color for a specific label.
        
        Args:
            label: The label value
            
        Returns:
            Color string
        """
        if label not in self.color_map:
            # Use modulo to cycle through colors
            color_idx = len(self.color_map) % len(self.colors)
            self.color_map[label] = self.colors[color_idx]
        return self.color_map[label]
    
    def plot_data_partition(self, 
                          dataset, 
                          indices_list: List[List[int]], 
                          splitter_name: str = "splitter",
                          exp_name: str = "experiment",
                          special_clients: Optional[List[int]] = None) -> str:
        """
        Create a horizontal stacked bar chart showing data partition.
        
        Args:
            dataset: The dataset object
            indices_list: List of indices for each client
            splitter_name: Name of the splitter used
            exp_name: Name of the experiment
            special_clients: Optional client indices to highlight in the plot
            
        Returns:
            Path to saved plot
        """
        # Get label counts
        label_counts_per_client, total_samples_per_client, label_to_name_mapping = self._get_label_counts(dataset, indices_list)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Get all unique labels
        all_labels = set()
        for label_counts in label_counts_per_client:
            all_labels.update(label_counts.keys())
        all_labels = sorted(list(all_labels))
        
        # Create horizontal stacked bars
        y_pos = np.arange(len(indices_list))
        bottom = np.zeros(len(indices_list))
        
        # Plot bars for each label
        for label in all_labels:
            counts = [label_counts.get(label, 0) for label_counts in label_counts_per_client]
            color = self._get_color_for_label(label)
            
            # Outline highlighted clients
            if special_clients is not None:
                # Create a pattern for special clients
                bars = ax.barh(y_pos, counts, left=bottom, 
                             color=color, alpha=0.8, 
                             edgecolor='black' if any(i in special_clients for i in range(len(counts))) else 'none',
                             linewidth=1.5 if any(i in special_clients for i in range(len(counts))) else 0.5)
            else:
                ax.barh(y_pos, counts, left=bottom, color=color, alpha=0.8)
            
            bottom += counts
        
        # Customize plot
        ax.set_xlabel('Number of Samples', fontsize=12)
        ax.set_ylabel('Client ID', fontsize=12)
        ax.set_title(f'Data Partition - {splitter_name.upper()} Splitter\n{exp_name}', fontsize=14, fontweight='bold')
        
        # Set y-axis labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'Client {i}' for i in range(len(indices_list))])
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='x')
        
        # Create legend
        legend_elements = []
        for label in all_labels:
            color = self._get_color_for_label(label)
            display_name = label_to_name_mapping.get(label, f'Class {label}')
            legend_elements.append(mpatches.Patch(color=color, label=display_name))
        
        # Add special client legend if applicable
        if special_clients is not None:
            legend_elements.append(mpatches.Patch(color='black', label='Highlighted Clients', alpha=0.3))
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
        
        # Add statistics text
        stats_text = f'Total Clients: {len(indices_list)}\n'
        stats_text += f'Total Samples: {sum(total_samples_per_client)}\n'
        stats_text += f'Avg Samples/Client: {np.mean(total_samples_per_client):.1f}\n'
        stats_text += f'Std Samples/Client: {np.std(total_samples_per_client):.1f}'
        
        if special_clients is not None:
            stats_text += f'\nSpecial Clients: {len(special_clients)}'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        filename = f"{exp_name}_{splitter_name}_partition.png"
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _plot_partition_with_special_clients(self, ax, label_counts_per_client, total_samples_per_client, special_clients, label_to_name_mapping):
        """Plot data partition with special client highlighting."""
        all_labels = set()
        for label_counts in label_counts_per_client:
            all_labels.update(label_counts.keys())
        all_labels = sorted(list(all_labels))
        
        y_pos = np.arange(len(label_counts_per_client))
        bottom = np.zeros(len(label_counts_per_client))
        
        for label in all_labels:
            counts = [label_counts.get(label, 0) for label_counts in label_counts_per_client]
            color = self._get_color_for_label(label)
            
            # Create bars for this label with different styling for special clients
            for i, count in enumerate(counts):
                if count > 0:  # Only create bar if there's data
                    if i in special_clients:
                        # Special clients get red border
                        ax.barh(i, count, left=bottom[i], color=color, alpha=0.9, 
                               edgecolor='red', linewidth=2)
                    else:
                        # Regular clients get no border
                        ax.barh(i, count, left=bottom[i], color=color, alpha=0.8, 
                               edgecolor='none', linewidth=0.5)
            
            bottom += counts
        
        ax.set_xlabel('Number of Samples')
        ax.set_ylabel('Client ID')
        ax.set_title('Data Partition (Red border = Special clients)')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'Client {i}' for i in range(len(label_counts_per_client))])
        ax.grid(True, alpha=0.3, axis='x')
    
    def create_summary_plot(self, 
                          dataset, 
                          indices_list: List[List[int]], 
                          splitter_name: str = "splitter",
                          exp_name: str = "experiment",
                          special_clients: Optional[List[int]] = None) -> str:
        """
        Create a comprehensive summary plot.
        
        Args:
            dataset: The dataset object
            indices_list: List of indices for each client
            splitter_name: Name of the splitter used
            exp_name: Name of the experiment
            special_clients: Optional client indices to highlight in the plot
            
        Returns:
            Path to saved plot
        """
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))
        
        # Main partition plot (takes up most space)
        ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        label_counts_per_client, total_samples_per_client, label_to_name_mapping = self._get_label_counts(dataset, indices_list)
        self._plot_partition_with_special_clients(ax_main, 
                                                label_counts_per_client, 
                                                total_samples_per_client, 
                                                special_clients or [], 
                                                label_to_name_mapping)
        
        # Statistics subplot
        ax_stats = plt.subplot2grid((3, 3), (0, 2))
        self._plot_statistics(ax_stats, indices_list, special_clients)
        
        # Class distribution subplot
        ax_dist = plt.subplot2grid((3, 3), (1, 2))
        self._plot_class_distribution(ax_dist, dataset, indices_list, label_to_name_mapping)
        
        # Client size distribution
        ax_size = plt.subplot2grid((3, 3), (2, 0), colspan=3)
        self._plot_client_size_distribution(ax_size, indices_list, special_clients)
        
        # Add overall title
        fig.suptitle(f'Comprehensive Data Partition Analysis - {splitter_name.upper()}\n{exp_name}', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        filename = f"{exp_name}_{splitter_name}_summary.png"
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _plot_statistics(self, ax, indices_list, special_clients):
        """Plot basic statistics."""
        total_samples = [len(indices) for indices in indices_list]
        
        stats = {
            'Total Clients': len(indices_list),
            'Total Samples': sum(total_samples),
            'Avg Samples/Client': np.mean(total_samples),
            'Std Samples/Client': np.std(total_samples),
            'Min Samples': np.min(total_samples),
            'Max Samples': np.max(total_samples)
        }
        
        if special_clients:
            stats['Special Clients'] = len(special_clients)
            stats['Regular Clients'] = len(indices_list) - len(special_clients)
        
        # Create text representation
        stats_text = '\n'.join([f'{k}: {v:.1f}' if isinstance(v, float) else f'{k}: {v}' 
                              for k, v in stats.items()])
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax.set_title('Statistics')
        ax.axis('off')
    
    def _plot_class_distribution(self, ax, dataset, indices_list, label_to_name_mapping):
        """Plot class distribution across all clients."""
        label_counts_per_client, _, _ = self._get_label_counts(dataset, indices_list)
        
        # Get all unique labels
        all_labels = set()
        for label_counts in label_counts_per_client:
            all_labels.update(label_counts.keys())
        all_labels = sorted(list(all_labels))
        
        # Calculate total counts per class
        class_totals = {}
        for label in all_labels:
            class_totals[label] = sum(label_counts.get(label, 0) for label_counts in label_counts_per_client)
        
        # Create pie chart with actual category names
        labels = [label_to_name_mapping.get(label, f'Class {label}') for label in all_labels]
        sizes = list(class_totals.values())
        colors = [self._get_color_for_label(label) for label in all_labels]
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Class Distribution')
    
    def _plot_client_size_distribution(self, ax, indices_list, special_clients):
        """Plot client size distribution."""
        client_sizes = [len(indices) for indices in indices_list]
        client_ids = list(range(len(indices_list)))
        
        colors = ['lightcoral' if i in (special_clients or []) else 'lightblue' 
                 for i in client_ids]
        
        bars = ax.bar(client_ids, client_sizes, color=colors, alpha=0.8)
        
        # Add value labels on bars
        for bar, size in zip(bars, client_sizes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                   str(size), ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Client ID')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Client Size Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add legend if special clients exist
        if special_clients:
            regular_patch = mpatches.Patch(color='lightblue', label='Regular Clients')
            special_patch = mpatches.Patch(color='lightcoral', label='Highlighted Clients')
            ax.legend(handles=[regular_patch, special_patch])
