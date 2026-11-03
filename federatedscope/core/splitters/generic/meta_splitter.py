import random
import numpy as np
import os

from federatedscope.core.splitters import BaseSplitter


class MetaSplitter(BaseSplitter):
    """
    This splitter split dataset with meta information with LLM dataset.

    Args:
        client_num: the dataset will be split into ``client_num`` pieces
    """
    def __init__(self, client_num, save_plot=False, exp_name="experiment", plot_dir="exp/plots", **kwargs):
        self.save_plot = save_plot
        self.exp_name = exp_name
        self.plot_dir = plot_dir
        super(MetaSplitter, self).__init__(client_num)

    def __call__(self, dataset, prior=None, **kwargs):
        from torch.utils.data import Dataset, Subset

        tmp_dataset = [ds for ds in dataset]
        if isinstance(tmp_dataset[0], tuple):
            label = np.array([y for x, y in tmp_dataset])
        elif isinstance(tmp_dataset[0], dict):
            label = np.array([x['categories'] for x in tmp_dataset])
        else:
            raise TypeError(f'Unsupported data formats {type(tmp_dataset[0])}')

        # Split by categories
        categories = set(label)
        idx_slice = []
        for cat in categories:
            idx_slice.append(np.where(np.array(label) == cat)[0].tolist())
        random.shuffle(idx_slice)

        # Merge / pad to exactly client_num pieces by assigning category
        # slices round-robin. A split with fewer categories than client_num
        # still yields client_num pieces (some of them empty).
        new_idx_slice = [[] for _ in range(self.client_num)]
        for i in range(len(idx_slice)):
            new_idx_slice[i % self.client_num].extend(idx_slice[i])

        if isinstance(dataset, Dataset):
            data_list = [Subset(dataset, idxs) for idxs in new_idx_slice]
        else:
            data_list = [[dataset[idx] for idx in idxs] for idxs in new_idx_slice]
        
        # Generate visualization if requested and this is the training data (not test/val)
        # Plot only large splits (more than 1000 samples), i.e. the training data
        if self.save_plot and len(dataset) > 1000:  # Only visualize if dataset is large enough (training data)
            self._generate_visualization(dataset, idx_slice)
        
        return data_list
    
    def _generate_visualization(self, dataset, idx_slice):
        """Generate visualization plots for the data partition."""
        try:
            from federatedscope.core.splitters.visualization import DataPartitionVisualizer
            
            visualizer = DataPartitionVisualizer(save_dir=self.plot_dir)
            
            # Create main partition plot
            plot_path = visualizer.plot_data_partition(
                dataset=dataset,
                indices_list=idx_slice,
                splitter_name="meta",
                exp_name=self.exp_name
            )
            
            # Create summary plot
            summary_path = visualizer.create_summary_plot(
                dataset=dataset,
                indices_list=idx_slice,
                splitter_name="meta",
                exp_name=self.exp_name
            )
            
            print(f"Meta Splitter visualization saved:")
            print(f"  - Partition plot: {plot_path}")
            print(f"  - Summary plot: {summary_path}")
            
        except ImportError as e:
            print(f"Warning: Could not generate visualization plots: {e}")
        except Exception as e:
            print(f"Warning: Error generating visualization plots: {e}")
