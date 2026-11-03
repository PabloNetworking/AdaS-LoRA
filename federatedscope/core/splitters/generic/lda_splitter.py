import numpy as np
import os
from federatedscope.core.splitters import BaseSplitter
from federatedscope.core.splitters.utils import \
    dirichlet_distribution_noniid_slice


class LDASplitter(BaseSplitter):
    """
    This splitter split dataset with LDA.

    Args:
        client_num: the dataset will be split into ``client_num`` pieces
        alpha (float): Partition hyperparameter in LDA, smaller alpha \
            generates more extreme heterogeneous scenario see \
            ``np.random.dirichlet``
    """
    def __init__(self, client_num, alpha=0.1, save_plot=False, exp_name="experiment", plot_dir="exp/plots"):
        self.alpha = alpha
        self.save_plot = save_plot
        self.exp_name = exp_name
        self.plot_dir = plot_dir
        super(LDASplitter, self).__init__(client_num)

    def __call__(self, dataset, prior=None, **kwargs):
        from torch.utils.data import Dataset, Subset
        tmp_dataset = [ds for ds in dataset]
        if isinstance(tmp_dataset[0], tuple):
            label = np.array([y for x, y in tmp_dataset])
        elif isinstance(tmp_dataset[0], dict):
            if 'categories' in tmp_dataset[0]:
                label = np.array([x['categories'] for x in tmp_dataset])
            elif 'label' in tmp_dataset[0]:
                label = np.array([x['label'] for x in tmp_dataset])
            # GLUE datasets use 'labels' (a scalar tensor). int() yields the
            # class id (classification) or a coarse score-bin (STS-B regression)
            # for the Dirichlet label-skew split.
            elif 'labels' in tmp_dataset[0]:
                label = np.array([int(x['labels']) for x in tmp_dataset])
            else:
                raise ValueError('Cannot find label or categories in dataset')
        else:
            raise TypeError(f'Unsupported data formats {type(tmp_dataset[0])}')
        idx_slice = dirichlet_distribution_noniid_slice(label,
                                                        self.client_num,
                                                        self.alpha,
                                                        prior=prior)
        if isinstance(dataset, Dataset):
            data_list = [Subset(dataset, idxs) for idxs in idx_slice]
        else:
            data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice]
        
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
                splitter_name="lda",
                exp_name=self.exp_name
            )
            
            # Create summary plot
            summary_path = visualizer.create_summary_plot(
                dataset=dataset,
                indices_list=idx_slice,
                splitter_name="lda",
                exp_name=self.exp_name
            )
            
            print(f"LDA Splitter visualization saved:")
            print(f"  - Partition plot: {plot_path}")
            print(f"  - Summary plot: {summary_path}")
            
        except ImportError as e:
            print(f"Warning: Could not generate visualization plots: {e}")
        except Exception as e:
            print(f"Warning: Error generating visualization plots: {e}")
