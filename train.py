import datetime
import sys
import math
import time
from os import path as osp
from typing import Dict, Any, Tuple, List, Optional

# import torch.cuda
import torch

from datasets import build_dataloader, build_dataset
from datasets.data_sampler import EnlargedSampler

from models import build_model
from utils import (AvgTimer, MessageLogger, get_env_info, get_root_logger,
                   init_tb_logger)
from utils.options import dict2str, parse_options

import wandb

torch.autograd.set_detect_anomaly(True)

# wandb.init(mode="disabled") 

class Trainer:
    """Main trainer class that encapsulates the training pipeline."""
    
    def __init__(self, root_path: str):
        """Initialize the trainer with the project root path.
        
        Args:
            root_path: Root path of the project
        """
        # Parse options, set distributed setting, set random seed
        self.opt = parse_options(root_path, is_train=True)
        self.opt['root_path'] = root_path
        
        # Initialize loggers
        self._setup_loggers()
        
        # Create dataloaders
        self._setup_dataloaders()
        
        # Create model
        self.model = build_model(self.opt)
        
        # Create message logger (formatted outputs)
        self.msg_logger = MessageLogger(self.opt, self.model.curr_iter, self.tb_logger)
        
        # Initialize timers
        self.data_timer = AvgTimer()
        self.iter_timer = AvgTimer()
        
        # Training state
        self.start_time = None
    
    def _setup_loggers(self) -> None:
        """Set up all loggers (file, tensorboard, wandb)."""
        # WARNING: should not use get_root_logger in the above codes
        # Otherwise the logger will not be properly initialized
        log_file = osp.join(self.opt['path']['log'], f"train_{self.opt['name']}.log")
        self.logger = get_root_logger(log_file=log_file)
        self.logger.info(get_env_info())
        self.logger.info(dict2str(self.opt))
        
        # Initialize tensorboard logger
        self.tb_logger = init_tb_logger(self.opt['path']['experiments_root'])
        
        # Initialize wandb logger
        wandb.init(
            project="Shape-Correspondence",
            name=self.opt['name'],
            config=self.opt,
        )
    
    def _setup_dataloaders(self) -> None:
        """Set up training and validation dataloaders."""
        result = self._create_train_val_dataloader()
        self.train_loader, self.train_sampler, self.val_loader, self.total_epochs, total_iters = result
        self.opt['train']['total_iter'] = total_iters
    
    def _create_train_val_dataloader(self) -> Tuple:
        """Create training and validation dataloaders.
        
        Returns:
            Tuple containing train_loader, train_sampler, val_loader, total_epochs, total_iters
        """
        train_set, val_set = None, None
        dataset_enlarge_ratio = 1
        
        # Create train and val datasets
        for dataset_name, dataset_opt in self.opt['datasets'].items():
            if isinstance(dataset_opt, int):  # batch_size, num_worker
                continue
            if dataset_name.startswith('train'):
                dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
                if train_set is None:
                    train_set = build_dataset(dataset_opt)
                else:
                    train_set += build_dataset(dataset_opt)
            elif dataset_name.startswith('val') or dataset_name.startswith('test'):
                if val_set is None:
                    val_set = build_dataset(dataset_opt)
                else:
                    val_set += build_dataset(dataset_opt)

        # Create train and val dataloaders
        train_sampler = EnlargedSampler(
            train_set, 
            self.opt['world_size'], 
            self.opt['rank'], 
            dataset_enlarge_ratio
        )
        
        train_loader = build_dataloader(
            train_set,
            self.opt['datasets'],
            'train',
            num_gpu=self.opt['num_gpu'],
            dist=self.opt['dist'],
            sampler=train_sampler,
            seed=self.opt['manual_seed']
        )
        
        batch_size = self.opt['datasets']['batch_size']
        num_iter_per_epoch = math.ceil(len(train_set) * dataset_enlarge_ratio / batch_size)
        total_epochs = int(self.opt['train']['total_epochs'])
        total_iters = total_epochs * num_iter_per_epoch
        
        self.logger.info('Training statistics:'
                    f'\n\tNumber of train images: {len(train_set)}'
                    f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                    f'\n\tBatch size: {batch_size}'
                    f'\n\tWorld size (gpu number): {self.opt["world_size"]}'
                    f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                    f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        val_loader = build_dataloader(
            val_set, 
            self.opt['datasets'], 
            'val', 
            num_gpu=self.opt['num_gpu'], 
            dist=self.opt['dist'], 
            sampler=None,
            seed=self.opt['manual_seed']
        )
        
        self.logger.info('Validation statistics:'
                    f'\n\tNumber of val images: {len(val_set)}')

        return train_loader, train_sampler, val_loader, total_epochs, total_iters
    
    def train(self) -> None:
        """Execute the training pipeline."""
        self.logger.info(
            f'Start training from epoch: {self.model.curr_epoch}, iter: {self.model.curr_iter}'
        )
        self.start_time = time.time()
        
        try:
            self._train_loop()
            self._finalize_training()
        except KeyboardInterrupt:
            self._handle_interrupt()
    
    def _train_loop(self) -> None:
        """Main training loop."""
        while self.model.curr_epoch < self.total_epochs:
            self.model.curr_epoch += 1
            self.train_sampler.set_epoch(self.model.curr_epoch)
            
            for train_data in self.train_loader:
                self.data_timer.record()
                self.model.curr_iter += 1
                
                # Process data and forward pass
                self.model.feed_data(train_data)
                # Backward pass
                self.model.optimize_parameters()
                # Update model per iteration
                self.model.update_model_per_iteration()
                
                self.iter_timer.record()
                if self.model.curr_iter == 1:
                    # Reset start time in msg_logger for more accurate eta_time
                    # Not work in resume mode
                    self.msg_logger.reset_start_time()
                
                # Log progress
                self._log_progress()
                
                # Save checkpoint
                self._save_checkpoint()
                
                # Run validation
                self._run_validation()
                
                self.data_timer.start()
                self.iter_timer.start()
                # End of iter
            
            # Update model per epoch
            self.model.update_model_per_epoch()
            # End of epoch
    
    def _log_progress(self) -> None:
        """Log training progress."""
        if self.model.curr_iter % self.opt['logger']['print_freq'] == 0:
            log_vars = {'epoch': self.model.curr_epoch, 'iter': self.model.curr_iter}
            log_vars.update({'lrs': self.model.get_current_learning_rate()})
            log_vars.update({
                'time': self.iter_timer.get_avg_time(), 
                'data_time': self.data_timer.get_avg_time()
            })
            log_vars.update(self.model.get_loss_metrics())
            wandb.log(log_vars)  # Log to wandb
            self.msg_logger(log_vars)
    
    def _save_checkpoint(self) -> None:
        """Save model checkpoint."""
        if self.model.curr_iter % self.opt['logger']['save_checkpoint_freq'] == 0:
            self.logger.info('Saving models and training states.')
            self.model.save_model(net_only=False, best=False)
    
    def _run_validation(self) -> None:
        """Run validation if needed."""
        if (self.opt.get('val') is not None and 
            (self.model.curr_iter % self.opt['val']['val_freq'] == 0)):
            self.logger.info('Start validation.')
            torch.cuda.empty_cache()
            self.model.validation(self.val_loader, self.tb_logger, wandb=wandb, update=True)
    
    def _finalize_training(self) -> None:
        """Finalize training, run final validation and save best model."""
        consumed_time = str(datetime.timedelta(seconds=int(time.time() - self.start_time)))
        self.logger.info(f'End of training. Time consumed: {consumed_time}')
        self.logger.info(f'Last Validation.')
        
        if self.opt.get('val') is not None:
            self.model.validation(self.val_loader, self.tb_logger, wandb=wandb, update=False)
        
        self.logger.info('Save the best model.')
        self.model.save_model(net_only=True, best=True)  # Save the best model
        
        # Close loggers
        if self.tb_logger:
            self.tb_logger.close()
        wandb.finish()
    
    def _handle_interrupt(self) -> None:
        """Handle keyboard interrupt by saving the current model."""
        self.logger.info('Keyboard interrupt. Save model and exit...')
        self.model.save_model(net_only=False, best=False)
        self.model.save_model(net_only=True, best=True)
        
        # Close loggers
        if self.tb_logger:
            self.tb_logger.close()
        wandb.finish()
        
        sys.exit(0)


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    trainer = Trainer(root_path)
    trainer.train()
