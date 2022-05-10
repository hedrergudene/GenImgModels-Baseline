#
# Custom fitter (TorchFitterBase backbone from benatools)
#

# Requirements
import torch
from datetime import datetime
import time
import numpy as np
import pandas as pd
from typing import Iterable, Callable, Dict, Tuple
import os


# Helper method (extracted from benatools: https://github.com/benayas1/benatools)
class AverageMeter(object):
    """
    Computes and stores the average and current value
    Attributes
    ----------
    val : float
        Stores the average loss of the last batch
    avg : float
        Average loss
    sum : float
        Sum of all losses
    count : int
        number of elements
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates current internal state
        Parameters
        ----------
        val : float
            loss on each training step
        n : int, Optional
            batch size
        """
        if np.isnan(val) or np.isinf(val):
            return

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Original fitter (extracted from benatools: https://github.com/benayas1/benatools)
class TorchFitterBase:
    """
    Helper class to implement a training loop in PyTorch
    """

    def __init__(self,
                 model: torch.nn.Module = None,
                 device: str = 'cuda',
                 loss: torch.nn.Module = None,
                 optimizer: torch.optim = None,
                 scheduler: torch.optim.lr_scheduler = None,
                 validation_scheduler: bool = True,
                 step_scheduler: bool = False,
                 folder: str = 'models',
                 verbose: bool = True,
                 save_log: bool = True,
                 use_amp: bool = False,
                 ):
        """
        Args:
            model (torch.nn.Module): Model to be fitted
            device (str): Device can be cuda or cpu
            loss (torch.nn.Module): DataFrame to split
            optimizer (torch.optim): Optimizer object
            scheduler (torch.optim.lr_scheduler, optional): Scheduler object. Defaults to None.
            validation_scheduler (bool, optional): Run scheduler step on the validation step. Defaults to True.
            step_scheduler (bool, optional): Run scheduler step on every training step. Defaults to False.
            folder (str, optional): Folder where to store checkpoints. Defaults to 'models'.
            verbose (bool, optional): Whether to print outputs or not. Defaults to True.
            save_log (bool, optional): Whether to write the log in log.txt or not. Defaults to True.
        """
        if loss is not None:
            if type(loss) == type:
                self.loss_function = loss()
            else:
                self.loss_function = loss
        else:
            self.loss_function = None

        self.epoch = 0  # current epoch
        self.verbose = verbose

        self.base_dir = f'{folder}'

        self.save_log = save_log
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_metric = 0

        self.model = model
        self.device = device
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # Optimizer object
        self.optimizer = optimizer

        # Scheduler Object
        self.scheduler = scheduler
        self.validation_scheduler = validation_scheduler  # do scheduler.step after validation stage loss
        self.step_scheduler = step_scheduler  # do scheduler.step after optimizer.step
        self.log(f'Fitter prepared. Device is {self.device}')

    def unpack(self, data):
        raise NotImplementedError('This class is a base class')

    def reduce_loss(self, loss, weights):
        # Apply sample weights if existing
        if len(loss.shape) > 0:
            # apply weights
            if weights is not None:
                loss = loss * torch.unsqueeze(weights, 1)

            # reduction
            loss = loss.mean()
        return loss

    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader = None,
            n_epochs: int = 1,
            metrics: Iterable[Tuple[Callable[[Iterable, Iterable], float], dict]] = None,
            early_stopping: int = 0,
            early_stopping_mode: str = 'min',
            early_stopping_alpha: float = 0.0,
            early_stopping_pct: float = 0.0,
            save_checkpoint: bool = False,
            save_best_checkpoint: bool = True,
            verbose_steps: int = 0,
            callbacks: Iterable[Callable[[Dict], None]] = None):
        """
        Fits a model
        Args:
            train_loader (torch.utils.data.DataLoader): Training data
            val_loader (torch.utils.data.DataLoader, optional): Validation Data. Defaults to None.
            n_epochs (int, optional): Maximum number of epochs to train. Defaults to 1.
            metrics ( function with (y_true, y_pred, **metric_kwargs) signature, optional): Metric to evaluate results on. Defaults to None.
            metric_kwargs (dict, optional): Arguments for the passed metric. Ignored if metric is None. Defaults to {}.
            early_stopping (int, optional): Early stopping epochs. Defaults to 0.
            early_stopping_mode (str, optional): Min or max criteria. Defaults to 'min'.
            early_stopping_alpha (float, optional): Value that indicates how much to improve to consider early stopping. Defaults to 0.0.
            early_stopping_pct (float, optional): Value between 0 and 1 that indicates how much to improve to consider early stopping. Defaults to 0.0.
            save_checkpoint (bool, optional): Whether to save the checkpoint when training. Defaults to False.
            save_best_checkpoint (bool, optional): Whether to save the best checkpoint when training. Defaults to True.
            verbose_steps (int, optional): Number of step to print every training summary. Defaults to 0.
            callbacks (list of callable, optional): List of callback functions to be called after an epoch
        Returns:
            pd.DataFrame: DataFrame containing training history
        """
        if self.model is None or self.loss_function is None or self.optimizer is None:
            self.log(f"ERROR: Either model, loss function or optimizer is not existing.")
            raise ValueError(f"ERROR: Either model, loss function or optimizer is not existing.")

        if self.best_metric == 0.0:
            self.best_metric = np.inf if early_stopping_mode == 'min' else -np.inf

        initial_epochs = self.epoch

        # Use the same train loader for validation. A possible use case is for autoencoders
        if isinstance(val_loader, str) and val_loader == 'training':
            val_loader = train_loader

        training_history = []
        es_epochs = 0
        for e in range(n_epochs):
            history = {'epoch': e}  # training history log for this epoch

            # Update log
            lr = self.optimizer.param_groups[0]['lr']
            self.log(f'\n{datetime.utcnow().isoformat(" ", timespec="seconds")}\n \
                        EPOCH {str(self.epoch+1)}/{str(n_epochs+initial_epochs)} - LR: {lr}')

            # Run one training epoch
            t = time.time()
            train_summary_loss = self.train_one_epoch(train_loader, verbose_steps=verbose_steps)
            history['train'] = train_summary_loss.avg  # training loss
            history['lr'] = self.optimizer.param_groups[0]['lr']

            # Save checkpoint
            if save_checkpoint:
                self.save(f'{self.base_dir}/last-checkpoint.bin', False)

            if val_loader is not None:
                # Run epoch validation
                val_summary_loss, calculated_metrics = self.validation(val_loader,
                                                                       metrics=metrics,
                                                                       verbose_steps=verbose_steps)
                history['val'] = val_summary_loss.avg  # validation loss

                # Write log
                metric_log = ' - ' + ' - '.join([f'{fname}: {value}' for value, fname in calculated_metrics]) if calculated_metrics else ''
                self.log(f'\r[RESULT] {(time.time() - t):.2f}s - train loss: {train_summary_loss.avg:.5f} - val loss: {val_summary_loss.avg:.5f}' + metric_log)

                if calculated_metrics:
                    history.update({fname: value for value, fname in calculated_metrics})
                    #history['val_metric'] = calculated_metrics

                calculated_metric = calculated_metrics[0][0] if calculated_metrics else val_summary_loss.avg
            else:
                # If no validation is provided, training loss is used as metric
                calculated_metric = train_summary_loss.avg

            es_pct = early_stopping_pct * self.best_metric

            # Check if result is improved, then save model
            if (
                ((metrics) and
                 (
                  ((early_stopping_mode == 'max') and (calculated_metric - max(early_stopping_alpha, es_pct) > self.best_metric)) or
                  ((early_stopping_mode == 'min') and (calculated_metric + max(early_stopping_alpha, es_pct) < self.best_metric))
                 )
                ) or
                ((metrics is None) and
                 (calculated_metric + max(early_stopping_alpha, es_pct) < self.best_metric) # the standard case is to minimize
                )
               ):
                self.log(f'Validation metric improved from {self.best_metric} to {calculated_metric}')
                self.best_metric = calculated_metric
                self.model.eval()
                if save_best_checkpoint:
                    savepath = f'{self.base_dir}/best-checkpoint.bin'
                    self.save(savepath)
                es_epochs = 0  # reset early stopping count
            else:
                es_epochs += 1  # increase epoch count with no improvement, for early stopping check

            # Callbacks receive the history dict of this epoch
            if callbacks is not None:
                if not isinstance(callbacks, list):
                    callbacks = [callbacks]
                for c in callbacks:
                    c(history)

            # Check if Early Stopping condition is met
            if (early_stopping > 0) & (es_epochs >= early_stopping):
                self.log(f'Early Stopping: {early_stopping} epochs with no improvement')
                training_history.append(history)
                break

            # Scheduler step after validation
            if self.validation_scheduler and self.scheduler is not None:
                self.scheduler.step(metrics=calculated_metric)

            training_history.append(history)
            self.epoch += 1

        return pd.DataFrame(training_history).set_index('epoch')

    def train_one_epoch(self, train_loader, verbose_steps=0):
        """
        Run one epoch on the train dataset
        Parameters
        ----------
        train_loader : torch.data.utils.DataLoader
            DataLoaders containing the training dataset
        verbose_steps : int, defaults to 0
            number of step to print every training summary
        Returns
        -------
        AverageMeter
            Object with this epochs's average loss
        """
        self.model.train()  # set train mode
        summary_loss = AverageMeter()  # object to track the average loss
        t = time.time()
        batch_size = train_loader.batch_size

        # run epoch
        for step, data in enumerate(train_loader):
            if self.verbose & (verbose_steps > 0):
                if step % verbose_steps == 0:
                    print(
                        f'\rTrain Step {step}/{len(train_loader)} | ' +
                        f'summary_loss: {summary_loss.avg:.5f} | ' +
                        f'time: {(time.time() - t):.2f} secs | ' +
                        f'ETA: {(len(train_loader)-step)*(time.time() - t)/(step+1):.2f}', end=''
                    )
            # Unpack batch of data
            x, y, w = self.unpack(data)

            # Run one batch
            loss = self.train_one_batch(x, y, w)

            summary_loss.update(loss.detach().item(), batch_size)

            # update optimizer using mixed precision if requested
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # LR Scheduler step after epoch
            if self.step_scheduler and self.scheduler is not None:
                self.scheduler.step()

        self.log(f'\r[TRAIN] {(time.time() - t):.2f}s - train loss: {summary_loss.avg:.5f}')

        return summary_loss

    def train_one_batch(self, x, y, w=None):
        """
        Trains one batch of data.
        The actions to be done here are:
        - extract x and y (labels)
        - calculate output and loss
        - backpropagate
        Args:
            x (List or Tuple or Dict): Data
            y (torch.Tensor): Labels
            w (torch.Tensor, optional): Weights. Defaults to None.
        Returns:
            torch.Tensor: A tensor with the calculated loss
        """
        self.optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # Output and loss
            if isinstance(x, tuple) or isinstance(x, list):
                output = self.model(*x)
            elif isinstance(x, dict):
                output = self.model(**x)
            else:
                output = self.model(x)

            loss = self.loss_function(output, y)

            # Reduce loss and apply sample weights if existing
            loss = self.reduce_loss(loss, w)
        
        # backpropagation
        self.scaler.scale(loss).backward()


        return loss

    def validation(self, val_loader, metrics=None, verbose_steps=0):
        """
        Validates a model
        Parameters
        ----------
        val_loader : torch.utils.data.DataLoader
            Validation Data
        metrics : function with (y_true, y_pred, **metric_kwargs) signature
            Metric to evaluate results on
        metric_kwargs : dict
            Arguments for the passed metric. Ignored if metric is None
        verbose_steps : int, defaults to 0
            number of step to print every training summary
        Returns
        -------
        AverageMeter
            Object with this epochs's average loss
        float
            Calculated metric if a metric is provided, else None
        """
        if self.model is None or self.loss_function is None or self.optimizer is None:
            self.log(f"ERROR: Either model, loss function or optimizer is not existing.")
            raise ValueError(f"ERROR: Either model, loss function or optimizer is not existing.")

        self.model.eval()
        summary_loss = AverageMeter()
        y_preds = []
        y_true = []
        batch_size = val_loader.batch_size

        t = time.time()
        for step, data in enumerate(val_loader):
            if self.verbose & (verbose_steps > 0):
                if step % verbose_steps == 0:
                    print(
                        f'\rVal Step {step}/{len(val_loader)} | ' +
                        f'summary_loss: {summary_loss.avg:.5f} | ' +
                        f'time: {(time.time() - t):.2f} secs |' +
                        f'ETA: {(len(val_loader)-step)*(time.time() - t)/(step+1):.2f}', end=''
                    )
            with torch.no_grad():  # no gradient update
                x, y, w = self.unpack(data)

                if metrics:
                    y_true += y.cpu().numpy().tolist()

                # just forward pass
                if isinstance(x, tuple) or isinstance(x, list):
                    output = self.model(*x)
                elif isinstance(x, dict):
                    output = self.model(**x)
                else:
                    output = self.model(x)

                loss = self.loss_function(output, y)

                # Reduce loss and apply sample weights if existing
                loss = self.reduce_loss(loss, w)
                summary_loss.update(loss.detach().item(), batch_size)

                if metrics:
                    y_preds += output.cpu().numpy().tolist()

        # Callback metrics
        metric_log = ' '*30
        if metrics:
            calculated_metrics = []
            y_pred = np.argmax(y_preds, axis=1)
            for f, args in metrics:
                value = f(y_true, y_pred, **args)
                calculated_metrics.append((value, f.__name__))
                metric_log = f'- {f.__name__} {value:.5f} '
        else:
            calculated_metrics = None

        self.log(f'\r[VALIDATION] {(time.time() - t):.2f}s - val. loss: {summary_loss.avg:.5f} ' + metric_log)
        return summary_loss, calculated_metrics

    def predict(self, test_loader, verbose_steps=0):
        """
        Makes predictions using the trained model
        Parameters
        ----------
        test_loader : torch.utils.data.DataLoader
            Test Data
        verbose_steps : int, defaults to 0
            number of step to print every training summary
        Returns
        -------
        np.array
            Predicted values by the model
        """
        if self.model is None:
            self.log(f"ERROR: Model is not existing.")
            raise ValueError(f"ERROR: Model is not existing.")

        self.model.eval()
        y_preds = []
        t = time.time()

        for step, data in enumerate(test_loader):
            if self.verbose & (verbose_steps > 0) > 0:
                if step % verbose_steps == 0:
                    print(
                        f'\rPrediction Step {step}/{len(test_loader)} | ' +
                        f'time: {(time.time() - t):.2f} secs |' +
                        f'ETA: {(len(test_loader)-step)*(time.time() - t)/(step+1):.2f}', end=''
                    )
            with torch.no_grad():  # no gradient update
                x, _, _ = self.unpack(data)

                # Output
                if isinstance(x, tuple) or isinstance(x, list):
                    output = self.model(*x)
                elif isinstance(x, dict):
                    output = self.model(**x)
                else:
                    output = self.model(x)

                y_preds += output.cpu().numpy().tolist()

        return np.array(y_preds)

    def save(self, path, verbose=True):
        """
        Save model and other metadata
        Args:
            path (str): Path of the file to be saved
            verbose (bool, optional): True = print logs, False = silence. Defaults to True.
        """

        if verbose:
            self.log(f'Checkpoint is saved to {path}')
        self.model.eval()

        data = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_summary_loss': self.best_metric,
                'epoch': self.epoch,
                'scaler': self.scaler.state_dict()
        }

        if self.scheduler is not None:
            data['scheduler_state_dict'] = self.scheduler.state_dict()

        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        torch.save(data, path)

    def load(self, path, only_model=False):
        """
        Load model and other metadata
        Args:
            path (str): Path of the file to be loaded
            only_model (bool, optional): Whether to load just the model weights. Defaults to False.
        """
        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if only_model:
            return

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint["scaler"])

        self.best_metric = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1

        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    @staticmethod
    def load_model_weights(path, model):
        """
        Static method that loads weights into a torch module, extracted from a checkpoint
        Args:
            path (str): Path containing the weights. Normally a .bin or .tar file
            model (torch.nn.Module): Module to load the weights on
        Returns:
            torch.nn.Module: The input model with loaded weights
        """

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def log(self, message):
        """
        Log training ouput into console and file
        Args:
            message (str): Message to be logged
        """
        if self.verbose:
            print(message)

        if self.save_log is True:
            if not os.path.exists(self.base_dir):
                os.makedirs(self.base_dir)
            with open(self.log_path, 'a+') as logger:
                logger.write(f'{message}\n')



#
# GAN
#

class GANFitter(TorchFitterBase):
    def __init__(self,
                 model: torch.nn.Module = None,
                 device: str = 'cuda',
                 discr_steps:int=5,
                 gp_weight:float=10.,
                 gen_optimizer: torch.optim = None,
                 discr_optimizer: torch.optim = None,
                 gen_scheduler: torch.optim.lr_scheduler = None,
                 discr_scheduler: torch.optim.lr_scheduler = None,
                 validation_scheduler: bool = True,
                 step_scheduler: bool = False,
                 folder: str = 'models',
                 verbose: bool = True,
                 save_log: bool = True,
                 use_amp: bool = False,
                 ):
        """
        Args:
            model (torch.nn.Module): Model to be fitted
            device (str): Device can be cuda or cpu
            optimizer (torch.optim): Optimizer object
            scheduler (torch.optim.lr_scheduler, optional): Scheduler object. Defaults to None.
            validation_scheduler (bool, optional): Run scheduler step on the validation step. Defaults to True.
            step_scheduler (bool, optional): Run scheduler step on every training step. Defaults to False.
            folder (str, optional): Folder where to store checkpoints. Defaults to 'models'.
            verbose (bool, optional): Whether to print outputs or not. Defaults to True.
            save_log (bool, optional): Whether to write the log in log.txt or not. Defaults to True.
        """

        self.epoch = 0  # current epoch
        self.discr_steps = discr_steps
        self.gp_weight = gp_weight
        self.verbose = verbose

        self.base_dir = f'{folder}'

        self.save_log = save_log
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_metric = 0

        self.model = model
        self.device = device
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # Optimizer object
        self.gen_optimizer = gen_optimizer
        self.discr_optimizer = discr_optimizer

        # Scheduler Object
        self.gen_scheduler = gen_scheduler
        self.discr_scheduler = discr_scheduler
        self.validation_scheduler = validation_scheduler  # do scheduler.step after validation stage loss
        self.step_scheduler = step_scheduler  # do scheduler.step after optimizer.step
        self.log(f'Fitter prepared. Device is {self.device}')


    # Return data in expected format
    def unpack(self, data):
        # extract x and y from the dataloader
        x = data['x']

        # send them to device
        x = x.to(self.device)
        x = x.float()

        return x


    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader = None,
            n_epochs: int = 1,
            metrics: Iterable[Tuple[Callable[[Iterable, Iterable], float], dict]] = None,
            early_stopping: int = 0,
            early_stopping_mode: str = 'min',
            early_stopping_alpha: float = 0.0,
            early_stopping_pct: float = 0.0,
            save_checkpoint: bool = False,
            save_best_checkpoint: bool = True,
            verbose_steps: int = 0,
            callbacks: Iterable[Callable[[Dict], None]] = None):
        """
        Fits a model
        Args:
            train_loader (torch.utils.data.DataLoader): Training data
            val_loader (torch.utils.data.DataLoader, optional): Validation Data. Defaults to None.
            n_epochs (int, optional): Maximum number of epochs to train. Defaults to 1.
            metrics ( function with (y_true, y_pred, **metric_kwargs) signature, optional): Metric to evaluate results on. Defaults to None.
            metric_kwargs (dict, optional): Arguments for the passed metric. Ignored if metric is None. Defaults to {}.
            early_stopping (int, optional): Early stopping epochs. Defaults to 0.
            early_stopping_mode (str, optional): Min or max criteria. Defaults to 'min'.
            early_stopping_alpha (float, optional): Value that indicates how much to improve to consider early stopping. Defaults to 0.0.
            early_stopping_pct (float, optional): Value between 0 and 1 that indicates how much to improve to consider early stopping. Defaults to 0.0.
            save_checkpoint (bool, optional): Whether to save the checkpoint when training. Defaults to False.
            save_best_checkpoint (bool, optional): Whether to save the best checkpoint when training. Defaults to True.
            verbose_steps (int, optional): Number of step to print every training summary. Defaults to 0.
            callbacks (list of callable, optional): List of callback functions to be called after an epoch
        Returns:
            pd.DataFrame: DataFrame containing training history
        """
        if self.model is None or self.gen_optimizer is None or self.discr_optimizer is None:
            self.log(f"ERROR: Either model or optimizers are not existing.")
            raise ValueError(f"ERROR: Either model or optimizers are not existing.")

        if self.best_metric == 0.0:
            self.best_metric = np.inf if early_stopping_mode == 'min' else -np.inf

        initial_epochs = self.epoch

        # Use the same train loader for validation. A possible use case is for autoencoders
        if isinstance(val_loader, str) and val_loader == 'training':
            val_loader = train_loader

        training_history = []
        es_epochs = 0
        for e in range(n_epochs):
            history = {'epoch': e}  # training history log for this epoch

            # Update log
            lr = self.gen_optimizer.param_groups[0]['lr']
            self.log(f'\n{datetime.utcnow().isoformat(" ", timespec="seconds")}\n \
                        EPOCH {str(self.epoch+1)}/{str(n_epochs+initial_epochs)} - LR: {lr}')

            # Run one training epoch
            t = time.time()
            train_summary_loss = self.train_one_epoch(train_loader, verbose_steps=verbose_steps)
            history['train_generator_loss'] = train_summary_loss['gen_loss'].avg
            history['train_discriminator_loss'] = train_summary_loss['discr_loss'].avg
            history['generator_lr'] = self.gen_optimizer.param_groups[0]['lr']
            history['discriminator_lr'] = self.discr_optimizer.param_groups[0]['lr']

            # Save checkpoint
            if save_checkpoint:
                self.save(f'{self.base_dir}/last-checkpoint.bin', False)

            if val_loader is not None:
                # Run epoch validation
                val_summary_loss, calculated_metrics = self.validation(val_loader,
                                                                       metrics=metrics,
                                                                       verbose_steps=verbose_steps)
                history['val_generator_loss'] = val_summary_loss['gen_loss'].avg
                history['val_discriminator_loss'] = val_summary_loss['discr_loss'].avg

                # Write log
                metric_log = ' - ' + ' - '.join([f'{fname}: {value}' for value, fname in calculated_metrics]) if calculated_metrics else ''
                self.log(f"\r[RESULT] {(time.time() - t):.2f}s -"\
                         f"train generator loss: {train_summary_loss['gen_loss'].avg:.5f} - "\
                         f"train discriminator loss: {train_summary_loss['discr_loss'].avg:.5f} - "\
                         f"val generator loss: {val_summary_loss['gen_loss'].avg:.5f} - "\
                         f"val discriminator loss: {val_summary_loss['discr_loss'].avg:.5f}" + metric_log)

                if calculated_metrics:
                    history.update({fname: value for value, fname in calculated_metrics})
                    #history['val_metric'] = calculated_metrics

                calculated_metric = calculated_metrics[0][0] if calculated_metrics else .5*(val_summary_loss['gen_loss'].avg + val_summary_loss['discr_loss'].avg)
            else:
                # If no validation is provided, training loss is used as metric
                calculated_metric = .5*(train_summary_loss['gen_loss'].avg + train_summary_loss['discr_loss'].avg)

            es_pct = early_stopping_pct * self.best_metric

            # Check if result is improved, then save model
            if (
                ((metrics) and
                 (
                  ((early_stopping_mode == 'max') and (calculated_metric - max(early_stopping_alpha, es_pct) > self.best_metric)) or
                  ((early_stopping_mode == 'min') and (calculated_metric + max(early_stopping_alpha, es_pct) < self.best_metric))
                 )
                ) or
                ((metrics is None) and
                 (calculated_metric + max(early_stopping_alpha, es_pct) < self.best_metric) # the standard case is to minimize
                )
               ):
                self.log(f'Validation metric improved from {self.best_metric} to {calculated_metric}')
                self.best_metric = calculated_metric
                self.model.eval()
                if save_best_checkpoint:
                    savepath = f'{self.base_dir}/best-checkpoint.bin'
                    self.save(savepath)
                es_epochs = 0  # reset early stopping count
            else:
                es_epochs += 1  # increase epoch count with no improvement, for early stopping check

            # Callbacks receive the history dict of this epoch and the model
            if callbacks is not None:
                if not isinstance(callbacks, list):
                    callbacks = [callbacks]
                for c in callbacks:
                    c(history, self.model)

            # Check if Early Stopping condition is met
            if (early_stopping > 0) & (es_epochs >= early_stopping):
                self.log(f'Early Stopping: {early_stopping} epochs with no improvement')
                training_history.append(history)
                break

            # Scheduler step after validation
            if self.validation_scheduler and self.gen_scheduler is not None and self.discr_scheduler is not None:
                self.gen_scheduler.step(metrics=val_summary_loss['gen_loss'].avg)
                self.discr_scheduler.step(metrics=val_summary_loss['discr_loss'].avg)

            training_history.append(history)
            self.epoch += 1

        return pd.DataFrame(training_history).set_index('epoch')


    def train_one_epoch(self, train_loader, verbose_steps=0):
        """
        Run one epoch on the train dataset
        Parameters
        ----------
        train_loader : torch.data.utils.DataLoader
            DataLoaders containing the training dataset
        verbose_steps : int, defaults to 0
            number of step to print every training summary
        Returns
        -------
        AverageMeter
            Object with this epochs's average loss
        """
        self.model.train()  # set train mode
        generator_loss = AverageMeter()  # object to track the average loss
        discriminator_loss = AverageMeter()  # object to track the average loss
        t = time.time()

        # run epoch
        for step, data in enumerate(train_loader):
            if self.verbose & (verbose_steps > 0):
                if step % verbose_steps == 0:
                    print(
                        f'\rTrain Step {step}/{len(train_loader)} | ' +
                        f'generator_loss: {generator_loss.avg:.5f} | ' +
                        f'discriminator_loss: {discriminator_loss.avg:.5f} | ' +
                        f'time: {(time.time() - t):.2f} secs | ' +
                        f'ETA: {(len(train_loader)-step)*(time.time() - t)/(step+1):.2f}', end=''
                    )
            # Unpack batch of data
            x = self.unpack(data)

            # Run one batch
            generator_loss, discriminator_loss = self.train_one_batch(x, generator_loss, discriminator_loss)

            # LR Scheduler step after epoch
            if self.step_scheduler and self.gen_scheduler is not None and self.discr_scheduler is not None:
                self.gen_scheduler.step()
                self.discr_scheduler.step()

        self.log(f"\r[TRAIN] {(time.time() - t):.2f}s - "\
                 f"train generator loss: {generator_loss.avg:.5f} - "\
                 f"train discriminator loss: {discriminator_loss.avg:.5f}")

        return {'gen_loss':generator_loss, 'discr_loss':discriminator_loss}


    def train_one_batch(self, x, generator_loss, discriminator_loss):
        """
        Trains one batch of data.
        The actions to be done here are:
        - extract x
        - calculate output and loss
        - backpropagate
        Args:
            x (List or Tuple or Dict): Data
            generator_loss (AverageMeter): Object to collect loss.
            discriminator_loss (AverageMeter): Object to collect loss.
        Returns:
            Dict: A dictionary with updated AverageMeter objects
        """
        #
        # Parameters
        #
        batch_size = x.shape[0]
        #
        # Discriminator training
        #
        for _ in range(self.discr_steps):

            self.discr_optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # Output and loss
                if isinstance(x, tuple) or isinstance(x, list):
                    output = self.model(*x)
                elif isinstance(x, dict):
                    output = self.model(**x)
                else:
                    output = self.model(x)
                # Assemble labels discriminating real from fake images
                discr_labels = torch.cat([torch.ones((batch_size, 1)), torch.zeros((batch_size, 1))], axis=0).to(self.device)
                # Add random noise to the labels - important trick!
                discr_labels = discr_labels + 0.05 * torch.rand(discr_labels.shape).to(self.device)
                discr_loss = torch.nn.functional.binary_cross_entropy_with_logits(output['discr_preds'], discr_labels) + self.gradient_penalty(x, output['generated_images'])
            # Backpropagation
            self.scaler.scale(discr_loss).backward()
            # Update loss gatherer
            discriminator_loss.update(discr_loss.detach().item(), 2*batch_size)
            # Update optimizer using mixed precision if requested
            self.scaler.step(self.discr_optimizer)
            self.scaler.update()
        #
        # Generator training
        #

        self.gen_optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # Output and loss
            if isinstance(x, tuple) or isinstance(x, list):
                output = self.model(*x)
            elif isinstance(x, dict):
                output = self.model(**x)
            else:
                output = self.model(x)
        gen_labels = torch.ones((batch_size, 1)).to(self.device)
        gen_loss = torch.nn.functional.binary_cross_entropy_with_logits(output['gen_preds'], gen_labels)
        self.scaler.scale(gen_loss).backward()
        generator_loss.update(gen_loss.detach().item(), batch_size)
        self.scaler.step(self.gen_optimizer)
        self.scaler.update()

        return generator_loss, discriminator_loss


    # Adapt validation step to generator and discriminator losses
    def validation(self, val_loader, metrics=None, verbose_steps=0):
        """
        Validates a model
        Parameters
        ----------
        val_loader : torch.utils.data.DataLoader
            Validation Data
        metric : function with (y_true, y_pred, **metric_kwargs) signature
            Metric to evaluate results on
        metric_kwargs : dict
            Arguments for the passed metric. Ignored if metric is None
        verbose_steps : int, defaults to 0
            number of step to print every training summary
        Returns
        -------
        AverageMeter
            Object with this epochs's average loss
        float
            Calculated metric if a metric is provided, else None
        """
        if self.model is None or self.gen_optimizer is None or self.discr_optimizer is None:
            self.log(f"ERROR: Either model or optimizers are not existing.")
            raise ValueError(f"ERROR: Either model or optimizers are not existing.")

        self.model.eval()
        gen_loss = AverageMeter()
        discr_loss = AverageMeter()
        batch_metrics = {f.__name__: [] for f in metrics}
        batch_size = val_loader.batch_size

        t = time.time()
        for step, data in enumerate(val_loader):
            if self.verbose & (verbose_steps > 0):
                if step % verbose_steps == 0:
                    print(
                        f'\rVal Step {step}/{len(val_loader)} | ' +
                        f'generator_loss: {gen_loss.avg:.5f} | ' +
                        f'discriminator_loss: {discr_loss.avg:.5f} | ' +
                        f'time: {(time.time() - t):.2f} secs |' +
                        f'ETA: {(len(val_loader)-step)*(time.time() - t)/(step+1):.2f}', end=''
                    )
            with torch.no_grad():  # no gradient update
                batch, w = self.unpack(data)

                # just forward pass
                if isinstance(batch, tuple) or isinstance(batch, list):
                    output = self.model(*batch)
                elif isinstance(batch, dict):
                    output = self.model(**batch)
                else:
                    output = self.model(batch)
            # Gen loss
            batch_size = batch.shape[0]
            gen_loss = self.loss_function(output['gen_preds'], torch.ones((batch_size, 1)))
            # Assemble labels discriminating real from fake images
            discr_labels = torch.cat([torch.ones((batch_size, 1)), torch.zeros((batch_size, 1))], axis=0)
            # Add random noise to the labels - important trick!
            discr_labels = discr_labels + 0.05 * torch.rand(discr_labels.shape)
            discr_loss = self.loss_function(output['discr_preds'], discr_labels)
            
            # Reduce loss and apply sample weights if existing
            gen_loss = self.reduce_loss(gen_loss, w)
            discr_loss = self.reduce_loss(discr_loss, w)

            if metrics is not None:
                for f in metrics:
                    batch_metrics[f.__name__].append(f.forward(output, batch))

        # Callback metrics
        metric_log = ' '*30
        if metrics is not None:
            calculated_metrics = []
            for f in metrics:
                value = torch.Tensor(batch_metrics[f.__name__]).mean().item()
                calculated_metrics.append((value, f.__name__))
                metric_log = f'- {f.__name__} {value:.5f}'
        else:
            calculated_metrics = None

        self.log(f"\r[VALIDATION] {(time.time() - t):.2f}s - "\
                 f"val generator loss: {gen_loss.avg:.5f} "\
                 f"val discriminator loss: {discr_loss.avg:.5f} " + metric_log)
        return {'gen_loss':gen_loss, 'discr_loss':discr_loss}, calculated_metrics


    def gradient_penalty(self,
                         real_images:torch.Tensor,
                         fake_images:torch.Tensor,
                         ):
        batch_size = real_images.size()[0]
        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_images).to(self.device)
        interpolated = alpha * real_images + (1 - alpha) * fake_images
        interpolated = torch.autograd.Variable(interpolated, requires_grad=True).to(self.device)

        # Calculate probability of interpolated examples
        prob_interpolated = self.model.discriminator(interpolated)

        # Calculate gradients of probabilities with respect to examples
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                            grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                                            create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()
