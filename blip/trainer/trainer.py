"""
Class for a generic model trainer.
"""
import torch
import numpy as np
import os
from tqdm import tqdm
from blip.utils.logger import Logger
from blip.losses import LossHandler
from blip.models import ModelHandler, ModelChecker
from blip.metrics import MetricHandler
from blip.optimizers import Optimizer
from blip.utils.timing import Timers
from blip.utils.memory import MemoryTrackers
from blip.utils.callbacks import CallbackHandler
from blip.utils.callbacks import TimingCallback, MemoryTrackerCallback


class Trainer:
    """
    This class is an attempt to reduce code rewriting by putting together
    a set of functions that do everything that we could need with
    respect to training.  There are a few objects which must be passed
    to the trainer, which include:
        (a) model     - an object which inherits from nn.Module
        (b) criterion - an object which has a defined function called "loss"
        (c) optimizer - some choice of optimizer, e.g. Adam
        (d) metrics   - (optional) an object which has certain defined functions
        (e) callbacks - (optional) an object which has certain defined functions
    """
    def __init__(
        self,
        name:       str = 'default',
        config:     dict = {},
        meta:   dict = {},
        seed:   int = 0,
    ):
        self.name = name + '_trainer'
        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']
        else:
            self.device = 'cpu'
        if meta['verbose']:
            self.logger = Logger(self.name, output="both", file_mode="w")
        else:
            self.logger = Logger(self.name, level='warning', file_mode="w")
        self.config = config
        # check for devices
        self.gpu = self.meta['gpu']
        self.seed = seed

        self.process_config()

    def process_config(self):
        self.logger.info("constructing model trainer.")
        self.process_model()
        self.process_directories()
        self.process_criterion()
        self.process_optimizer()
        self.process_metrics()
        self.process_callbacks()
        self.process_consistency_check()

    def process_model(self):
        if "model" not in self.meta:
            self.logger.error('no model specified in meta!')
        if not isinstance(self.meta['model'], ModelHandler):
            self.logger.error(
                f'specified model is of type {type(self.meta["model"])}, but should be of type ModelHandler!'
            )
        self.model = self.meta['model'].model
        self.model_checker = ModelChecker(self.name)
        self.model.set_device(self.device)

    def process_directories(self):
        # define directories
        if "timing_dir" not in self.config:
            self.config["timing_dir"] = f'{self.meta["local_scratch"]}/plots/{self.model.name}/timing/'
        self.meta['timing_dir'] = self.config['timing_dir']
        if not os.path.isdir(self.config['timing_dir']):
            self.logger.info(f"creating timing directory {self.config['timing_dir']}")
            os.makedirs(self.config['timing_dir'])
        if "memory_dir" not in self.config:
            self.config["memory_dir"] = f'{self.meta["local_scratch"]}/plots/{self.model.name}/timing/'
        self.meta['memory_dir'] = self.config['memory_dir']
        if not os.path.isdir(self.config['memory_dir']):
            self.logger.info(f"creating timing directory {self.config['memory_dir']}")
            os.makedirs(self.config['memory_dir'])

    def process_criterion(self):
        if "criterion" not in self.meta:
            self.logger.error('no criterion specified in meta!')
        if not isinstance(self.meta['criterion'], LossHandler):
            self.logger.error(
                f'specified criterion is of type {type(self.meta["criterion"])}, but should be of type LossHandler!'
            )
        self.criterion = self.meta['criterion']
        self.criterion.set_device(self.device)

    def process_optimizer(self):
        if "optimizer" not in self.meta:
            self.logger.error('no optimizer specified in meta!')
        if not isinstance(self.meta['optimizer'], Optimizer):
            self.logger.error(
                f'specified optimizer is of type {type(self.meta["optimizer"])}, but should be of type Optimizer!'
            )
        self.optimizer = self.meta['optimizer']

    def process_metrics(self):
        if "metrics" not in self.meta:
            self.logger.error('no metrics specified in meta!')
        if not isinstance(self.meta['metrics'], MetricHandler):
            if not isinstance(self.meta['metrics'], None):
                self.logger.error(
                    f'specified metrics is of type {type(self.meta["metrics"])}, but should be of type MetricHandler or None!'
                )
        self.metrics = self.meta['metrics']
        if self.metrics is not None:
            self.metrics.set_device(self.device)

    def process_callbacks(self):
        if "callbacks" not in self.meta:
            self.logger.error('no callbacks specified in meta!')
        if not isinstance(self.meta['callbacks'], CallbackHandler):
            if not isinstance(self.meta['callbacks'], None):
                self.logger.error(
                    f'specified callbacks is of type {type(self.meta["callbacks"])}, ' +
                    'but should be of type CallbackHandler or None!'
                )
        self.callbacks = self.meta['callbacks']
        if self.callbacks is None:
            # add generic callbacks
            self.callbacks = CallbackHandler(
                name="default"
            )
        # add timing info
        self.timers = Timers(gpu=self.gpu)
        self.timer_callback = TimingCallback(
            output_dir=self.meta['timing_dir'],
            timers=self.timers
        )
        self.callbacks.add_callback(self.timer_callback)

        # add memory info
        self.memory_trackers = MemoryTrackers(gpu=self.gpu)
        self.memory_callback = MemoryTrackerCallback(
            output_dir=self.meta['memory_dir'],
            memory_trackers=self.memory_trackers
        )
        self.callbacks.add_callback(self.memory_callback)

    def process_consistency_check(self):
        # run consistency check
        self.logger.info("running consistency check...")
        self.shapes = self.model_checker.run_consistency_check(
            dataset_loader=self.meta['loader'],
            model=self.model,
            criterion=self.criterion,
            metrics=self.metrics
        )

    def save_checkpoint(
        self,
        epoch:  int = 99999
    ):
        if not os.path.exists(f"{self.meta['local_scratch']}/.checkpoints/"):
            os.makedirs(f"{self.meta['local_scratch']}/.checkpoints/")
        torch.save(
            self.model.state_dict(),
            f"{self.meta['local_scratch']}/.checkpoints/checkpoint_{epoch}.ckpt"
        )

    def train(
        self,
        epochs:     int = 100,          # number of epochs to train
        checkpoint: int = 10,           # epochs inbetween weight saving
        progress_bar:   str = 'all',    # progress bar from tqdm
        rewrite_bar:    bool = False,   # wether to leave the bars after each epoch
        save_predictions: bool = True,  # wether to save network outputs for all events to original file
        no_timing:      bool = False,   # wether to keep the bare minimum timing info as a callback
        skip_metrics:   bool = False,   # wether to skip metrics except for testing sets
    ):
        """
        Main training loop.  First, we see if the user wants to omit timing information.
        """
        if (self.model.device != self.device):
            self.logger.error(f"device: '{self.device}' and model device: '{self.model.device}' are different!")
        if (self.criterion.device != self.device):
            self.logger.error(f"device: '{self.device}' and model device: '{self.criterion.device}' are different!")

        self.model.save_model(flag='init')
        # setting values in callbacks
        self.callbacks.set_device(self.device)
        self.callbacks.set_training_info(
            epochs,
            self.meta['loader'].num_train_batches,
            self.meta['loader'].num_validation_batches,
            self.meta['loader'].num_test_batches
        )
        if not skip_metrics and self.metrics is None:
            self.logger.error('skip_metrics set to false in config, but no metrics are specified!')
        # Training
        self.logger.info(f"training dataset '{self.meta['dataset'].name}' for {epochs} epochs.")
        if no_timing:
            # TODO: Need to fix this so that memory and timing callbacks aren't called.
            self.callbacks.callbacks['timing_callback'].no_timing = True
            self.callbacks.callbacks['memory_callback'].no_timing = True
            if self.meta['debug']:
                self.__train_debug(
                    epochs,
                    checkpoint,
                    progress_bar,
                    rewrite_bar,
                    save_predictions,
                    skip_metrics
                )
            else:
                self.__train_no_timing(
                    epochs,
                    checkpoint,
                    progress_bar,
                    rewrite_bar,
                    save_predictions,
                    skip_metrics
                )
        else:
            self.__train_with_timing(
                epochs,
                checkpoint,
                progress_bar,
                rewrite_bar,
                save_predictions,
                skip_metrics
            )

    def report_failure(
        self,
        data,
        outputs,
        batch,
        epoch,
        step,
        exception
    ):
        self.logger.error(
            '\n****RECORDED AN ERROR IN COMPUTATION****\n' +
            f'data:     {data}\n\n' +
            f'outputs:  {outputs}\n' +
            f'epoch:    {epoch}\n' +
            f'batch:    {batch}\n' +
            f'step:     {step}\n' +
            f'exception:    {exception}\n' +
            '****RECORDED AN ERROR IN COMPUTATION****\n'
        )

    def __train_with_timing(
        self,
        epochs:     int = 100,          # number of epochs to train
        checkpoint: int = 10,           # epochs inbetween weight saving
        progress_bar:   str = 'all',    # progress bar from tqdm
        rewrite_bar:    bool = False,   # wether to leave the bars after each epoch
        save_predictions: bool = True,  # wether to save network outputs for all events to original file
        skip_metrics:   bool = False,   # wether to skip metrics except for testing sets.
    ):
        """
        Training usually consists of the following steps:
            (1) Zero-out training/validation/testing losses and metrics
            (2) Loop for N epochs:
                (a) Grab the current batch of (training/validation) data.
                (b) Run the data through the model and calculate losses/metrics.
                (c) Backpropagate the loss (training)
            (3) Evaluate the trained model on testing data.
        """
        # iterate over epochs
        for epoch in range(epochs):
            """
            Training stage.
            Setup the progress bar for the training loop.
            """
            if (progress_bar == 'all' or progress_bar == 'train'):
                training_loop = tqdm(
                    enumerate(self.meta['loader'].train_loader, 0),
                    total=len(self.meta['loader'].train_loader),
                    leave=rewrite_bar,
                    position=0,
                    colour='green'
                )
            else:
                training_loop = enumerate(self.meta['loader'].train_loader, 0)

            # make sure to set model to train() during training!
            self.model.train()
            """
            Setup timing/memory information for epoch.
            """
            self.timers.timers['epoch_training'].start()
            self.memory_trackers.memory_trackers['epoch_training'].start()
            self.timers.timers['training_data'].start()
            self.memory_trackers.memory_trackers['training_data'].start()
            for ii, data in training_loop:
                self.memory_trackers.memory_trackers['training_data'].end()
                self.timers.timers['training_data'].end()
                # zero the parameter gradients
                """
                There are choices here, either one can do:
                    model.zero_grad() or
                    optimizer.zero_grad() or
                    for param in model.parameters():        <== optimal choice
                        param.grad = None
                """
                self.timers.timers['training_zero_grad'].start()
                self.memory_trackers.memory_trackers['training_zero_grad'].start()
                for param in self.model.parameters():
                    param.grad = None
                self.memory_trackers.memory_trackers['training_zero_grad'].end()
                self.timers.timers['training_zero_grad'].end()
                # get the network output
                """
                The forward call takes in the entire data
                stream, which could have multiple inputs needed.
                It's up to the model to determine what to do with it.
                The forward call of the model could send out
                multiple output tensors, depending on the application
                (such as in an AE where the latent space values are
                important). It's up to the loss function to know what to expect.
                """
                self.timers.timers['training_forward'].start()
                self.memory_trackers.memory_trackers['training_forward'].start()
                outputs = self.model(data)
                self.memory_trackers.memory_trackers['training_forward'].end()
                self.timers.timers['training_forward'].end()

                # compute loss
                self.timers.timers['training_loss'].start()
                self.memory_trackers.memory_trackers['training_loss'].start()
                loss = self.criterion.loss(outputs, data)
                self.memory_trackers.memory_trackers['training_loss'].end()
                self.timers.timers['training_loss'].end()

                # backprop
                self.timers.timers['training_loss_backward'].start()
                self.memory_trackers.memory_trackers['training_loss_backward'].start()
                loss.backward()
                self.memory_trackers.memory_trackers['training_loss_backward'].end()
                self.timers.timers['training_loss_backward'].end()

                # record backprop timing
                self.timers.timers['training_backprop'].start()
                self.memory_trackers.memory_trackers['training_backprop'].start()
                self.optimizer.step()
                self.memory_trackers.memory_trackers['training_backprop'].end()
                self.timers.timers['training_backprop'].end()

                # update progress bar
                self.timers.timers['training_progress'].start()
                self.memory_trackers.memory_trackers['training_progress'].start()
                if (progress_bar == 'all' or progress_bar == 'train'):
                    training_loop.set_description(f"Training: Epoch [{epoch+1}/{epochs}]")
                    training_loop.set_postfix_str(f"loss={loss.item():.2e}")
                self.memory_trackers.memory_trackers['training_progress'].end()
                self.timers.timers['training_progress'].end()

                self.timers.timers['training_data'].start()
                self.memory_trackers.memory_trackers['training_data'].start()
            # update timing info
            self.memory_trackers.memory_trackers['epoch_training'].end()
            self.timers.timers['epoch_training'].end()
            if not skip_metrics:
                self.model.eval()
                with torch.no_grad():
                    """
                    Run through a metric loop if there are any metrics
                    defined.
                    """
                    if self.metrics is not None:
                        if (progress_bar == 'all' or progress_bar == 'train'):
                            metrics_training_loop = tqdm(
                                enumerate(self.meta['loader'].train_loader, 0),
                                total=len(self.meta['loader'].train_loader),
                                leave=rewrite_bar,
                                position=0,
                                colour='green'
                            )
                        else:
                            metrics_training_loop = enumerate(self.meta['loader'].train_loader, 0)
                        self.metrics.reset_batch()
                        for ii, data in metrics_training_loop:
                            # update metrics
                            self.timers.timers['training_metrics'].start()
                            self.memory_trackers.memory_trackers['training_metrics'].start()
                            outputs = self.model(data)
                            self.metrics.update(outputs, data, train_type="train")
                            self.memory_trackers.memory_trackers['training_metrics'].end()
                            self.timers.timers['training_metrics'].end()
                            if (progress_bar == 'all' or progress_bar == 'train'):
                                metrics_training_loop.set_description(f"Training Metrics: Epoch [{epoch+1}/{epochs}]")

            # evaluate callbacks
            self.timers.timers['training_callbacks'].start()
            self.memory_trackers.memory_trackers['training_callbacks'].start()
            self.callbacks.evaluate_epoch(train_type='train')
            self.memory_trackers.memory_trackers['training_callbacks'].end()
            self.timers.timers['training_callbacks'].end()

            """
            Validation stage.
            Setup the progress bar for the validation loop.
            """
            if (progress_bar == 'all' or progress_bar == 'validation'):
                validation_loop = tqdm(
                    enumerate(self.meta['loader'].validation_loader, 0),
                    total=len(self.meta['loader'].validation_loader),
                    leave=rewrite_bar,
                    position=0,
                    colour='blue'
                )
            else:
                validation_loop = enumerate(self.meta['loader'].validation_loader, 0)
            # make sure to set model to eval() during validation!
            self.model.eval()
            with torch.no_grad():
                """
                Setup timing information for epoch.
                """
                self.timers.timers['epoch_validation'].start()
                self.memory_trackers.memory_trackers['epoch_validation'].start()
                self.timers.timers['validation_data'].start()
                self.memory_trackers.memory_trackers['validation_data'].start()
                for ii, data in validation_loop:
                    self.memory_trackers.memory_trackers['validation_data'].end()
                    self.timers.timers['validation_data'].end()
                    # get the network output
                    self.timers.timers['validation_forward'].start()
                    self.memory_trackers.memory_trackers['validation_forward'].start()
                    outputs = self.model(data)
                    self.memory_trackers.memory_trackers['validation_forward'].end()
                    self.timers.timers['validation_forward'].end()

                    # compute loss
                    self.timers.timers['validation_loss'].start()
                    self.memory_trackers.memory_trackers['validation_loss'].start()
                    loss = self.criterion.loss(outputs, data)
                    self.memory_trackers.memory_trackers['validation_loss'].end()
                    self.timers.timers['validation_loss'].end()

                    # update progress bar
                    self.timers.timers['validation_progress'].start()
                    self.memory_trackers.memory_trackers['validation_progress'].start()
                    if (progress_bar == 'all' or progress_bar == 'validation'):
                        validation_loop.set_description(f"Validation: Epoch [{epoch+1}/{epochs}]")
                        validation_loop.set_postfix_str(f"loss={loss.item():.2e}")
                    self.memory_trackers.memory_trackers['validation_progress'].end()
                    self.timers.timers['validation_progress'].end()

                    self.timers.timers['validation_data'].start()
                    self.memory_trackers.memory_trackers['validation_data'].start()
                # update timing info
                self.memory_trackers.memory_trackers['epoch_validation'].end()
                self.timers.timers['epoch_validation'].end()
                """
                Run through a metric loop if there are any metrics
                defined.
                """
                if not skip_metrics:
                    if self.metrics is not None:
                        if (progress_bar == 'all' or progress_bar == 'validation'):
                            metrics_validation_loop = tqdm(
                                enumerate(self.meta['loader'].validation_loader, 0),
                                total=len(self.meta['loader'].validation_loader),
                                leave=rewrite_bar,
                                position=0,
                                colour='blue'
                            )
                        else:
                            metrics_validation_loop = enumerate(self.meta['loader'].validation_loader, 0)
                        self.metrics.reset_batch()
                        for ii, data in metrics_validation_loop:
                            # update metrics
                            self.timers.timers['validation_metrics'].start()
                            self.memory_trackers.memory_trackers['validation_metrics'].start()
                            outputs = self.model(data)
                            self.metrics.update(outputs, data, train_type="validation")
                            self.memory_trackers.memory_trackers['validation_metrics'].end()
                            self.timers.timers['validation_metrics'].end()
                            if (progress_bar == 'all' or progress_bar == 'validation'):
                                metrics_validation_loop.set_description(f"Validation Metrics: Epoch [{epoch+1}/{epochs}]")

            # evaluate callbacks
            self.timers.timers['validation_callbacks'].start()
            self.memory_trackers.memory_trackers['validation_callbacks'].start()
            self.callbacks.evaluate_epoch(train_type='validation')
            self.memory_trackers.memory_trackers['validation_callbacks'].end()
            self.timers.timers['validation_callbacks'].end()

            # save weights if at checkpoint step
            if epoch % checkpoint == 0:
                self.save_checkpoint(epoch)
            # free up gpu resources
            torch.cuda.empty_cache()
        # evaluate epoch callbacks
        self.callbacks.evaluate_training()
        self.logger.info("training finished.")
        """
        Testing stage.
        Setup the progress bar for the testing loop.
        We do not have timing information for the test
        loop stage, since it is generally quick
        and doesn't need to be optimized for any reason.
        """
        if (progress_bar == 'all' or progress_bar == 'test'):
            test_loop = tqdm(
                enumerate(self.meta['loader'].test_loader, 0),
                total=len(self.meta['loader'].test_loader),
                leave=rewrite_bar,
                position=0,
                colour='red'
            )
        else:
            test_loop = enumerate(self.meta['loader'].test_loader, 0)
        # make sure to set model to eval() during validation!
        self.model.eval()
        if self.metrics is not None:
            self.metrics.reset_batch()
        with torch.no_grad():
            for ii, data in test_loop:
                # get the network output
                outputs = self.model(data)

                # compute loss
                loss = self.criterion.loss(outputs, data)

                # update metrics
                if self.metrics is not None:
                    self.metrics.update(outputs, data, train_type="test")

                # update progress bar
                if (progress_bar == 'all' or progress_bar == 'test'):
                    test_loop.set_description(f"Testing: Batch [{ii+1}/{self.meta['loader'].num_test_batches}]")
                    test_loop.set_postfix_str(f"loss={loss.item():.2e}")

            # evaluate callbacks
            self.callbacks.evaluate_epoch(train_type='test')
        self.callbacks.evaluate_testing()
        # save the final model
        self.model.save_model(flag='trained')

        # see if predictions should be saved
        if save_predictions:
            self.logger.info("running inference to save predictions.")
            return self.inference(
                dataset_type='all',
                outputs=[output for output in self.shapes["output"].keys()],
                progress_bar=progress_bar,
                rewrite_bar=rewrite_bar,
                save_predictions=True,
            )

    def __train_no_timing(
        self,
        epochs:     int = 100,          # number of epochs to train
        checkpoint: int = 10,           # epochs inbetween weight saving
        progress_bar:   str = 'all',    # progress bar from tqdm
        rewrite_bar:    bool = False,   # wether to leave the bars after each epoch
        save_predictions: bool = True,  # wether to save network outputs for all events to original file
        skip_metrics:   bool = False,   # wether to skip metrics except for testing sets.
    ):
        """
        No comments here since the code is identical to the __train_with_timing function
        except for the lack of calls to timers.
        """
        for epoch in range(epochs):
            if (progress_bar == 'all' or progress_bar == 'train'):
                training_loop = tqdm(
                    enumerate(self.meta['loader'].train_loader, 0),
                    total=len(self.meta['loader'].train_loader),
                    leave=rewrite_bar,
                    position=0,
                    colour='green'
                )
            else:
                training_loop = enumerate(self.meta['loader'].train_loader, 0)
            self.model.train()
            for ii, data in training_loop:
                for param in self.model.parameters():
                    param.grad = None
                outputs = self.model(data)
                loss = self.criterion.loss(outputs, data)
                loss.backward()
                self.optimizer.step()
                if (progress_bar == 'all' or progress_bar == 'train'):
                    training_loop.set_description(f"Training: Epoch [{epoch+1}/{epochs}]")
                    training_loop.set_postfix_str(f"loss={loss.item():.2e}")
            self.model.eval()
            if not skip_metrics:
                if self.metrics is not None:
                    if (progress_bar == 'all' or progress_bar == 'train'):
                        metrics_training_loop = tqdm(
                            enumerate(self.meta['loader'].train_loader, 0),
                            total=len(self.meta['loader'].train_loader),
                            leave=rewrite_bar,
                            position=0,
                            colour='green'
                        )
                    else:
                        metrics_training_loop = enumerate(self.meta['loader'].train_loader, 0)
                    self.metrics.reset_batch()
                    for ii, data in metrics_training_loop:
                        outputs = self.model(data)
                        self.metrics.update(outputs, data, train_type="train")
                        if (progress_bar == 'all' or progress_bar == 'train'):
                            metrics_training_loop.set_description(f"Training Metrics: Epoch [{epoch+1}/{epochs}]")
            self.callbacks.evaluate_epoch(train_type='train')
            if (progress_bar == 'all' or progress_bar == 'validation'):
                validation_loop = tqdm(
                    enumerate(self.meta['loader'].validation_loader, 0),
                    total=len(self.meta['loader'].validation_loader),
                    leave=rewrite_bar,
                    position=0,
                    colour='blue'
                )
            else:
                validation_loop = enumerate(self.meta['loader'].validation_loader, 0)
            self.model.eval()
            with torch.no_grad():
                for ii, data in validation_loop:
                    outputs = self.model(data)
                    loss = self.criterion.loss(outputs, data)
                    if (progress_bar == 'all' or progress_bar == 'validation'):
                        validation_loop.set_description(f"Validation: Epoch [{epoch+1}/{epochs}]")
                        validation_loop.set_postfix_str(f"loss={loss.item():.2e}")
                if not skip_metrics:
                    if self.metrics is not None:
                        if (progress_bar == 'all' or progress_bar == 'validation'):
                            metrics_validation_loop = tqdm(
                                enumerate(self.meta['loader'].validation_loader, 0),
                                total=len(self.meta['loader'].validation_loader),
                                leave=rewrite_bar,
                                position=0,
                                colour='blue'
                            )
                        else:
                            metrics_validation_loop = enumerate(self.meta['loader'].validation_loader, 0)
                        self.metrics.reset_batch()
                        for ii, data in metrics_validation_loop:
                            outputs = self.model(data)
                            self.metrics.update(outputs, data, train_type="validation")
                            if (progress_bar == 'all' or progress_bar == 'validation'):
                                metrics_validation_loop.set_description(f"Validation Metrics: Epoch [{epoch+1}/{epochs}]")
            self.callbacks.evaluate_epoch(train_type='validation')
            if epoch % checkpoint == 0:
                self.save_checkpoint(epoch)
        self.callbacks.evaluate_training()
        self.logger.info("training finished.")
        if (progress_bar == 'all' or progress_bar == 'test'):
            test_loop = tqdm(
                enumerate(self.meta['loader'].test_loader, 0),
                total=len(self.meta['loader'].test_loader),
                leave=rewrite_bar,
                position=0,
                colour='red'
            )
        else:
            test_loop = enumerate(self.meta['loader'].test_loader, 0)
        self.model.eval()
        with torch.no_grad():
            for ii, data in test_loop:
                outputs = self.model(data)
                loss = self.criterion.loss(outputs, data)
                if self.metrics is not None:
                    self.metrics.reset_batch()
                    self.metrics.update(outputs, data, train_type="test")
                if (progress_bar == 'all' or progress_bar == 'test'):
                    test_loop.set_description(f"Testing: Batch [{ii+1}/{self.meta['loader'].num_test_batches}]")
                    test_loop.set_postfix_str(f"loss={loss.item():.2e}")
            self.callbacks.evaluate_epoch(train_type='test')
        self.callbacks.evaluate_testing()
        self.model.save_model(flag='trained')
        if save_predictions:
            self.logger.info("running inference to save predictions.")
            return self.inference(
                dataset_type='all',
                outputs=[output for output in self.shapes["output"].keys()],
                progress_bar=progress_bar,
                rewrite_bar=rewrite_bar,
                save_predictions=True,
            )

    def __train_debug(
        self,
        epochs:     int = 100,          # number of epochs to train
        checkpoint: int = 10,           # epochs inbetween weight saving
        progress_bar:   str = 'all',    # progress bar from tqdm
        rewrite_bar:    bool = False,   # wether to leave the bars after each epoch
        save_predictions: bool = True,  # wether to save network outputs for all events to original file
        skip_metrics:   bool = False,   # wether to skip metrics except for testing sets.
    ):
        """
        No comments here since the code is identical to the __train_with_timing function
        except for the lack of calls to timers.
        """
        for epoch in range(epochs):
            if (progress_bar == 'all' or progress_bar == 'train'):
                training_loop = tqdm(
                    enumerate(self.meta['loader'].train_loader, 0),
                    total=len(self.meta['loader'].train_loader),
                    leave=rewrite_bar,
                    position=0,
                    colour='green'
                )
            else:
                training_loop = enumerate(self.meta['loader'].train_loader, 0)
            self.model.train()
            for ii, data in training_loop:
                for param in self.model.parameters():
                    param.grad = None
                try:
                    outputs = self.model(data)
                except Exception as exception:
                    self.report_failure(
                        data=data,
                        outputs=None,
                        batch=ii,
                        epoch=epoch,
                        step='training model evaluation',
                        exception=exception
                    )
                try:
                    loss = self.criterion.loss(outputs, data)
                except Exception as exception:
                    self.report_failure(
                        data=data,
                        outputs=outputs,
                        batch=ii,
                        epoch=epoch,
                        step='training loss evaluation',
                        exception=exception
                    )
                try:
                    loss.backward()
                except Exception as exception:
                    self.report_failure(
                        data=data,
                        outputs=outputs,
                        batch=ii,
                        epoch=epoch,
                        step='training loss backwards',
                        exception=exception
                    )
                try:
                    self.optimizer.step()
                except Exception as exception:
                    self.report_failure(
                        data=data,
                        outputs=outputs,
                        batch=ii,
                        epoch=epoch,
                        step='training optimizer step',
                        exception=exception
                    )
                if (progress_bar == 'all' or progress_bar == 'train'):
                    training_loop.set_description(f"Training: Epoch [{epoch+1}/{epochs}]")
                    training_loop.set_postfix_str(f"loss={loss.item():.2e}")
            self.model.eval()
            if not skip_metrics:
                if self.metrics is not None:
                    if (progress_bar == 'all' or progress_bar == 'train'):
                        metrics_training_loop = tqdm(
                            enumerate(self.meta['loader'].train_loader, 0),
                            total=len(self.meta['loader'].train_loader),
                            leave=rewrite_bar,
                            position=0,
                            colour='green'
                        )
                    else:
                        metrics_training_loop = enumerate(self.meta['loader'].train_loader, 0)
                    self.metrics.reset_batch()
                    for ii, data in metrics_training_loop:
                        try:
                            outputs = self.model(data)
                        except Exception as exception:
                            self.report_failure(
                                data=data,
                                outputs=None,
                                batch=ii,
                                epoch=epoch,
                                step='training model evaluation for metrics',
                                exception=exception
                            )
                        try:
                            self.metrics.update(outputs, data, train_type="train")
                        except Exception as exception:
                            self.report_failure(
                                data=data,
                                outputs=outputs,
                                batch=ii,
                                epoch=epoch,
                                step='training metric evaluation',
                                exception=exception
                            )
                        if (progress_bar == 'all' or progress_bar == 'train'):
                            metrics_training_loop.set_description(f"Training Metrics: Epoch [{epoch+1}/{epochs}]")
            try:
                self.callbacks.evaluate_epoch(train_type='train')
            except Exception as exception:
                self.report_failure(
                    data=data,
                    outputs=outputs,
                    batch=ii,
                    epoch=epoch,
                    step='training callback epoch evaluation',
                    exception=exception
                )
            if (progress_bar == 'all' or progress_bar == 'validation'):
                validation_loop = tqdm(
                    enumerate(self.meta['loader'].validation_loader, 0),
                    total=len(self.meta['loader'].validation_loader),
                    leave=rewrite_bar,
                    position=0,
                    colour='blue'
                )
            else:
                validation_loop = enumerate(self.meta['loader'].validation_loader, 0)
            self.model.eval()
            with torch.no_grad():
                for ii, data in validation_loop:
                    try:
                        outputs = self.model(data)
                    except Exception as exception:
                        self.report_failure(
                            data=data,
                            outputs=None,
                            batch=ii,
                            epoch=epoch,
                            step='validation model evaluation',
                            exception=exception
                        )
                    try:
                        loss = self.criterion.loss(outputs, data)
                    except Exception as exception:
                        self.report_failure(
                            data=data,
                            outputs=outputs,
                            batch=ii,
                            epoch=epoch,
                            step='validation loss evaluation',
                            exception=exception
                        )
                    if (progress_bar == 'all' or progress_bar == 'validation'):
                        validation_loop.set_description(f"Validation: Epoch [{epoch+1}/{epochs}]")
                        validation_loop.set_postfix_str(f"loss={loss.item():.2e}")
                if not skip_metrics:
                    if self.metrics is not None:
                        if (progress_bar == 'all' or progress_bar == 'validation'):
                            metrics_validation_loop = tqdm(
                                enumerate(self.meta['loader'].validation_loader, 0),
                                total=len(self.meta['loader'].validation_loader),
                                leave=rewrite_bar,
                                position=0,
                                colour='blue'
                            )
                        else:
                            metrics_validation_loop = enumerate(self.meta['loader'].validation_loader, 0)
                        self.metrics.reset_batch()
                        for ii, data in metrics_validation_loop:
                            try:
                                outputs = self.model(data)
                            except Exception as exception:
                                self.report_failure(
                                    data=data,
                                    outputs=None,
                                    batch=ii,
                                    epoch=epoch,
                                    step='validation model evaluation for metrics',
                                    exception=exception
                                )
                            try:
                                self.metrics.update(outputs, data, train_type="train")
                            except Exception as exception:
                                self.report_failure(
                                    data=data,
                                    outputs=outputs,
                                    batch=ii,
                                    epoch=epoch,
                                    step='validation metric evaluation',
                                    exception=exception
                                )
                            if (progress_bar == 'all' or progress_bar == 'validation'):
                                metrics_validation_loop.set_description(f"Validation Metrics: Epoch [{epoch+1}/{epochs}]")
            try:
                self.callbacks.evaluate_epoch(train_type='validation')
            except Exception as exception:
                self.report_failure(
                    data=data,
                    outputs=outputs,
                    batch=ii,
                    epoch=epoch,
                    step='validation callback epoch evaluation',
                    exception=exception
                )

            if epoch % checkpoint == 0:
                self.save_checkpoint(epoch)
        try:
            self.callbacks.evaluate_training()
        except Exception as exception:
            self.report_failure(
                data=data,
                outputs=outputs,
                batch=ii,
                epoch=epoch,
                step='training callback evaluation',
                exception=exception
            )
        self.logger.info("training finished.")
        if (progress_bar == 'all' or progress_bar == 'test'):
            test_loop = tqdm(
                enumerate(self.meta['loader'].test_loader, 0),
                total=len(self.meta['loader'].test_loader),
                leave=rewrite_bar,
                position=0,
                colour='red'
            )
        else:
            test_loop = enumerate(self.meta['loader'].test_loader, 0)
        self.model.eval()
        with torch.no_grad():
            for ii, data in test_loop:
                try:
                    outputs = self.model(data)
                except Exception as exception:
                    self.report_failure(
                        data=data,
                        outputs=None,
                        batch=ii,
                        epoch=-1,
                        step='test model evaluation',
                        exception=exception
                    )
                try:
                    loss = self.criterion.loss(outputs, data)
                except Exception as exception:
                    self.report_failure(
                        data=data,
                        outputs=outputs,
                        batch=ii,
                        epoch=-1,
                        step='test loss evaluation',
                        exception=exception
                    )
                if self.metrics is not None:
                    self.metrics.reset_batch()
                    try:
                        self.metrics.update(outputs, data, train_type="test")
                    except Exception as exception:
                        self.report_failure(
                            data=data,
                            outputs=outputs,
                            batch=ii,
                            epoch=epoch,
                            step='test metric evaluation',
                            exception=exception
                        )
                if (progress_bar == 'all' or progress_bar == 'test'):
                    test_loop.set_description(f"Testing: Batch [{ii+1}/{self.meta['loader'].num_test_batches}]")
                    test_loop.set_postfix_str(f"loss={loss.item():.2e}")
            try:
                self.callbacks.evaluate_epoch(train_type='test')
            except Exception as exception:
                self.report_failure(
                    data=data,
                    outputs=outputs,
                    batch=ii,
                    epoch=-1,
                    step='test callback epoch evaluation',
                    exception=exception
                )
        try:
            self.callbacks.evaluate_testing()
        except Exception as exception:
            self.report_failure(
                data=data,
                outputs=outputs,
                batch=ii,
                epoch=-1,
                step='test callback evaluation',
                exception=exception
            )
        try:
            self.model.save_model(flag='trained')
        except Exception as exception:
            self.report_failure(
                data=None,
                outputs=None,
                batch=ii,
                epoch=-1,
                step='save model',
                exception=exception
            )
        if save_predictions:
            self.logger.info("running inference to save predictions.")
            return self.inference(
                dataset_type='all',
                outputs=[output for output in self.shapes["output"].keys()],
                progress_bar=progress_bar,
                rewrite_bar=rewrite_bar,
                save_predictions=True,
            )

    def inference(
        self,
        dataset_type:   str = 'all',    # which dataset to use for inference
        layers:         list = [],      # which forward views to save
        outputs:        list = [],      # which outputs to save
        save_predictions: bool = True,  # wether to save the predictions
        progress_bar:   bool = True,    # progress bar from tqdm
        rewrite_bar:    bool = True,    # wether to leave the bars after each epoch
        skip_metrics:   bool = False,   # wether to skip metrics except for testing sets
    ):
        """
        Here we just do inference on a particular part
        of the dataset_loader, either 'train', 'validation',
        'test' or 'all'.
        """
        # check that everything is on the same device
        if (self.model.device != self.device):
            self.logger.error(f"device: '{self.device}' and model device: '{self.model.device}' are different!")
        if (self.criterion.device != self.device):
            self.logger.error(f"device: '{self.device}' and model device: '{self.criterion.device}' are different!")

        # determine loader
        if dataset_type == 'train':
            inference_loader = self.meta['loader'].train_loader
            num_batches = self.meta['loader'].num_training_batches
            inference_indices = self.meta['loader'].train_indices
        elif dataset_type == 'validation':
            inference_loader = self.meta['loader'].validation_loader
            num_batches = self.meta['loader'].num_validation_batches
            inference_indices = self.meta['loader'].validation_indices
        elif dataset_type == 'test':
            inference_loader = self.meta['loader'].test_loader
            num_batches = self.meta['loader'].num_test_batches
            inference_indices = self.meta['loader'].test_indices
        else:
            inference_loader = self.meta['loader'].all_loader
            num_batches = self.meta['loader'].num_all_batches
            inference_indices = self.meta['loader'].all_indices

        """
        Set up progress bar.
        """
        if (progress_bar is True):
            inference_loop = tqdm(
                enumerate(inference_loader, 0),
                total=len(list(inference_indices)),
                leave=rewrite_bar,
                position=0,
                colour='magenta'
            )
        else:
            inference_loop = enumerate(inference_loader, 0)

        # set up array for predictions
        predictions = {
            layer: []
            for layer in layers
        }
        for output in outputs:
            predictions[output] = []

        self.logger.info(f"running inference on dataset '{self.meta['dataset'].name}'.")
        # make sure to set model to eval() during validation!
        self.model.eval()
        with torch.no_grad():
            if self.metrics is not None:
                self.metrics.reset_batch()
            for ii, data in inference_loop:
                # get the network output
                model_output = self.model(data)
                for jj, key in enumerate(model_output.keys()):
                    if key in predictions.keys():
                        predictions[key].append([model_output[key].cpu().numpy()])
                for jj, key in enumerate(layers):
                    if key in predictions.keys():
                        predictions[key].append([self.model.forward_views[key].cpu().numpy()])
                # compute loss
                if self.criterion is not None:
                    loss = self.criterion.loss(model_output, data)

                # update metrics
                if not skip_metrics:
                    if self.metrics is not None:
                        self.metrics.update(model_output, data, train_type="inference")

                # update progress bar
                if (progress_bar is True):
                    inference_loop.set_description(f"Inference: Batch [{ii+1}/{num_batches}]")
                    inference_loop.set_postfix_str(f"loss={loss.item():.2e}")
        for key in predictions.keys():
            predictions[key] = np.vstack(np.array(predictions[key], dtype=object))
        # save predictions if wanted
        if save_predictions:
            self.meta['dataset'].append_dataset_files(
                self.model.name + "_predictions",
                predictions,
                np.array(inference_indices, dtype=object)
            )
        self.callbacks.evaluate_inference()
        self.logger.info("returning predictions.")
        return predictions
