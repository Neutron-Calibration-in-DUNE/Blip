"""
Class for a generic model trainer.
"""
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast
from torch.distributed import ReduceOp
import numpy as np
import os
import time
from mpi4py import MPI
from tqdm import tqdm

from blip.utils.logger import BlipError
from blip.utils.utils import profiler
from blip.utils import comm


class BlipTrainer:
    """
    This class is an attempt to reduce code rewriting by putting together
    a set of functions that do everything that we could need with
    respect to training.  There are a few objects which must be passed
    to the trainer, which include:
        (a) model     - an object which inherits from nn.Module
        (b) criterion - an object which has a defined function called "loss"
        (c) optimizer - some choice of optimizer, e.g. Adam
        (d) metrics   - (optional) an object which has certain defined functions
    """
    def __init__(
        self,
        config: dict = {},
        meta: dict = {},
    ):
        self.config = config
        self.meta = meta

        if "device" in self.meta:
            self.device = self.meta["device"]

        self.event_errors = []

        self.parse_config()
        self.parse_meta()

    @profiler
    def parse_config(self):
        if "num_epochs" not in self.meta:
            raise BlipError('"epochs" not specified in config!')
        else:
            self.num_epochs = self.meta["num_epochs"]
        if "checkpoint" not in self.config.keys():
            self.checkpoint = 10
        else:
            self.checkpoint = self.config["checkpoint"]
        if "progress_bar" not in self.config.keys():
            self.progress_bar = 'all'
        else:
            self.progress_bar = self.config["progress_bar"]
        if "rewrite_bar" not in self.config.keys():
            self.rewrite_bar = False
        else:
            self.rewrite_bar = self.config["rewrite_bar"]
        if "save_predictions" not in self.config.keys():
            self.save_predictions = False
        else:
            self.save_predictions = self.config["save_predictions"]

    @profiler
    def cleanup():
        dist.destroy_process_group()

    @profiler
    def parse_meta(self):
        self.parse_model()
        self.parse_directories()
        self.parse_criterion()
        self.parse_optimizer()
        self.parse_metrics()
        self.parse_consistency_check()

    @profiler
    def parse_model(self):
        if "model" not in self.meta:
            raise BlipError('no model specified in meta!')
        self.model = self.meta['model'].model

    @profiler
    def parse_directories(self):
        # define directories
        pass
        # if "timing_dir" not in self.config:
        #     self.config["timing_dir"] = f'{self.meta["local_scratch"]}/plots/{self.model.name}/timing/'
        # self.meta['timing_dir'] = self.config['timing_dir']
        # if not os.path.isdir(self.config['timing_dir']):
        #     self.logger.info(f"creating timing directory {self.config['timing_dir']}")
        #     os.makedirs(self.config['timing_dir'])
        # if "memory_dir" not in self.config:
        #     self.config["memory_dir"] = f'{self.meta["local_scratch"]}/plots/{self.model.name}/timing/'
        # self.meta['memory_dir'] = self.config['memory_dir']
        # if not os.path.isdir(self.config['memory_dir']):
        #     self.logger.info(f"creating timing directory {self.config['memory_dir']}")
        #     os.makedirs(self.config['memory_dir'])

    @profiler
    def parse_criterion(self):
        if "criterion" not in self.meta:
            raise BlipError('no criterion specified in meta!')
        self.criterion = self.meta['criterion']
        self.criterion.set_device(self.device)

    @profiler
    def parse_optimizer(self):
        if "optimizer" not in self.meta:
            raise BlipError('no optimizer specified in meta!')
        self.optimizer = self.meta['optimizer']

    @profiler
    def parse_metrics(self):
        if "metrics" not in self.meta:
            raise BlipError('no metrics specified in meta!')
        self.metrics = self.meta['metrics']
        self.metrics.set_device(self.device)

    @profiler
    def parse_consistency_check(self):
        # run consistency check
        pass
        # self.logger.info("running consistency check...")
        # self.shapes = self.model_checker.run_consistency_check(
        #     dataset_loader=self.meta['loader'],
        #     model=self.model,
        #     criterion=self.criterion,
        #     metrics=self.metrics
        # )

    @profiler
    def save_checkpoint(
        self,
        epoch: int = 99999
    ):
        if not os.path.exists(f"{self.meta['local_scratch']}/.checkpoints/"):
            os.makedirs(f"{self.meta['local_scratch']}/.checkpoints/")
        torch.save(
            self.model.state_dict(),
            f"{self.meta['local_scratch']}/.checkpoints/checkpoint_{epoch}.ckpt"
        )

    @profiler
    def report_failure(
        self,
        data,
        batch,
        epoch,
        step,
        exception
    ):
        self.save_checkpoint(epoch=f'error_{epoch}')
        self.event_errors.append(
            '\n****RECORDED AN ERROR IN COMPUTATION****\n' +
            f'data:     {data}\n\n' +
            f'epoch:    {epoch}\n' +
            f'batch:    {batch}\n' +
            f'step:     {step}\n' +
            f'exception:    {exception}\n' +
            '****RECORDED AN ERROR IN COMPUTATION****\n'
        )

    @profiler
    def train(
        self,
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
        iterations = 0

        """Iterate over epochs"""
        for epoch in range(self.meta['num_epochs']):
            """Synchronize devices"""
            try:
                torch.cuda.synchronize()
            except Exception:
                raise BlipError('error occurred attempting to synchronize cuda')

            """Set sampler epoch"""
            try:
                if self.meta["distributed"]:
                    self.meta['loader'].train_sampler.set_epoch(epoch)
            except Exception:
                raise BlipError('error occurred setting the epoch for distributed training')

            """Set the start time for tensorboard"""
            try:
                start = time.time()
            except Exception:
                raise BlipError('error occurred getting start time')

            """Set the step_count for tensorboard"""
            step_count = 0

            """
            Setup the progress bar for the training loop.
            """
            try:
                if self.progress_bar in ['all', 'train']:
                    training_loop = tqdm(
                        enumerate(self.meta['loader'].train_loader, 0),
                        total=self.meta['num_iterations'],
                        leave=self.rewrite_bar,
                        position=0,
                        colour='green'
                    )
                else:
                    training_loop = enumerate(self.meta['loader'].train_loader, 0)
            except Exception:
                raise BlipError('error occurred setting up training loop')

            """Set model to train"""
            try:
                self.model.train()
            except Exception:
                raise BlipError('error occurred setting model to train')

            for ii, data in training_loop:
                """Set scheduler parameters"""
                try:
                    if self.meta["world_rank"] == 0:
                        if (epoch == 3 and ii == 0):
                            torch.cuda.profiler.start()
                        if (epoch == 3 and ii == self.meta['num_iterations'] - 1):
                            torch.cuda.profiler.stop()
                except Exception:
                    raise BlipError('error occurred getting cuda profiler information')

                """Break if iterations maxed out"""
                if iterations >= self.meta['num_iterations']:
                    break

                """
                There are choices here, either one can do:
                    model.zero_grad() or
                    optimizer.zero_grad() or
                    for param in model.parameters():        <== optimal choice
                        param.grad = None
                """
                try:
                    for param in self.model.parameters():
                        param.grad = None
                except Exception:
                    raise BlipError('error occurred resetting model parameters')

                """Push to NVTX"""
                try:
                    torch.cuda.nvtx.range_push(f"step {ii}")
                except Exception:
                    raise BlipError('error occurred pushing step number to nvtx')

                iterations += 1

                """
                The forward call takes in the entire data
                stream, which could have multiple inputs needed.
                It's up to the model to determine what to do with it.
                The forward call of the model could send out
                multiple output tensors, depending on the application
                (such as in an AE where the latent space values are
                important). It's up to the loss function to know what to expect.
                """
                try:
                    torch.cuda.nvtx.range_push("forward")
                except Exception:
                    raise BlipError('error occurred pushing forward call to nvtx')

                with autocast(
                    enabled=self.meta["amp_enabled"],
                    dtype=self.meta["amp_dtype"]
                ):
                    try:
                        data['outputs'] = self.model(data)
                    except Exception as exception:
                        self.report_failure(
                            data=data,
                            epoch=epoch,
                            batch=ii,
                            step='model forward',
                            exception=exception
                        )

                    """Compute loss"""
                    try:
                        loss = self.criterion.loss(data, iteration=iterations)
                    except Exception as exception:
                        self.report_failure(
                            data=data,
                            epoch=epoch,
                            batch=ii,
                            step='loss evaluation',
                            exception=exception
                        )

                try:
                    torch.cuda.nvtx.range_pop()
                except Exception:
                    raise BlipError('error occurred trying to range pop nvtx')

                """Backprop step"""
                try:
                    if self.meta['amp_dtype'] == torch.float16:
                        self.meta["scaler"].scale(loss).backward()
                        torch.cuda.nvtx.range_push("optimizer")
                        self.meta["scaler"].step(self.meta["optimizer"])
                        torch.cuda.nvtx.range_pop()
                    else:
                        loss.backward()
                        torch.cuda.nvtx.range_push("optimizer")
                        self.meta["optimizer"].step()
                        torch.cuda.nvtx.range_pop()
                except Exception as exception:
                    self.report_failure(
                        data=data,
                        epoch=epoch,
                        batch=ii,
                        step='backprop',
                        exception=exception
                    )

                """Reduce loss"""
                try:
                    if self.meta["distributed"]:
                        torch.distributed.all_reduce(
                            loss,
                            op=ReduceOp.AVG,
                            group=comm.get_group("data")
                        )
                except Exception as exception:
                    self.report_failure(
                        data=data,
                        epoch=epoch,
                        batch=ii,
                        step='loss reduction',
                        exception=exception
                    )

                try:
                    torch.cuda.nvtx.range_pop()
                except Exception:
                    raise BlipError('error occurred trying to range pop nvtx')

                """Update scheduler"""
                try:
                    self.meta["scheduler"].step()
                except Exception as exception:
                    self.report_failure(
                        data=data,
                        epoch=epoch,
                        batch=ii,
                        step='update scheduler',
                        exception=exception
                    )

                """Update progress bar"""
                try:
                    if self.progress_bar in ['all', 'train']:
                        training_loop.set_description(
                            f"Training: Epoch [{epoch+1}/{self.meta['num_epochs']}]"
                        )
                        training_loop.set_postfix_str(f"loss={loss.item():.2e}")
                except Exception:
                    raise BlipError('error occurred setting training loop annotation')

                """Update metrics if last epoch"""
                if (epoch == self.meta['num_epochs'] - 1):
                    """Evaluate metrics"""
                    if self.metrics is not None:
                        self.metrics.update(data)

                step_count += 1

            """Synchronize CUDA"""
            try:
                torch.cuda.synchronize()
            except Exception:
                raise BlipError('error occurred attempting to synchronize cuda')

            try:
                end = time.time()
            except Exception:
                raise BlipError('error occurred getting end time for tensorboard')

            try:
                if self.meta['world_rank'] == 0:
                    iters_per_sec = step_count / (end - start)
                    samples_per_sec = self.meta["global_batch_size"] * iters_per_sec
                    self.criterion.report_tensorboard(iterations, train_type='train')
                    self.optimizer.report_tensorboard(iterations, train_type='train')
                    self.meta['tensorboard'].add_scalar('Avg iters per sec (train)', iters_per_sec, iterations)
                    self.meta['tensorboard'].add_scalar('Avg samples per sec (train)', samples_per_sec, iterations)

                    """Update metrics if last epoch"""
                    if (epoch == self.meta['num_epochs'] - 1):
                        if self.metrics is not None:
                            metrics = self.metrics.compute()
                            self.metrics.report_tensorboard(iterations, train_type='train')
            except Exception:
                raise BlipError('error occurred sending information to tensorboard')

            """
            Validation stage.
            """

            """Set the step_count for tensorboard"""
            step_count = 0

            """
            Setup the progress bar for the validation loop.
            """
            try:
                if self.progress_bar in ['all', 'validation']:
                    validation_loop = tqdm(
                        enumerate(self.meta['loader'].validation_loader, 0),
                        total=len(self.meta['loader'].validation_loader),
                        leave=self.rewrite_bar,
                        position=1,
                        colour='blue'
                    )
                else:
                    validation_loop = enumerate(self.meta['loader'].validation_loader, 0)
            except Exception:
                raise BlipError('error occurred setting up validation loop')

            """Set model to eval"""
            try:
                self.model.eval()
            except Exception:
                raise BlipError('error occurred setting model to eval')

            """Set the start time for tensorboard"""
            try:
                start = time.time()
            except Exception:
                raise BlipError('error occurred getting start time')

            with torch.no_grad():
                """
                Setup timing information for epoch.
                """
                for ii, data in validation_loop:

                    with autocast(
                        enabled=self.meta["amp_enabled"],
                        dtype=self.meta["amp_dtype"]
                    ):
                        try:
                            data['outputs'] = self.model(data)
                        except Exception as exception:
                            self.report_failure(
                                data=data,
                                epoch=epoch,
                                batch=ii,
                                step='model forward',
                                exception=exception
                            )

                        """Compute loss"""
                        try:
                            loss = self.criterion.loss(data, iteration=iterations)
                        except Exception as exception:
                            self.report_failure(
                                data=data,
                                epoch=epoch,
                                batch=ii,
                                step='loss evaluation',
                                exception=exception
                            )

                        """Update metrics if last epoch"""
                        if (epoch == self.meta['num_epochs'] - 1):
                            """Evaluate metrics"""
                            if self.metrics is not None:
                                self.metrics.update(data)

                    step_count += 1

                    """Update progress bar"""
                    try:
                        if self.progress_bar in ['all', 'validation']:
                            validation_loop.set_description(
                                f"Validation: Epoch [{epoch+1}/{self.meta['num_epochs']}]"
                            )
                            validation_loop.set_postfix_str(f"loss={loss.item():.2e}")
                    except Exception:
                        raise BlipError('error occurred setting validation loop annotation')

                try:
                    end = time.time()
                except Exception:
                    raise BlipError('error occurred getting end time for tensorboard')

                try:
                    if self.meta['world_rank'] == 0:
                        iters_per_sec = step_count / (end - start)
                        samples_per_sec = self.meta["global_batch_size"] * iters_per_sec
                        self.criterion.report_tensorboard(iterations, train_type='validation')
                        self.optimizer.report_tensorboard(iterations, train_type='validation')
                        self.meta['tensorboard'].add_scalar('Avg iters per sec (validation)', iters_per_sec, iterations)
                        self.meta['tensorboard'].add_scalar('Avg samples per sec (validation)', samples_per_sec, iterations)

                        """Update metrics if last epoch"""
                        if (epoch == self.meta['num_epochs'] - 1):
                            if self.metrics is not None:
                                metrics = self.metrics.compute()
                                self.metrics.report_tensorboard(iterations, train_type='validation')
                except Exception:
                    raise BlipError('error occurred sending information to tensorboard')

            """Save weights if at checkpoint step"""
            try:
                if epoch % self.checkpoint == 0:
                    self.save_checkpoint(epoch)
            except Exception:
                raise BlipError('error occurred saving checkpoint')

            """Free up gpu resources"""
            try:
                torch.cuda.empty_cache()
            except Exception:
                raise BlipError('error occurred emptying cache')

        """
        Testing stage.
        Setup the progress bar for the testing loop.
        We do not have timing information for the test
        loop stage, since it is generally quick
        and doesn't need to be optimized for any reason.
        """
        try:
            if self.progress_bar in ['all', 'test']:
                test_loop = tqdm(
                    enumerate(self.meta['loader'].test_loader, 0),
                    total=len(self.meta['loader'].test_loader),
                    leave=self.rewrite_bar,
                    position=0,
                    colour='red'
                )
            else:
                test_loop = enumerate(self.meta['loader'].test_loader, 0)
        except Exception:
            raise BlipError('error occurred setting up testing loop')

        """Set model to eval"""
        try:
            self.model.eval()
        except Exception:
            raise BlipError('error occurred setting model to eval')

        with torch.no_grad():
            for ii, data in test_loop:

                with autocast(
                    enabled=self.meta["amp_enabled"],
                    dtype=self.meta["amp_dtype"]
                ):
                    try:
                        data['outputs'] = self.model(data)
                    except Exception as exception:
                        self.report_failure(
                            data=data,
                            epoch=epoch,
                            batch=ii,
                            step='model forward',
                            exception=exception
                        )

                    """Compute loss"""
                    try:
                        loss = self.criterion.loss(data, iteration=iterations)
                    except Exception as exception:
                        self.report_failure(
                            data=data,
                            epoch=epoch,
                            batch=ii,
                            step='loss evaluation',
                            exception=exception
                        )

                """Evaluate metrics"""
                if self.metrics is not None:
                    self.metrics.update(data)

                """Update progress bar"""
                try:
                    if self.progress_bar in ['all', 'test']:
                        test_loop.set_description(
                            f"Testing: Batch [{ii+1}/{self.meta['loader'].num_test_batches}]"
                        )
                        test_loop.set_postfix_str(f"loss={loss.item():.2e}")
                except Exception:
                    raise BlipError('error occurred setting test loop annotation')

            if self.metrics is not None:
                metrics = self.metrics.compute()

            try:
                if self.meta['world_rank'] == 0:
                    iters_per_sec = step_count / (end - start)
                    samples_per_sec = self.meta["global_batch_size"] * iters_per_sec
                    self.criterion.report_tensorboard(iterations, train_type='test')
                    self.optimizer.report_tensorboard(iterations, train_type='test')
                    self.meta['tensorboard'].add_scalar('Avg iters per sec (test)', iters_per_sec, iterations)
                    self.meta['tensorboard'].add_scalar('Avg samples per sec (test)', samples_per_sec, iterations)

                    """Update metrics if last epoch"""
                    if self.metrics is not None:
                        metrics = self.metrics.compute()
                        self.metrics.report_tensorboard(iterations, train_type='test')
            except Exception:
                raise BlipError('error occurred sending information to tensorboard')

        """Save the final model"""
        try:
            self.model.save_model(flag='trained')
        except Exception:
            raise BlipError('error saving final model')

        """Get predictions if wanted"""
        if self.save_predictions:
            return self.inference(
                dataset_type='all',
            )

    def inference(
        self,
        dataset_type:   str = 'all',    # which dataset to use for inference
        layers:         list = [],      # which forward views to save
    ):
        """
        Here we just do inference on a particular part
        of the dataset_loader, either 'train', 'validation',
        'test' or 'all'.
        """

        # determine loader
        try:
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
        except Exception:
            raise BlipError('error occurred setting up inference loader')

        """
        Set up progress bar.
        """
        try:
            if (self.progress_bar is True):
                inference_loop = tqdm(
                    enumerate(inference_loader, 0),
                    total=len(list(inference_indices)),
                    leave=self.rewrite_bar,
                    position=0,
                    colour='magenta'
                )
            else:
                inference_loop = enumerate(inference_loader, 0)
        except Exception as exception:
            raise BlipError('error occurred setting up inference loop')

        """Set up array for predictions"""
        predictions = {
            layer: []
            for layer in layers
        }
        for output in outputs:
            predictions[output] = []

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
                if not self.skip_metrics:
                    if self.metrics is not None:
                        self.metrics.update(model_output, data, train_type="inference")

                # update progress bar
                if (self.progress_bar is True):
                    inference_loop.set_description(f"Inference: Batch [{ii+1}/{num_batches}]")
                    inference_loop.set_postfix_str(f"loss={loss.item():.2e}")
        for key in predictions.keys():
            predictions[key] = np.vstack(np.array(predictions[key], dtype=object))
        # save predictions if wanted
        if self.save_predictions:
            self.meta['dataset'].save_predictions(
                self.model.name + "_predictions",
                predictions,
                np.array(inference_indices, dtype=object)
            )
        return predictions
