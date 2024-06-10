"""
Class for a generic model trainer.
"""
import ray
from ray import train
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast
from torch.distributed import ReduceOp
import os
import time
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
        if "mode" not in self.meta:
            self.mode = 'train'
        else:
            self.mode = self.config['mode']
        if self.mode not in ['train', 'inference']:
            raise BlipError(f"trainer mode set to {self.mode}, but can only be 'train' or 'inference'")
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

    def save_checkpoint(
        self,
        epoch: int = 99999
    ):
        if self.meta["local_rank"] == 0:
            try:
                if self.meta["distributed"]:
                    self.model.module.save_model(
                        epoch=epoch,
                        flag=f'checkpoint_{epoch}'
                    )
                else:
                    self.model.save_model(
                        epoch=epoch,
                        flag=f'checkpoint_{epoch}'
                    )
            except Exception:
                raise BlipError(f'error saving checkpoint {epoch}')

    def report_failure(
        self,
        data,
        batch,
        epoch,
        step,
        exception
    ):
        # self.save_checkpoint(epoch=epoch)
        self.event_errors.append(
            '\n****RECORDED AN ERROR IN COMPUTATION****\n' +
            f'data:     {data}\n\n' +
            f'epoch:    {epoch}\n' +
            f'batch:    {batch}\n' +
            f'step:     {step}\n' +
            f'exception:    {exception}\n' +
            '****RECORDED AN ERROR IN COMPUTATION****\n'
        )

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
        """Save the initial model"""
        if self.meta["local_rank"] == 0:
            try:
                if self.meta['distributed']:
                    self.model.module.save_model(flag='initial')
                else:
                    self.model.save_model(flag='initial')
            except Exception:
                raise BlipError('error saving initial model')

        iterations = 0

        """Iterate over epochs"""
        for epoch in range(self.meta['num_epochs']):
            """Set tuning variables"""
            epoch_loss = 0

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
                        colour='green',
                        initial=iterations
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
                        if (epoch == self.meta['num_epochs'] - 1 and ii == 0):
                            torch.cuda.profiler.start()
                        if (epoch == self.meta['num_epochs'] - 1 and ii == self.meta['num_iterations'] - 1):
                            torch.cuda.profiler.stop()
                except Exception as exception:
                    self.report_failure(
                        data=data,
                        epoch=epoch,
                        batch=ii,
                        step='cuda profiler training',
                        exception=exception
                    )

                """Break if iterations maxed out"""
                if iterations >= self.meta['num_iterations']:
                    break

                """Push to NVTX"""
                try:
                    torch.cuda.nvtx.range_push(f"step {ii}")
                except Exception as exception:
                    self.report_failure(
                        data=data,
                        epoch=epoch,
                        batch=ii,
                        step='nvtx range push step training',
                        exception=exception
                    )

                """
                There are choices here, either one can do:
                    model.zero_grad() or
                    optimizer.zero_grad() or
                    for param in model.parameters():        <== optimal choice
                        param.grad = None
                """
                try:
                    torch.cuda.nvtx.range_push("zero grad")
                except Exception as exception:
                    self.report_failure(
                        data=data,
                        epoch=epoch,
                        batch=ii,
                        step='nvtx range push zero grad training',
                        exception=exception
                    )
                try:
                    for param in self.model.parameters():
                        param.grad = None
                except Exception as exception:
                    self.report_failure(
                        data=data,
                        epoch=epoch,
                        batch=ii,
                        step='optimizer zero grad training',
                        exception=exception
                    )
                try:
                    torch.cuda.nvtx.range_pop()
                except Exception as exception:
                    self.report_failure(
                        data=data,
                        epoch=epoch,
                        batch=ii,
                        step='nvtx range pop zero grad training',
                        exception=exception
                    )

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
                with autocast(
                    enabled=self.meta["amp_enabled"],
                    dtype=self.meta["amp_dtype"]
                ):
                    """Forward call"""
                    try:
                        torch.cuda.nvtx.range_push("forward")
                    except Exception as exception:
                        self.report_failure(
                            data=data,
                            epoch=epoch,
                            batch=ii,
                            step='nvtx range push forward training',
                            exception=exception
                        )
                    try:
                        data['outputs'] = self.model(data)
                    except Exception as exception:
                        self.report_failure(
                            data=data,
                            epoch=epoch,
                            batch=ii,
                            step='model forward training',
                            exception=exception
                        )
                    try:
                        torch.cuda.nvtx.range_pop()
                    except Exception as exception:
                        self.report_failure(
                            data=data,
                            epoch=epoch,
                            batch=ii,
                            step='nvtx range pop forward training',
                            exception=exception
                        )

                    """Compute loss"""
                    try:
                        torch.cuda.nvtx.range_push("loss")
                    except Exception as exception:
                        self.report_failure(
                            data=data,
                            epoch=epoch,
                            batch=ii,
                            step='nvtx range push loss training',
                            exception=exception
                        )
                    try:
                        loss = self.criterion.loss(data, iteration=iterations)
                    except Exception as exception:
                        self.report_failure(
                            data=data,
                            epoch=epoch,
                            batch=ii,
                            step='loss evaluation training',
                            exception=exception
                        )
                    try:
                        torch.cuda.nvtx.range_pop()
                    except Exception as exception:
                        self.report_failure(
                            data=data,
                            epoch=epoch,
                            batch=ii,
                            step='nvtx range pop loss training',
                            exception=exception
                        )

                """Backprop step"""
                try:
                    torch.cuda.nvtx.range_push("optimizer")
                except Exception as exception:
                    self.report_failure(
                        data=data,
                        epoch=epoch,
                        batch=ii,
                        step='nvtx range push optimizer training',
                        exception=exception
                    )
                try:
                    if self.meta['amp_dtype'] == torch.float16:
                        self.meta["scaler"].scale(loss).backward()
                        self.meta["scaler"].step(self.meta["optimizer"].optimizer)
                        self.meta["scaler"].update()
                    else:
                        loss.backward()
                        self.meta["optimizer"].step()
                except Exception as exception:
                    self.report_failure(
                        data=data,
                        epoch=epoch,
                        batch=ii,
                        step='backprop training',
                        exception=exception
                    )
                try:
                    torch.cuda.nvtx.range_pop()
                except Exception as exception:
                    self.report_failure(
                        data=data,
                        epoch=epoch,
                        batch=ii,
                        step='nvtx range pop optimizer training',
                        exception=exception
                    )

                """Reduce loss"""
                try:
                    torch.cuda.nvtx.range_push("reduce loss")
                except Exception as exception:
                    self.report_failure(
                        data=data,
                        epoch=epoch,
                        batch=ii,
                        step='nvtx range push reduce loss training',
                        exception=exception
                    )
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
                        step='loss reduction training',
                        exception=exception
                    )
                try:
                    torch.cuda.nvtx.range_pop()
                except Exception as exception:
                    self.report_failure(
                        data=data,
                        epoch=epoch,
                        batch=ii,
                        step='nvtx range pop reduce loss training',
                        exception=exception
                    )

                try:
                    torch.cuda.nvtx.range_pop()
                except Exception as exception:
                    self.report_failure(
                        data=data,
                        epoch=epoch,
                        batch=ii,
                        step='nvtx range pop step training',
                        exception=exception
                    )

                """Update scheduler"""
                try:
                    torch.cuda.nvtx.range_push("scheduler")
                except Exception as exception:
                    self.report_failure(
                        data=data,
                        epoch=epoch,
                        batch=ii,
                        step='nvtx range push scheduler training',
                        exception=exception
                    )
                try:
                    self.meta["scheduler"].step()
                except Exception as exception:
                    self.report_failure(
                        data=data,
                        epoch=epoch,
                        batch=ii,
                        step='update scheduler training',
                        exception=exception
                    )
                try:
                    torch.cuda.nvtx.range_pop()
                except Exception as exception:
                    self.report_failure(
                        data=data,
                        epoch=epoch,
                        batch=ii,
                        step='nvtx range pop scheduler training',
                        exception=exception
                    )

                """Update progress bar"""
                try:
                    if self.progress_bar in ['all', 'train']:
                        training_loop.set_description(
                            f"Training: Epoch [{epoch+1}/{self.meta['num_epochs']}] "
                            + f"[{ii+1}/{len(self.meta['loader'].train_loader)}]"
                        )
                        training_loop.set_postfix_str(f"loss={loss.item():.2e}")
                except Exception as exception:
                    self.report_failure(
                        data=data,
                        epoch=epoch,
                        batch=ii,
                        step='training progress bar update',
                        exception=exception
                    )

                """Update tuning variables"""
                if self.meta['world_rank'] == 0:
                    epoch_loss += loss.item() / len(training_loop)

                """Update metrics if last epoch"""
                if (epoch == self.meta['num_epochs'] - 1):
                    try:
                        torch.cuda.nvtx.range_push("metric update")
                    except Exception as exception:
                        self.report_failure(
                            data=data,
                            epoch=epoch,
                            batch=ii,
                            step='nvtx range push metric update training',
                            exception=exception
                        )
                    try:
                        """Evaluate metrics"""
                        if self.metrics is not None:
                            self.metrics.update(data)
                    except Exception as exception:
                        self.report_failure(
                            data=data,
                            epoch=epoch,
                            batch=ii,
                            step='update metrics training',
                            exception=exception
                        )
                    try:
                        torch.cuda.nvtx.range_pop()
                    except Exception as exception:
                        self.report_failure(
                            data=data,
                            epoch=epoch,
                            batch=ii,
                            step='nvtx range pop metric update training',
                            exception=exception
                        )

                step_count += 1

            """Update tuning variables"""
            if self.meta['world_rank'] == 0:
                train.report(
                    epoch=epoch,
                    epoch_loss=epoch_loss
                )

            """Synchronize CUDA"""
            try:
                torch.cuda.synchronize()
            except Exception:
                raise BlipError('error occurred attempting to synchronize cuda')

            try:
                end = time.time()
            except Exception:
                raise BlipError('error occurred getting end time for tensorboard')

            """If last epoch, report training loss"""
            if (epoch == self.meta['num_epochs'] - 1):
                if self.meta['world_rank'] == 0:
                    train.report(
                        train_loss=epoch_loss,
                        training_time=(end - start)
                    )

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
                            self.metrics.report_tensorboard(iterations, train_type='train')
            except Exception:
                raise BlipError('error occurred sending information to tensorboard')

            """
            Validation stage.
            """
            """Update metrics if last epoch"""
            if (epoch == self.meta['num_epochs'] - 1):
                try:
                    if self.metrics is not None:
                        self.metrics.reset_batch()
                except Exception as exception:
                    self.report_failure(
                        data=None,
                        epoch=epoch,
                        batch=-1,
                        step='reset metrics post training',
                        exception=exception
                    )

            """Set up epoch loss for raytune"""
            epoch_loss = 0

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
                                step='model forward validation',
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
                                step='loss evaluation validation',
                                exception=exception
                            )

                        """Update tuning variables"""
                        if self.meta['world_rank'] == 0:
                            epoch_loss += loss.item() / len(validation_loop)

                        """Update metrics if last epoch"""
                        if (epoch == self.meta['num_epochs'] - 1):
                            try:
                                """Evaluate metrics"""
                                if self.metrics is not None:
                                    self.metrics.update(data)
                            except Exception as exception:
                                self.report_failure(
                                    data=data,
                                    epoch=epoch,
                                    batch=ii,
                                    step='update metrics validation',
                                    exception=exception
                                )

                    step_count += 1

                    """Update progress bar"""
                    try:
                        if self.progress_bar in ['all', 'validation']:
                            validation_loop.set_description(
                                f"Validation: Epoch [{epoch+1}/{self.meta['num_epochs']}]"
                            )
                            validation_loop.set_postfix_str(f"loss={loss.item():.2e}")
                    except Exception as exception:
                        self.report_failure(
                            data=data,
                            epoch=epoch,
                            batch=ii,
                            step='validation progress bar update',
                            exception=exception
                        )

                try:
                    end = time.time()
                except Exception:
                    raise BlipError('error occurred getting end time for tensorboard')

                """If last epoch, report training loss"""
                if (epoch == self.meta['num_epochs'] - 1):
                    if self.meta['world_rank'] == 0:
                        train.report(
                            val_loss=epoch_loss
                        )

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
            if self.metrics is not None:
                self.metrics.reset_batch()
        except Exception as exception:
            self.report_failure(
                data=None,
                epoch=epoch,
                batch=-1,
                step='reset metrics validation',
                exception=exception
            )
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

        """Set the start time for tensorboard"""
        try:
            start = time.time()
        except Exception:
            raise BlipError('error occurred getting start time')

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
                            step='model forward test',
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
                            step='loss evaluation test',
                            exception=exception
                        )

                """Evaluate metrics"""
                try:
                    if self.metrics is not None:
                        self.metrics.update(data)
                except Exception as exception:
                    self.report_failure(
                        data=data,
                        epoch=epoch,
                        batch=ii,
                        step='update metrics test',
                        exception=exception
                    )

                """Update progress bar"""
                try:
                    if self.progress_bar in ['all', 'test']:
                        test_loop.set_description(
                            f"Testing: Batch [{ii+1}/{self.meta['loader'].num_test_batches}]"
                        )
                        test_loop.set_postfix_str(f"loss={loss.item():.2e}")
                except Exception as exception:
                    self.report_failure(
                        data=data,
                        epoch=epoch,
                        batch=ii,
                        step='test progress bar update',
                        exception=exception
                    )
        try:
            end = time.time()
        except Exception:
            raise BlipError('error occurred getting end time for tensorboard')

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
                    self.metrics.report_tensorboard(iterations, train_type='test')
        except Exception:
            raise BlipError('error occurred sending information to tensorboard')

        """Save the final model"""
        if self.meta["local_rank"] == 0:
            try:
                if self.meta['distributed']:
                    self.model.module.save_model(flag='trained')
                else:
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
        dataset_type: str = 'all',
        layers: list = [],
    ):
        """
        Here we just do inference on a particular part
        of the dataset_loader, either 'train', 'validation',
        'test' or 'all'.
        """
        try:
            if dataset_type == 'train':
                inference_loader = self.meta['loader'].train_loader
                inference_indices = self.meta['loader'].train_indices
            elif dataset_type == 'validation':
                inference_loader = self.meta['loader'].validation_loader
                inference_indices = self.meta['loader'].validation_indices
            elif dataset_type == 'test':
                inference_loader = self.meta['loader'].test_loader
                inference_indices = self.meta['loader'].test_indices
            else:
                inference_loader = self.meta['loader'].all_loader
                inference_indices = self.meta['loader'].all_indices
        except Exception:
            raise BlipError('error occurred setting up inference loader')

        """
        Set up progress bar.
        """
        try:
            if self.progress_bar in ['all', 'inference']:
                inference_loop = tqdm(
                    enumerate(inference_loader, 0),
                    total=len(list(inference_indices)),
                    leave=self.rewrite_bar,
                    position=0,
                    colour='magenta'
                )
            else:
                inference_loop = enumerate(inference_loader, 0)
        except Exception:
            raise BlipError('error occurred setting up inference loop')

        """Make sure to set model to eval() during validation!"""
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
            """Reset metrics"""
            try:
                if self.metrics is not None:
                    self.metrics.reset_batch()
            except Exception as exception:
                self.report_failure(
                    data=None,
                    epoch=-1,
                    batch=-1,
                    step='reset metrics inference',
                    exception=exception
                )

            iterations = 0

            for ii, data in inference_loop:
                """Get network output"""
                try:
                    data['outputs'] = self.model(data)
                except Exception as exception:
                    self.report_failure(
                        data=data,
                        epoch=-1,
                        batch=ii,
                        step='model forward inference',
                        exception=exception
                    )
                try:
                    for jj, key in enumerate(layers):
                        data[key] = self.model.forward_views[key].cpu().numpy()
                except Exception as exception:
                    self.report_failure(
                        data=data,
                        epoch=-1,
                        batch=ii,
                        step='layer inference',
                        exception=exception
                    )

                """Compute the loss"""
                try:
                    if self.criterion is not None:
                        loss = self.criterion.loss(data, iteration=iterations)
                except Exception as exception:
                    self.report_failure(
                        data=data,
                        epoch=-1,
                        batch=ii,
                        step='loss evaluation inference',
                        exception=exception
                    )

                """Update metrics"""
                try:
                    if self.metrics is not None:
                        self.metrics.update(data)
                except Exception as exception:
                    self.report_failure(
                        data=data,
                        epoch=-1,
                        batch=ii,
                        step='update metrics inference',
                        exception=exception
                    )

                """Pass predictions to dataset for saving"""
                try:
                    self.meta['dataset'].save_predictions(data)
                except Exception as exception:
                    self.report_failure(
                        data=data,
                        epoch=-1,
                        batch=ii,
                        step='saving predictions inference',
                        exception=exception
                    )

                """Update progress bar"""
                try:
                    if self.progress_bar in ['all', 'inference']:
                        inference_loop.set_description("Inference")
                        if self.criterion is not None:
                            inference_loop.set_postfix_str(f"loss={loss.item():.2e}")
                except Exception as exception:
                    self.report_failure(
                        data=data,
                        epoch=-1,
                        batch=ii,
                        step='inference progress bar update',
                        exception=exception
                    )
                iterations += 1

        try:
            end = time.time()
        except Exception:
            raise BlipError('error occurred getting end time for tensorboard')

        try:
            if self.meta['world_rank'] == 0:
                iters_per_sec = iterations / (end - start)
                samples_per_sec = self.meta["global_batch_size"] * iters_per_sec
                self.criterion.report_tensorboard(iterations, train_type='inference')
                self.optimizer.report_tensorboard(iterations, train_type='inference')
                self.meta['tensorboard'].add_scalar('Avg iters per sec (inference)', iters_per_sec, iterations)
                self.meta['tensorboard'].add_scalar('Avg samples per sec (inference)', samples_per_sec, iterations)

                """Update metrics if last epoch"""
                if self.metrics is not None:
                    self.metrics.report_tensorboard(iterations, train_type='inference')
        except Exception:
            raise BlipError('error occurred sending information to tensorboard')
