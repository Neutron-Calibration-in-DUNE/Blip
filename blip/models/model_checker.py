"""
Container for models
"""
from torch_geometric.data.batch import Batch

from blip.utils.logger import Logger
from blip.utils.utils import get_method_arguments

class ModelChecker:
    """
    """
    def __init__(self,
        name:   str
    ):
        self.name = name
        self.logger = Logger(self.name, output="both", file_mode='w')
        # set to whatever the last call of set_device was.
        self.shapes = {
            "input": {},
            "output": {}
        }
        self.device = 'None'
    
    def run_consistency_check(self,
        dataset_loader,
        model,
        criterion,
        metrics
    ):
        """
        This function performs various checks on the dataset, dataset_loader,
        model, criterion, metrics and callbacks to make sure that things are
        configured correctly and that the shapes of various tensors are also
        set correctly.
        The information we need to grab:
            dataset:    feature_shape, class_shape
            model:      input_shape, output_shape
            dataset_loader: num_***_batches, 
        """
        # check dataset from dataset_loader
        try:
            data = next(iter(dataset_loader.inference_loader))
        except Exception as e:
            self.logger.error(f"problem indexing elements of dataset {dataset_loader.meta['dataset']}: {e}")
        if isinstance(data, Batch):
            self.shapes["input"]["features"] = data['x'][0].shape
            self.shapes["input"]["positions"] = data['pos'][0].shape
            self.shapes["input"]["category"] = data.category[0].shape

        model.eval()
        try:
            output = model(data)
        except Exception as e:
            self.logger.error(
                f"model '{model}' forward function incompatible with data from dataset_loader!"
                + f"  Perhaps you forgot 'x = x.to(self.device)'?: {e}"
            )
        for classes in output.keys():
            if classes == 'augmentations':
                continue
            self.shapes["output"][classes] = output[classes][0].shape

        # confirm shapes and behavior with criterion
        for name, loss in criterion.losses.items():
            try:
                loss_value = loss.loss(output, data)
            except Exception as e:
                self.logger.error(
                    f"loss function '{loss}' evaluation failed with inputs:"
                    +f"\noutput={output}\ndata={data}\n{e}"
                )
        
        # confirm shapes and behavior with metrics
        """
        There are two classes of metrics, 
            generic:    saves a single output tensor and a single target tensor
            tuple:      saves a tuple of output tensors and all dataloader tensors
        If each type is present in our metrics list, then we should check that 
        tensor operations defined within the metrics are compatible with the 
        dataloader and the model.
        """
        # # create empty tensors
        # if metrics != None:
        #     # first generic shapes are tested
        #     if self.num_output_elements == 1:
        #         try:
        #             test_output = torch.empty(
        #                 size=(0,*self.input_shape), 
        #                 dtype=torch.float, device=self.device
        #             )
        #         except Exception as e:
        #             self.logger.error(f"problem creating tensor with output shape '{output.shape}'.")
        #     else:
        #         for key, value in self.input_shape.items():
        #             try:
        #                 test_output = torch.empty(
        #                     size=(0,*value), 
        #                     dtype=torch.float, device=self.device
        #                 )
        #             except Exception as e:
        #                 self.logger.error(f"problem creating tensor with output shape '{value}'.")
        #     try:
        #         test_target = torch.empty(
        #             size=(0,*target[0].shape), 
        #             dtype=torch.float, device=self.device
        #         )
        #     except Exception as e:
        #         self.logger.error(f"problem creating tensor with target shape '{target.shape}'.")
        #     if isinstance(metrics, MetricHandler):
        #         metrics.set_shapes(self.input_shape)
        #     metrics.reset()
        # confirm shapes and behavior with callbacks
        criterion.reset_batch()
        self.logger.info("passed consistency check.")
        return self.shapes