"""
File for iterating over hit proposal architecture
"""
import numpy as np
import torch
import torch.nn as nn
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF
from collections import OrderedDict

import torch
import MinkowskiEngine as ME
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score
from matplotlib import pyplot as plt

from blip.models.common import Identity
from blip.models.common import Identity, sparse_activations


def get_activation(
    activation: str,
):
    if activation in sparse_activations.keys():
        return sparse_activations[activation]



class DoubleConv(ME.MinkowskiNetwork):
    """
    """
    def __init__(self,
        name, 
        in_channels, 
        out_channels,
        kernel_size:    int=3,
        stride:         int=1,
        dilation:       int=1,
        activation:     str='relu',
        batch_norm:     bool=True,
        dimension:      int=3, 
    ):
        """
        """
        super(DoubleConv, self).__init__(dimension)
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.dimension = dimension
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bias = False
        else:
            self.bias = True
        self.activation = activation
        self.activation_fn = get_activation(self.activation)
        self.construct_model()

    def construct_model(self):
        """
        Create model dictionary
        """
        if self.in_channels != self.out_channels:
            self.residual = ME.MinkowskiLinear(
                self.in_channels, self.out_channels, bias=self.bias
            )
        else:
            self.residual = Identity()
        _first_conv = OrderedDict()
        _second_conv = OrderedDict()
        # create conv layer
        _first_conv[f'{self.name}_conv1'] = ME.MinkowskiConvolution(
            in_channels  = self.in_channels,
            out_channels = self.out_channels,
            kernel_size  = self.kernel_size,
            stride       = self.stride,
            dilation     = self.dilation,
            bias         = self.bias,
            dimension    = self.dimension
        )
        if self.batch_norm:
            _first_conv[f'{self.name}_batch_norm1'] = ME.MinkowskiBatchNorm(self.out_channels)
        # second conv layer
        _second_conv[f'{self.name}_conv2'] = ME.MinkowskiConvolution(
            in_channels  = self.out_channels,
            out_channels = self.out_channels,
            kernel_size  = self.kernel_size,
            stride       = self.stride,
            dilation     = self.dilation,
            bias         = self.bias,
            dimension    = self.dimension
        )
        if self.batch_norm:
            _second_conv[f'{self.name}_batch_norm2'] = ME.MinkowskiBatchNorm(self.out_channels)
        self.first_conv_dict = nn.ModuleDict(_first_conv)
        self.second_conv_dict = nn.ModuleDict(_second_conv)

    def forward(self, 
        x
    ):
        """
        Iterate over the module dictionary.
        """
        identity = self.residual(x)
        for layer in self.first_conv_dict.keys():
            x = self.first_conv_dict[layer](x)
        x = self.activation_fn(x)
        for layer in self.second_conv_dict.keys():
            x = self.second_conv_dict[layer](x)
        x = x + identity
        x = self.activation_fn(x)
        return x
    
class HitDataset(Dataset):
    def __init__(self, input_files):
        wire_plane_2 = []
        classes = []
        hits = []
        for input_file in input_files:
            f = np.load(input_file, allow_pickle=True)
            wire_plane_2.append(f['view_2_features'][0].astype(float))
            classes.append(f['view_2_classes'][0].astype(float))
            hits.append(f['view_2_hits'][0].astype(float))
        self.wire_plane_2 = np.array(wire_plane_2)
        self.classes = np.array(classes)
        self.hits = np.array(hits)
    
    def __len__(self):
        return len(self.wire_plane_2)
    
    def __getitem__(self, idx):
        hits = self.hits[idx]
        classes = self.classes[idx]
        features = self.wire_plane_2[idx]
        return features, hits
    
sparse_uresnet_params = {
    'in_channels':  1,
    'out_channels': [128],  # this is the number of classes for the semantic segmentation
    'classifications':  ['hit'],
    'filtrations':  [64, 128, 256, 512],    # the number of filters in each downsample
    'double_conv_params': {
        'kernel_size':       3,
        'stride':       1,
        'dilation':     1,
        'activation':   'relu',
        'dimension':    2,
        'batch_norm':   True,
    },
    'conv_transpose_params': {
        'kernel_size':    2,
        'stride':    2,
        'dilation':  1,
        'dimension': 2,
    },
    'max_pooling_params': {
        'kernel_size':   2,
        'stride':   2,
        'dilation': 1,
        'dimension':2,
    }
}
    
class UNet(ME.MinkowskiNetwork):

    def __init__(self, in_nchannel, out_nchannel, D):
        self.config = sparse_uresnet_params

        super(UNet, self).__init__(D)
        _down_dict = OrderedDict()
        _up_dict = OrderedDict()
        _classification_dict = OrderedDict()

        # iterate over the down part
        in_channels = self.config['in_channels']
        for filter in self.config['filtrations']:
            _down_dict[f'down_filter_double_conv{filter}'] = DoubleConv(
                name=f'down_{filter}',
                in_channels=in_channels,
                out_channels=filter,
                kernel_size=self.config['double_conv_params']['kernel_size'],
                stride=self.config['double_conv_params']['stride'],
                dilation=self.config['double_conv_params']['dilation'],
                dimension=self.config['double_conv_params']['dimension'],
                activation=self.config['double_conv_params']['activation'],
                batch_norm=self.config['double_conv_params']['batch_norm'],
            )
            # set new in channel to current filter size
            in_channels = filter

        # iterate over the up part
        for filter in reversed(self.config['filtrations']):
            _up_dict[f'up_filter_transpose{filter}'] = ME.MinkowskiConvolutionTranspose(
                in_channels=2*filter,   # adding the skip connection, so the input doubles
                out_channels=filter,
                kernel_size=self.config['conv_transpose_params']['kernel_size'],
                stride=self.config['conv_transpose_params']['stride'],
                dilation=self.config['conv_transpose_params']['dilation'],
                dimension=self.config['conv_transpose_params']['dimension']    
            )
            _up_dict[f'up_filter_double_conv{filter}'] = DoubleConv(
                name=f'up_{filter}',
                in_channels=2*filter,
                out_channels=filter,
                kernel_size=self.config['double_conv_params']['kernel_size'],
                stride=self.config['double_conv_params']['stride'],
                dilation=self.config['double_conv_params']['dilation'],
                dimension=self.config['double_conv_params']['dimension'],
                activation=self.config['double_conv_params']['activation'],
                batch_norm=self.config['double_conv_params']['batch_norm'],
            )

        # create bottleneck layer
        self.bottleneck = DoubleConv(
            name=f"bottleneck_{self.config['filtrations'][-1]}",
            in_channels=self.config['filtrations'][-1],
            out_channels=2*self.config['filtrations'][-1],
            kernel_size=self.config['double_conv_params']['kernel_size'],
            stride=self.config['double_conv_params']['stride'],
            dilation=self.config['double_conv_params']['dilation'],
            dimension=self.config['double_conv_params']['dimension'],
            activation=self.config['double_conv_params']['activation'],
            batch_norm=self.config['double_conv_params']['batch_norm'],
        )

        # create output layer
        for ii, classification in enumerate(self.config['classifications']):
            _classification_dict[f"{classification}"] = ME.MinkowskiConvolution(
                in_channels=self.config['filtrations'][0],      # to match first filtration
                out_channels=self.config['out_channels'][ii],   # to the number of classes
                kernel_size=1,                                  # a one-one convolution
                dimension=self.config['double_conv_params']['dimension'],
            )

        # create the max pooling layer
        self.max_pooling = ME.MinkowskiMaxPooling(
            kernel_size=self.config['max_pooling_params']['kernel_size'],
            stride=self.config['max_pooling_params']['stride'],
            dilation=self.config['max_pooling_params']['dilation'],
            dimension=self.config['max_pooling_params']['dimension']
        )

        # create the dictionaries
        self.module_down_dict = nn.ModuleDict(_down_dict)
        self.module_up_dict = nn.ModuleDict(_up_dict)
        self.classification_dict = nn.ModuleDict(_classification_dict)

    def forward(self, x):
        # record the skip connections
        skip_connections = {}
        # iterate over the down part
        for filter in self.config['filtrations']:
            x = self.module_down_dict[f'down_filter_double_conv{filter}'](x)
            skip_connections[f'{filter}'] = x
            x = self.max_pooling(x)
        # through the bottleneck layer
        x = self.bottleneck(x)
        
        for filter in reversed(self.config['filtrations']):
            x = self.module_up_dict[f'up_filter_transpose{filter}'](x)
            # concatenate the skip connections
            skip_connection = skip_connections[f'{filter}']
            # check for compatibility
            if x.shape != skip_connection.shape:
                x = F.resize(x, size=skip_connection.shape[2:])

            concat_skip = ME.cat(skip_connection, x)
            x = self.module_up_dict[f'up_filter_double_conv{filter}'](concat_skip)

        return self.classification_dict['hit'](x)

# Define the Point Proposal Network (PPN) with UNet
class PPNWithUNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(PPNWithUNet, self).__init__()

        # UNet with residual connections
        self.unet = UNet(in_channels, 128, 2)

        self.fc_reg1 = ME.MinkowskiLinear(128, 64)
        self.fc_reg2 = ME.MinkowskiLinear(64, 4)  # Subtract 1 for the offset predictions

        # Define the network architecture for classification
        self.fc_cls = ME.MinkowskiLinear(128, 1)

        # define the decoder part
        self.decoder1 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=4,
            out_channels=4,
            kernel_size=3,
            stride=1,
            dimension=2    
        )
        self.decoder2 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=4,
            out_channels=4,
            kernel_size=3,
            stride=1,
            dimension=2    
        )
        self.decoder3 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=4,
            out_channels=1,
            kernel_size=3,
            stride=1,
            dimension=2    
        )

    def forward(self, x):
        # Pass through UNet
        # Forward pass through the network
        features = x[0][:,2].unsqueeze(1).float()
        coordinates = x[0][:,:2].float()
        batch_indices = torch.ones((len(coordinates),1))
        coordinates = torch.cat(
            (batch_indices, coordinates),
            dim=1
        ).int()
        # Create a sparse tensor with dimension=2
        input_data = ME.SparseTensor(
            features=features,
            coordinates=coordinates,
        )
        x_unet = self.unet(input_data)
        # PPN layers (modify as needed)
        x_reg = self.fc_reg1(x_unet)
        x_reg = ME.MinkowskiReLU()(x_reg)
        x_reg = self.fc_reg2(x_reg)
        x_classification = self.fc_cls(x_unet)
        x_classification = ME.MinkowskiSigmoid()(x_classification)

        x_pred = torch.BoolTensor(x_classification.F > .5).squeeze(1)
        x_pred = ME.SparseTensor(
            features=x_reg.F[x_pred],
            coordinates=x_reg.C[x_pred]
        )

        x_decoder = self.decoder3(x_pred)
        # x_decoder = self.decoder2(x_decoder)
        # x_decoder = self.decoder3(x_decoder)

        return x_reg, x_classification, x_decoder, x_pred

class HitProposalTrainer:
    def __init__(self,
        model
    ):
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)

    def loss_fn(self, reg_output, cls_output, target, decoder, input, pred_indices):
        # Define custom loss function for regression and classification

        # Separate regression and classification targets
        target = target[0].float()

        target_mask = (target[:,-1] != -1)

        regression_target = target[target_mask]  # Exclude rows with -1 as the last element
        
        classification_target = torch.zeros(len(target))
        classification_target[target_mask] = 1.0
        
        # figure out what the original coords were and
        # what the decoder created.  Then, look at the differences
        # between the two to define the classification.
        input_coords = input[0][:,:2]
        decoder_coords = decoder.C[:,1:]
        unique_coords = torch.cat((input_coords, decoder_coords), dim=0).unique(dim=0)
        answer = torch.zeros(len(unique_coords))
        pred = torch.zeros(len(unique_coords))
        for ii, coord in enumerate(unique_coords):
            if coord in input_coords:
                answer[ii] = 1.0
            if coord in decoder_coords:
                pred[ii] = 1.0
        print(sum(answer),sum(pred))
        # Separate predicted values for regression and classification
        regression_output = reg_output[target_mask]
        classification_output = cls_output[:, 0]

        # Regression loss (e.g., Mean Squared Error)
        regression_loss = F.mse_loss(regression_output, regression_target, reduction='mean') / len(regression_target)
        classification_loss = F.binary_cross_entropy_with_logits(classification_output, classification_target)

        decoder_reco_loss = F.binary_cross_entropy_with_logits(answer, pred)
        print(decoder_reco_loss)
        #decoder_adc_loss = F.mse_loss()

        
        print(f"regression: {regression_loss}, hit: {classification_loss}, decoder_reco: {decoder_reco_loss}")#, decoder_adc: {decoder_adc_loss}")
        total_loss = classification_loss + decoder_reco_loss
        return total_loss

    def train(self, dataloader, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            all_predictions = []
            all_labels = []

            for batch_input, batch_target in dataloader:
                self.optimizer.zero_grad()
                
                # Forward pass
                reg_output, classification_output, decoder_output, pred_output = self.model(batch_input)
                
                # Calculate loss with MSE for the classification task
                regression_loss = self.loss_fn(reg_output.F, classification_output.F, batch_target, decoder_output, batch_input, pred_output)
                
                # Backpropagation
                regression_loss.backward()
                self.optimizer.step()
                all_predictions.append(classification_output.F.cpu().detach().round())
                target_mask = (batch_target[0][:,-1] != -1)
                classification_target = torch.zeros(len(batch_target[0]))
                classification_target[target_mask] = 1.0
                all_labels.append(classification_target.unsqueeze(1))

            
            # Compute precision and recall for the entire epoch
            all_predictions = torch.cat(all_predictions).numpy()
            all_labels = torch.cat(all_labels).numpy()
            precision = precision_score(all_labels, all_predictions)
            recall = recall_score(all_labels, all_predictions)

            # Print epoch statistics
            #avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {regression_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    def infer(self, dataloader):
        self.model.eval()
        input = []
        target = []
        reg_output = []
        class_output = []
        for batch_input, batch_target in dataloader:
            regs, classes = self.model(batch_input)
            input.append(batch_input[0].cpu().numpy())
            target_mask = (batch_target[0][:,-1] != -1)
            classification_target = torch.zeros(len(batch_target[0]))
            classification_target[target_mask] = 1.0
            target.append(classification_target.cpu().numpy())
            reg_output.append(regs.F.detach().cpu().numpy())
            class_output.append(classes.F.detach().cpu().numpy())
        # input = np.array(input)
        # target = np.array(target)
        # reg_output = np.array(reg_output)
        # class_output = np.array(class_output)
        return input, target, reg_output, class_output

if __name__ == "__main__":
    dataset = HitDataset([
        "data/labeling_sim_arrakis_0/tpc1.npz",
        "data/labeling_sim_arrakis_0/tpc2.npz",
        "data/labeling_sim_arrakis_0/tpc5.npz",
        "data/labeling_sim_arrakis_0/tpc6.npz",
        "data/labeling_sim_arrakis_0/tpc9.npz",
        "data/labeling_sim_arrakis_0/tpc10.npz"
    ])
    batch_size = 1

    # Create data loaders.
    train_dataloader = DataLoader(dataset, batch_size=batch_size)
    
    #model = PPNModel(num_input_features=1, num_output_features=4)
    model = PPNWithUNet(1, 4)
    trainer = HitProposalTrainer(model)
    trainer.train(train_dataloader, 50)

    input, target, reg_output, class_output = trainer.infer(train_dataloader)

    event = 0
    input = input[event]
    target = target[event]
    reg_output = reg_output[event]
    class_output = class_output[event].squeeze(1)

    fig, axs = plt.subplots(1,2,figsize=(10,6))
    true_hits = (target == 1)
    not_true_hits = (target == 0)
    
    

    pred_hits = (class_output > 0.5)
    not_pred_hits = (class_output < 0.5)

    print(class_output)
    
    false_negatives = true_hits & not_pred_hits
    false_positives = not_true_hits & pred_hits
    true_positives = true_hits & pred_hits
    true_negatives = not_true_hits & not_pred_hits
    
    print(f"False negatives: {sum(false_negatives)}/{sum(true_hits)}")
    print(f"False positives: {sum(false_positives)}/{sum(not_true_hits)}")
    print(f"True positives: {sum(true_positives)}/{sum(true_hits)}")
    print(f"True negatives: {sum(true_negatives)}/{sum(not_true_hits)}")

    axs[0].scatter(input[true_hits][:,0], input[true_hits][:,1], label="true hits", c='k')
    axs[1].scatter(input[pred_hits][:,0], input[pred_hits][:,1], label="pred hits", c='r')

    axs[0].legend()
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("output.png")
    plt.show()



    print("Done!")