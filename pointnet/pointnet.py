import torch
import torch_geometric




from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool
from pytorch_metric_learning.losses import NTXentLoss
from torch_geometric.datasets import ShapeNet
import tqdm



# We're lucky and pytorch geometric helps us with pre-implemented transforms 
# which can also be applied on the whole batch directly 
augmentation = T.Compose([T.RandomJitter(0.03), T.RandomFlip(1), T.RandomShear(0.2)])

class Model(torch.nn.Module):
    def __init__(self, k=20, aggr='max'):
        super().__init__()
        # Feature extraction
        self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        # Encoder head 
        self.lin1 = Linear(128 + 64, 128)
        # Projection head (See explanation in SimCLRv2)
        self.mlp = MLP([128, 256, 32], norm=None)

    def forward(self, data, train=True):
        if train:
            # Get 2 augmentations of the batch
            augm_1 = augmentation(data)
            augm_2 = augmentation(data)

            # Extract properties
            pos_1, batch_1 = augm_1.pos, augm_1.batch
            pos_2, batch_2 = augm_2.pos, augm_2.batch

            # Get representations for first augmented view
            x1 = self.conv1(pos_1, batch_1)
            x2 = self.conv2(x1, batch_1)
            h_points_1 = self.lin1(torch.cat([x1, x2], dim=1))

            # Get representations for second augmented view
            x1 = self.conv1(pos_2, batch_2)
            x2 = self.conv2(x1, batch_2)
            h_points_2 = self.lin1(torch.cat([x1, x2], dim=1))
            
            # Global representation
            h_1 = global_max_pool(h_points_1, batch_1)
            h_2 = global_max_pool(h_points_2, batch_2)
        else:
            x1 = self.conv1(data.pos, data.batch)
            x2 = self.conv2(x1, data.batch)
            h_points = self.lin1(torch.cat([x1, x2], dim=1))
            return global_max_pool(h_points, data.batch)

        # Transformation for loss function
        compact_h_1 = self.mlp(h_1)
        compact_h_2 = self.mlp(h_2)
        return h_1, h_2, compact_h_1, compact_h_2

def train():
    model.train()
    total_loss = 0
    for _, data in enumerate(tqdm.tqdm(data_loader)):
        data = data.to(device)
        optimizer.zero_grad()
        # Get data representations
        h_1, h_2, compact_h_1, compact_h_2 = model(data)
        # Prepare for loss
        embeddings = torch.cat((compact_h_1, compact_h_2))
        # The same index corresponds to a positive pair
        indices = torch.arange(0, compact_h_1.size(0), device=compact_h_2.device)
        labels = torch.cat((indices, indices))
        loss = loss_func(embeddings, labels)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(dataset)

if __name__ == "__main__":
    dataset = ShapeNet(root=".", categories=["Table", "Lamp", "Guitar", "Motorbike"]).shuffle()[:500]
    print("Number of Samples: ", len(dataset))
    print("Sample: ", dataset[0])
    print(dataset[0].pos)
    print(dataset[0].y)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    loss_func = NTXentLoss(temperature=0.10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Use a large batch size (might lead to RAM issues)
    # Free Colab Version has ~ 12 GB of RAM
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(1, 4):
        model.train()
        total_loss = 0
        for _, data in enumerate(tqdm.tqdm(data_loader)):
            data = data.to(device)
            optimizer.zero_grad()
            # Get data representations
            h_1, h_2, compact_h_1, compact_h_2 = model(data)
            # Prepare for loss
            embeddings = torch.cat((compact_h_1, compact_h_2))
            # The same index corresponds to a positive pair
            indices = torch.arange(0, compact_h_1.size(0), device=compact_h_2.device)
            labels = torch.cat((indices, indices))
            loss = loss_func(embeddings, labels)
            loss.backward()
            total_loss += loss.item() * data.num_graphs
            optimizer.step()
            loss = train()
        print(f'Epoch {epoch:03d}, Loss: {total_loss/len(dataset):.4f}')
        scheduler.step()
