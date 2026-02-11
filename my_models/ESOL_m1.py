import rdkit
from torch_geometric.datasets import MoleculeNet

data = MoleculeNet(root='data/ESOL', name='ESOL')

import torch
from torch.nn import Linear
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
embedding_size = 64

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList([
            GCNConv(
                data.num_features if i == 0 else embedding_size,
                embedding_size)
            for i in range(4)
        ])

        self.out = Linear(embedding_size, 1)
    def forward(self, x, edge_index, batch_index):
        hidden = x
        for conv in self.convs:
            hidden = conv(hidden, edge_index)
            hidden = F.relu(hidden)

        hidden = gap(hidden, batch_index)
        out = self.out(hidden)
        return out, hidden

model = GCN()
print(model)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

from torch_geometric.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

data_size = len(data)
NUM_GRAPHS_PER_BATCH = 64
loader = DataLoader(data[:int(data_size * 0.8)], 
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
test_loader = DataLoader(data[int(data_size * 0.8):], 
                         batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)

def train():
    for batch in loader:
      batch.to(device)  
      optimizer.zero_grad()
      pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch) 
      loss = loss_fn(pred, batch.y) 
      loss.backward()  
      optimizer.step()
    return loss, embedding

print("Starting training...")
losses = []
for epoch in range(2000):
    loss, h = train()
    losses.append(loss)
    if epoch % 100 == 0:
      print(f"Epoch {epoch} | Train Loss {loss}")

from master_functions import visualize_losses
visualize_losses(losses)

from master_functions import model_eval

model_eval(model, test_loader, device)