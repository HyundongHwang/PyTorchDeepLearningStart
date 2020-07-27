import myutil as mu
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset  # 텐서데이터셋
from torch.utils.data import DataLoader  # 데이터로더
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [[73, 80, 75],
                       [93, 88, 93],
                       [89, 91, 90],
                       [96, 98, 100],
                       [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y


dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
model = nn.Linear(3, 1)
optimizer = optim.SGD(model.parameters(), lr=1e-5)
nb_epoches = 20

for epoch in range(nb_epoches + 1):
    print("=" * 80)
    for batch_idx, samples in enumerate(dataloader):
        print("-" * 80)
        mu.log("batch_idx", batch_idx)
        mu.log("samples", samples)
        prediction = model(torch.FloatTensor(dataset.x_data))
        cost = F.mse_loss(prediction, torch.FloatTensor(dataset.y_data))
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print("epoch {:4d}/{} batch {}/{} cost {:.6f}".format(
            epoch,
            nb_epoches,
            batch_idx + 1,
            len(dataloader),
            cost.item()
        ))

        mu.log_model("model", model)
