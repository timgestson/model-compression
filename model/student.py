from torch import optim, nn
import torch.nn.functional as F
import lightning as L

class MnistStudent(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28, 800),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Linear(800, 10),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net.forward(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.net(x)
        loss = F.cross_entropy(z,y) 
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.net(x)
        test_loss = F.cross_entropy(z,y) 
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer