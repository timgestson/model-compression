import lightning as L
import torch
from torch import nn
import torch.functional as F


class ChannelPruning(L.LightningModule):
  def __init__(self) -> None:
    super().__init__()

    self.backbone = nn.Sequential(
      nn.Conv2d(1, 64, 3, padding=1, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 128, 3, padding=1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Conv2d(128, 256, 3, padding=1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(),
      nn.Conv2d(256, 256, 3, padding=1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Conv2d(256, 512, 3, padding=1, bias=False),
      nn.BatchNorm2d(512),
      nn.ReLU(),
      nn.Conv2d(512, 512, 3, padding=1, bias=False),
      nn.BatchNorm2d(512),
      nn.ReLU(),
      nn.MaxPool2d(2),
    )
    self.classifier = nn.Linear(512, 10)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.backbone(x)

    x = x.mean([2, 3])

    x = self.classifier(x)
    return x

  def prune(self, ratio):
    convs = [ c for c in self.backbone if isinstance(c, nn.Conv2d)]
    bns = [ c for c in self.backbone if isinstance(c, nn.BatchNorm2d)]
    for i in range(len(convs) - 1):
        conv_layer = convs[i]
        bn_layer = bns[i]
        next_conv_layer = convs[i+1]
        kept = int(next_conv_layer.in_channels * (1 - ratio))
        importances = torch.argsort(
            torch.Tensor(
                [
                    next_conv_layer.weight.detach()[:, i].norm().item()
                    for i
                    in range(next_conv_layer.weight.shape[1])
                ]
            ),
            descending=True
        )
        
        with torch.no_grad():
            conv_layer.weight.set_(
                torch.index_select(conv_layer.weight.detach(), 0, importances)[:kept]
            )
            bn_layer.weight.set_(
                torch.index_select(bn_layer.weight.detach(), 0, importances)[:kept]
            )
            bn_layer.bias.set_(
                torch.index_select(bn_layer.bias.detach(), 0, importances)[:kept]
            )
            bn_layer.running_mean.set_(
                torch.index_select(bn_layer.running_mean.detach(), 0, importances)[:kept]
            )
            bn_layer.running_var.set_(
                torch.index_select(bn_layer.running_var.detach(), 0, importances)[:kept]
            )

            next_conv_layer.weight.set_(
                torch.index_select(next_conv_layer.weight.detach(), 1, importances)[:,:kept]
            )

  def training_step(self, batch, batch_idx):
    x, y = batch
    z = self.forward(x)
    loss = F.cross_entropy(z,y)
    self.log("train_loss", loss)
    return loss

  def test_step(self, batch, batch_idx):
    x, y = batch
    z = self.forward(x)
    test_loss = F.cross_entropy(z,y)
    outputs = torch.argmax(z, dim=1)
    accuracy = ((outputs == y).sum() / y.size(0) * 100).item()
    self.log('test_accuracy', accuracy) 
    self.log("test_loss", test_loss)
  
  def configure_optimizers(self):
    optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
    return optimizer