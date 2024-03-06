from torch import optim
import torch.nn.functional as F
import lightning as L

class Distillation(L.LightningModule):
    def __init__(self, teacher, student, temperature):
        super().__init__()
        self.teacher = teacher
        self.teacher.freeze()
        self.student = student
        self.temperature = temperature
    
    def training_step(self, batch, _batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        sz = self.student(x)
        tz = self.teacher(x)
        teacher_soft_targets = F.softmax(tz / self.temperature)
        student_soft_targets = F.softmax(sz / self.temperature)
        loss = F.cross_entropy(sz,y) + F.cross_entropy(teacher_soft_targets, student_soft_targets)
        self.log("train_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.student(x)
        test_loss = F.cross_entropy(z,y) 
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer