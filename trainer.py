import torch
from tqdm import tqdm
from monai.metrics import DiceMetric

class Trainer:
    def __init__(self, model, optimizer, loss_function, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")

    def train_epoch(self, dataloader, epoch):
        """Train for one epoch using full volumes"""
        self.model.train()
        epoch_loss = 0

        progress = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}")

        for step, (images, masks) in progress:
            images, masks = images.to(self.device), masks.to(self.device)

            # Clear gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)

            # Calculate loss
            loss = self.loss_function(outputs, masks)

            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Update statistics
            epoch_loss += loss.item()
            progress.set_postfix({"loss": loss.item()})

        return epoch_loss / (step + 1)

    def evaluate(self, data_loader):
        self.model.eval()
        dice_scores = []
        self.dice_metric.reset()

        with torch.no_grad():
            for images, masks in data_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                binary_outputs = (torch.sigmoid(outputs) > 0.5).float()
                self.dice_metric(y_pred=binary_outputs, y=masks)
                dice_scores.append(self.dice_metric.aggregate().item())

        avg_dice = sum(dice_scores) / len(dice_scores)
        return avg_dice