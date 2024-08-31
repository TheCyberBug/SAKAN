import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class SentimentTrainer:
    """
    Training and evaluating the sentiment analysis model with PyTorch.

    Handles the training loop, data preparation, and evaluation.
    """
    def __init__(self, model, device, batch_size=32):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

    def prepare_data(self, X_train, y_train, X_test, y_test):
        """
        Prepare the training and testing data.

        Converts numpy arrays to PyTorch tensors and creates DataLoader
        objects for both training and testing datasets.

        Args:
            X_train (np.array): Training features.
            y_train (np.array): Training labels.
            X_test (np.array): Testing features.
            y_test (np.array): Testing labels.
        """
        X_train_tensor = torch.tensor(X_train.tolist()).to(self.device)
        X_test_tensor = torch.tensor(X_test.tolist()).to(self.device)
        y_train_tensor = torch.tensor(y_train.tolist()).to(self.device)
        y_test_tensor = torch.tensor(y_test.tolist()).to(self.device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def calculate_accuracy(self, preds, labels):
        """
        Calculate the accuracy.

        Args:
            preds (torch.Tensor): Model predictions.
            labels (torch.Tensor): True labels.

        Returns:
            int: Number of correct predictions.
        """
        _, predicted = torch.max(preds, 1)
        correct = (predicted == labels).sum().item()
        return correct

    def evaluate(self):
        """
        Evaluate the model on the test dataset.

        Returns:
            tuple: A tuple containing (accuracy, average_loss).
        """
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for texts, labels in self.test_loader:
                outputs = self.model(texts)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                correct += self.calculate_accuracy(outputs, labels)
                total += labels.size(0)
        accuracy = correct / total
        avg_loss = total_loss / total
        return accuracy, avg_loss

    def train(self, num_epochs):
        """
        Train the model for a specified number of epochs.
        Handles the entire training loop, including evaluation after each epoch.

        Args:
            num_epochs (int): Number of epochs to train for.

        Returns:
            list: A list of tuples containing (validation_accuracy, validation_loss) for each epoch.
        """
        model_history = []

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0

            with tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
                for texts, labels in pbar:
                    self.optimizer.zero_grad()
                    outputs = self.model(texts)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()
                    correct_predictions += self.calculate_accuracy(outputs, labels)
                    total_predictions += labels.size(0)

                    accuracy = correct_predictions / total_predictions

                    pbar.set_postfix(loss=total_loss / total_predictions, accuracy=accuracy)
            self.scheduler.step()

            val_accuracy, val_loss = self.evaluate()
            model_history.append((val_accuracy, val_loss))
            print(
                f'Epoch {epoch + 1}, Loss: {total_loss / (len(self.train_loader) * self.batch_size)}, Accuracy: {accuracy}')
            print(f'Val_Loss: {val_loss}, Val_Accuracy: {val_accuracy}')

        return model_history
