import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from model import FlowerModel
import mlflow
import mlflow.pytorch

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 10
num_classes = 5
input_image_shape = (256, 256)

# Data preparation

simple_transform = v2.Compose([
        v2.Resize(input_image_shape),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

dataset = datasets.ImageFolder(root='flowers', transform=simple_transform)

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = datasets.ImageFolder(root='data/val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, loss, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FlowerModel(n_blocks=4, start_channels=32).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize MLflow
mlflow.set_tracking_uri("http://localhost:8001")  # Adjust the URI as needed
mlflow.set_experiment("Flower Classification")

with mlflow.start_run():
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("num_classes", num_classes)

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_accuracy = 100. * correct / total
        mlflow.log_metric("train_loss", running_loss / len(train_loader), step=epoch)
        mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_accuracy = 100. * val_correct / val_total
        mlflow.log_metric("val_loss", val_loss / len(val_loader), step=epoch)
        mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Log the model
    mlflow.pytorch.log_model(model, "model")
    

    if __name__ == "__main__":
        model, best_accuracy = train_model(
            batch_size=32,
            learning_rate=0.001,
            num_epochs=10,
            num_classes=5,
            n_blocks=4,
            start_channels=32
        )