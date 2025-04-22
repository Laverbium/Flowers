import mlflow

import mlflow.pyfunc
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
# Set the MLflow tracking URI if needed
# mlflow.set_tracking_uri("your_tracking_uri")
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
set_seed(100)
device = torch.device('cuda')
CEloss = torch.nn.CrossEntropyLoss()
def eval(model, loader, type_data, step=0):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            print(outputs, labels)
            loss = CEloss(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    loss = total_loss / len(loader)
    #mlflow.log_metric(f"{type_data}_loss", loss, step=step)
    #mlflow.log_metric(f"{type_data}_accuracy", acc, step=step)
    model.train()
    return acc, loss

dataset = datasets.ImageFolder(root='flowers', transform=resnet_transform_advanced if RESNET else simple_transform)
#labels = [dataset[i][1] for i in range(len(dataset))]

train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])
test_loader = DataLoader(test_dataset, batch_size=32)
def load_mlflow_model(run_id, model_path="model"):
    """
    Load an MLflow model from artifacts
    
    Args:
        run_id (str): MLflow run ID
        model_path (str): Path to model in artifacts, defaults to "model"
    
    Returns:
        loaded_model: The loaded MLflow model
    """
    try:
        # Load model from MLflow
        mlflow.set_tracking_uri("http://localhost:6363")
        loaded_model = mlflow.pytorch.load_model(f"runs:/{run_id}/{model_path}")
        return loaded_model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

if __name__ == "__main__":
    dataset = datasets.ImageFolder(root='flowers', transform=resnet_transform_advanced if RESNET else simple_transform)
    #labels = [dataset[i][1] for i in range(len(dataset))]

    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])
    # Example usage
    RUN_ID = "fcc62012e0d8469a853d6d3a344319d7"
    model = load_mlflow_model(RUN_ID)
    acc, loss = eval(model, test_loader, "test")
    print(f"Test Accuracy: {acc:.2f}%, Test Loss: {loss:.4f}")
    # for name, param in model.named_parameters():
    #     print(name, param.shape, param.requires_grad)
    # print('-'*20)
    model
    plt.imshow(model.resnet.fc.weight.data[0].view(16,32).cpu().numpy())
    plt.show()
    if model:
        print("Model loaded successfully")