import mlflow
import mlflow.pyfunc
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
# Set the MLflow tracking URI if needed
# mlflow.set_tracking_uri("your_tracking_uri")
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
set_seed(100)
test_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((300,300)),
    v2.CenterCrop((224,224)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
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

def get_predicted_labels(model, loader):
    """
    Get the predicted labels from the model
    
    Args:
        model: The loaded MLflow model
    
    Returns:
        predicted_labels: The predicted labels
    """
    model.eval()
    labels = torch.tensor([], device=device)
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted_labels = outputs.max(1)
            #print(labels.device, predicted_labels.device)
            labels = torch.cat((labels, predicted_labels), dim=0)
    return labels.cpu().numpy()


if __name__ == "__main__":
    dataset = datasets.ImageFolder(root='flowers', transform=test_transform)
    #labels = [dataset[i][1] for i in range(len(dataset))]

    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    val_labels = [val_dataset[i][1] for i in range(len(val_dataset))]
    test_labels = [test_dataset[i][1] for i in range(len(test_dataset))]
    print(f"Train labels: {np.unique(train_labels, return_counts=True)}")
    print(f"Validation labels: {np.unique(val_labels, return_counts=True)}")
    print(f"Test labels: {np.unique(test_labels, return_counts=True)}")
    
    test_loader, val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False), DataLoader(val_dataset, batch_size=32, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    # Example usage
    RUN_ID = "a31aee994a0246308814e19683a5da2f"
    model = load_mlflow_model(RUN_ID)
    model.to(device)
    # acc, loss = eval(model, test_loader, "test")
    # print(f"Test Accuracy: {acc:.2f}%, Test Loss: {loss:.4f}")
    acc, loss = eval(model, train_loader, "train")
    print(f"Train Accuracy: {acc:.2f}%, Train Loss: {loss:.4f}")
    # val_preds = get_predicted_labels(model, val_loader)
    # test_preds = get_predicted_labels(model, test_loader)
    train_preds = get_predicted_labels(model, train_loader)
    # print(val_preds.shape, test_preds.shape)
    classes = dataset.classes
    print("train Classification Report:")
    print(classification_report(train_labels, train_preds, target_names=classes))
    # print("Test Classification Report:")
    # print(classification_report(test_labels, test_preds, target_names=classes))
    ConfusionMatrixDisplay.from_predictions(train_labels, train_preds, display_labels=classes)
    # plt.title("Validation Confusion Matrix")
    plt.show()
    # ConfusionMatrixDisplay.from_predictions(test_labels, test_preds, display_labels=classes)
    # plt.title("Test Confusion Matrix")
    #plt.show()


    # for name, param in model.named_parameters():
    #     print(name, param.shape, param.requires_grad)
    # print('-'*20)
    # model
    # plt.imshow(model.resnet.fc.weight.data[0].view(16,32).cpu().numpy())
    # plt.show()
    # if model:
    #     print("Model loaded successfully")