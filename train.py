import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import v2
from model import FlowerModel, ResNet, Inceptionv3
import mlflow
import numpy as np
from tqdm import tqdm
import sys
import argparse
import warnings

def parse_args():
    parser = argparse.ArgumentParser(description='Flower Classification')
    parser.add_argument('--batch_size', required=False, type=int, default=32, help='Batch size for training')
    parser.add_argument('--model', type=str, choices=['resnet', 'simple', 'inception'], default='simple', help='Model type to use')
    parser.add_argument('--lr',type=float, default=3e-4)
    parser.add_argument('--n_epochs', type=int, default=10)
    return parser.parse_args()

args = parse_args()

warnings.filterwarnings("ignore", module='mlflow')

random_seed = 100
batch_size = args.batch_size
learning_rate = args.lr
n_epochs = args.n_epochs
n_classes = 5
early_stop = 5
input_image_shape = (256, 256)
model_params = dict(n_blocks=4, start_channels=32)
MODEL = args.model
RUN_NAME = 'inception_advanced'
resize_shape = (256, 256)
if MODEL == 'resnet':
    resize_shape = (224,224)
elif MODEL == 'inception':
    resize_shape = (299, 299)
#torch.cuda.empty_cache()

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
set_seed(random_seed)
get_transform_from_model = {'simple':v2.Compose([
        v2.Resize(resize_shape),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ]),
    'resnet':v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'inception':v2.Compose([
    v2.Resize((299,299)),
    #v2.CenterCrop(299),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])}
transform_advanced = v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomResizedCrop(size = resize_shape, scale=(0.8,0.9)),
        v2.RandomRotation(degrees=30),
        v2.GaussianBlur(kernel_size=(9, 9), sigma=(0.5, 3.)),
        v2.RandomInvert(),
        v2.RandomAdjustSharpness(2),
        v2.RandomAutocontrast(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
test_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((300,300)),
    v2.CenterCrop(resize_shape),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
#transform = get_transform_from_model[MODEL]
transform = transform_advanced
dataset = datasets.ImageFolder(root='flowers', transform=transform)
#labels = [dataset[i][1] for i in range(len(dataset))]
print('Input image shape:', dataset[0][0].shape)

train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])
val_dataset.dataset.transform = test_transform
test_dataset.dataset.transform = test_transform

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if MODEL == 'resnet':
    model = ResNet(pretrained=True, freeze_layers=False)
elif MODEL == 'inception':
    model = Inceptionv3(pretrained=True, freeze_layers=False, top_layer=True, aux_logits = True)
elif MODEL == 'simple':
    model = FlowerModel(**model_params)

if model != 'simple':
    if model.freeze:
        param_groups = [{'params': model.fc.parameters()}]
    else:
        param_groups = [
                    {'params': model.body.parameters(), 'lr': 5e-6},
                    {'params': model.fc.parameters()}]
else:
    param_groups = [{'params': model.parameters()}]


model = model.to(device)
print(next(model.parameters()).device)

CEloss = nn.CrossEntropyLoss()
optimizer = optim.AdamW(param_groups, lr=learning_rate, weight_decay=0.05)

scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

def eval(loader, type_data, step=0):
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
    mlflow.log_metric(f"{type_data}_loss", loss, step=step)
    mlflow.log_metric(f"{type_data}_accuracy", acc, step=step)
    model.train()
    return acc, loss
    
total_steps = 0
early_stop_epochs = 0
best_val_loss = float('inf')
best_train_loss = float('inf')
step_log = len(train_loader) // 10

#print(step_log)
#sys.exit()
mlflow.set_tracking_uri("http://localhost:6363")   #mlflow server --host 127.0.0.1 --port 6363
mlflow.set_experiment("Flower Classification")


run = mlflow.start_run(run_name = RUN_NAME if RUN_NAME else None)
mlflow.log_param("model", MODEL)
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("learning_rate", learning_rate)
mlflow.log_param("num_epochs", n_epochs)
mlflow.log_params(model_params)
mlflow.log_param("seed", random_seed)
mlflow.log_text(str(transform_advanced), artifact_file="transform.txt")

for epoch in range(1, n_epochs+1):
    
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for step, (images, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}/{n_epochs}'), start=1):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        if MODEL == 'inception':
            if model.aux_logits:
                loss = CEloss(outputs[0], labels) + 0.4*CEloss(outputs[1], labels)
            else:
                loss = CEloss(outputs, labels) 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        outputs = outputs[0] if MODEL == 'inception' else outputs
        _, predicted = outputs.max(dim=1)
        
        total += batch_size
        correct += predicted.eq(labels).sum().item()
        total_steps += batch_size
        if  step == 1 or step % step_log == 0:
            mlflow.log_metric("train_loss", train_loss / step, step=total_steps)
            mlflow.log_metric("train_accuracy", 100. * correct / total, step=total_steps)

    epoch_train_loss  = train_loss / len(train_loader)
    epoch_train_acc = 100. * correct / total 
    mlflow.log_metric("epoch_loss", epoch_train_loss, step=epoch)
    mlflow.log_metric("epoch_train_accuracy", epoch_train_acc, step=epoch)

    val_acc, val_loss = eval(val_loader, "val", epoch)
    
    print(f"Epoch [{epoch}/{n_epochs}], \
    Train Loss: {epoch_train_loss:.4f},\
    Train Accuracy: {epoch_train_acc:.2f}, \
    Validation Loss: {val_loss/len(val_loader):.4f}, \
    Validation Accuracy: {val_acc:.2f}%")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        mlflow.pytorch.log_model(model, "model", pip_requirements=["torch=2.4.1"])
    if best_train_loss > epoch_train_loss:
        best_train_loss = epoch_train_loss
        early_stop_epochs = 0
    else:
        early_stop_epochs += 1
        if early_stop_epochs >= early_stop:
            print(f"Early stopping at epoch {epoch}")
            break
        
    mlflow.log_metric('lr', scheduler.get_last_lr()[0], step=epoch)
    scheduler.step()
            
test_acc, test_loss = eval(test_loader, "test")
print(f"test loss: {test_loss:.4f},\
      test accuracy: {test_acc:.2f}")
torch.save(model.state_dict(), "models/flower_model_test.pth")
mlflow.end_run()
