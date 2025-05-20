import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms import v2
import torch
from PIL import Image
import os
import argparse
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import mlflow
import numpy as np
import math

resize_shape = (224, 224)
test_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((300,300)),
        v2.CenterCrop(resize_shape),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(path_or_id):
    if os.path.isfile(path_or_id):
        model = torch.load(path_or_id)
    else:
        try:
            # Load model from MLflow
            mlflow.set_tracking_uri("http://localhost:6363")
            model = mlflow.pytorch.load_model(f"runs:/{path_or_id}/model")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    model.eval()
    return model

def process_image(image_path, resize_shape = (224, 224)):
    image = Image.open(image_path).convert('RGB')
    return test_transform(image), image

def plot_predictions(images, predictions, class_names, cols=3):
    n = len(images)
    rows = math.ceil(n / cols)
    
    figsize = (19,10) 
    
    fig, axs = plt.subplots(rows, cols * 2, figsize=figsize)
    y_pos = range(len(class_names))
    if rows == 1:
        axs = [axs]
    for idx, (img, pred) in enumerate(zip(images, predictions)):
        row = idx // cols
        col = idx % cols
        
        ax_img = axs[row][2 * col]
        ax_bar = axs[row][2 * col + 1]
        
        ax_img.imshow(img)
        ax_img.axis('off')
        
        ax_bar.barh(y_pos, pred)
        ax_bar.set_yticks(y_pos, labels = class_names)
        
    total_axes = rows * cols * 2

    for j in range(2 * n, total_axes):
        row = j // (cols * 2)
        col = j % (cols * 2)
        axs[row][col].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    parser = argparse.ArgumentParser(description='Evaluate images using a trained model')
    parser.add_argument('model_path', type=str, help='Path to the trained model')
    parser.add_argument('image_path', type=str, help='Path to directory of images')
    args = parser.parse_args()

    
    model = load_model(args.model_path)
    model.to(device)
    
    dataset = ImageFolder(args.image_path, transform=test_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"Found_images: {len(dataset)}")
   
    original_images = []
    predictions = []

    for inputs, _ in dataloader:
        
        img_path = dataset.samples[len(original_images)][0]
        original_images.append(Image.open(img_path).convert('RGB'))
        
       
        with torch.no_grad():
            inputs = inputs.to(device)
            output = model(inputs)
            pred = F.softmax(output[0], dim=0)
            predictions.append(pred.cpu())

    
    plot_predictions(original_images, predictions, class_names)
