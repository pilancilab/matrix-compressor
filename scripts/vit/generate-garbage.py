import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


# Define the transform to preprocess the images
transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load the CIFAR-10 dataset
train_set = torchvision.datasets.CIFAR10(
    root="/home/rsaha/varun/matrix-compressor/datasets",
    train=True,
    download=True,
    transform=transform,
)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=64, shuffle=False, num_workers=2
)

# Load the pre-trained Vision Transformer from TorchVision
model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

# Remove the last classification layer of the model
model.classifier = nn.Identity()

# Set the model to evaluation mode
model.eval()


# Define a function to generate embeddings from the model
def get_embeddings(loader, model):
    embeddings = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs, labels = data
            outputs = model(inputs)
            embeddings.append(outputs)
    return torch.cat(embeddings, dim=0)


# Generate embeddings for the test set using the pre-trained model
train_embeddings = get_embeddings(train_loader, model)
torch.save(
    f"artifacts/custom_data/knn/vit_b_16/cifar10/cifar10-vit_small_path16_224-train-embeddings.pt"
)
