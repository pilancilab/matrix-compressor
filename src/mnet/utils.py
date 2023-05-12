import torch

# Function to extract embeddings from the dataset using MobileNet
def extract_embeddings(dataloader, model):
    from tqdm import tqdm
    embeddings = []
    labels = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            images, targets = data
            features = model(images)
            features = features.view(features.size(0), -1)
            embeddings.append(features)
            labels.append(targets)
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    return embeddings, labels
