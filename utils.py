import torch
import torch.nn.functional as F
import numpy as np

def get_confidence_scores(model, dataloader):
    model.eval()
    confidences = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            confidences.extend(probabilities.numpy())
    return np.array(confidences)