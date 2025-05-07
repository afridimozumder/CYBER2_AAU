import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Check if GPU is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the SimpleCNN architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)  # Increase filters to 32
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)  # Increase filters to 64
        self.fc1 = nn.Linear(64 * 5 * 5, 256)  # Increase neurons to 256
        self.fc2 = nn.Linear(256, 128)  # Add another fully connected layer
        self.fc3 = nn.Linear(128, 10)  # Output layer (10 classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(nn.functional.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = x.view(-1, 64 * 5 * 5)  # Flatten the feature maps
        x = nn.functional.relu(self.fc1(x))  # FC1 -> ReLU
        x = nn.functional.relu(self.fc2(x))  # FC2 -> ReLU
        x = self.fc3(x)  # Output layer
        return x

# Define loss function
criterion = nn.CrossEntropyLoss()

# Function to train the model
def train_model(trainloader, valloader, epochs=100):  # Train for 100 epochs
    model = SimpleCNN()
    model = model.to(device)

    # Remove weight decay to encourage overfitting
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'cifar10_simplecnn.pth')
    print("Model saved to 'cifar10_simplecnn.pth'")

    return model

# Function to load the trained model
def load_model(model_path, device):
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# Function to perform inference and evaluate MIA based on confidence scores
def evaluate_mia(model, trainloader, testloader, device):
    model.eval()  # Set the model to evaluation mode

    # Get confidence scores for members (training data) and non-members (test data)
    def get_confidence_scores(dataloader):
        confidences = []
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                confidences.extend(probabilities.max(dim=1).values.cpu().numpy())  # Max confidence score
        return np.array(confidences)

    member_confidences = get_confidence_scores(trainloader)
    non_member_confidences = get_confidence_scores(testloader)

    # Create labels: 1 for members, 0 for non-members
    X = np.concatenate([member_confidences, non_member_confidences])
    y = np.concatenate([np.ones(len(member_confidences)), np.zeros(len(non_member_confidences))])

    # Split into train/test sets for the attack model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple attack model (logistic regression)
    from sklearn.linear_model import LogisticRegression
    attack_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    attack_model.fit(X_train.reshape(-1, 1), y_train)  # Reshape for single feature

    # Evaluate the attack model
    y_pred = attack_model.predict(X_test.reshape(-1, 1))
    y_proba = attack_model.predict_proba(X_test.reshape(-1, 1))[:, 1]

    # Calculate accuracy and AUC-ROC
    attack_accuracy = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    print(f'Attack Accuracy: {attack_accuracy:.4f}')
    print(f'AUC-ROC: {roc_auc:.4f}')

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Membership Inference Attack')
    plt.legend(loc="lower right")
    plt.show()

    # Plot confidence score distributions
    plt.figure()
    plt.hist(member_confidences, bins=50, alpha=0.5, label='Members')
    plt.hist(non_member_confidences, bins=50, alpha=0.5, label='Non-Members')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Confidence Scores')
    plt.legend()
    plt.show()