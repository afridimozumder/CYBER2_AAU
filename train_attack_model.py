import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from data_loader import load_cifar10
from train_target import load_model

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('cifar10_simplecnn.pth', device)

# Load CIFAR-10 dataset
trainloader, _, testloader = load_cifar10()

# Function to extract confidence scores
def get_confidence_scores(dataloader):
    confidences = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            confidences.extend(probabilities.max(dim=1).values.cpu().numpy())
    return np.array(confidences)

# Get confidence scores for members (training data) and non-members (test data)
member_confidences = get_confidence_scores(trainloader)
non_member_confidences = get_confidence_scores(testloader)

# Create labels: 1 for members, 0 for non-members
X = np.concatenate([member_confidences, non_member_confidences]).reshape(-1, 1)
y = np.concatenate([np.ones(len(member_confidences)), np.zeros(len(non_member_confidences))])

# Split into train/test sets for the attack model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the attack model (logistic regression)
attack_model = LogisticRegression(solver='lbfgs', max_iter=1000)
attack_model.fit(X_train, y_train)

# Evaluate the attack model
from sklearn.metrics import accuracy_score, roc_auc_score
y_pred = attack_model.predict(X_test)
y_proba = attack_model.predict_proba(X_test)[:, 1]
print(f'Attack Model Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(f'Attack Model AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}')

# Save the attack model
joblib.dump(attack_model, 'attack_model.pkl')
print("Attack model saved to 'attack_model.pkl'")