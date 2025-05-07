import matplotlib
matplotlib.use('TkAgg')  # Set the backend to TkAgg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

def train_attack_model(member_confidences, non_member_confidences):
    # Create labels: 1 for members, 0 for non-members
    X = np.concatenate([member_confidences, non_member_confidences])
    y = np.concatenate([np.ones(len(member_confidences)), np.zeros(len(non_member_confidences))])

    # Split into attack train/test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train attack model
    attack_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    attack_model.fit(X_train, y_train)

    # Evaluate attack model
    y_pred = attack_model.predict(X_test)
    y_proba = attack_model.predict_proba(X_test)[:, 1]

    print(f'Attack Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print(f'AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}')

    # Visualize confidence score distributions
    plt.hist(member_confidences.max(axis=1), bins=50, alpha=0.5, label='Members')
    plt.hist(non_member_confidences.max(axis=1), bins=50, alpha=0.5, label='Non-Members')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Confidence Scores')
    plt.legend()
    plt.show()