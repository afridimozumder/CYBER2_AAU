import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import numpy as np
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Define the SimpleCNN architecture (same as in your code)
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 5)
        self.fc1 = torch.nn.Linear(64 * 5 * 5, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the trained model
def load_model(model_path, device):
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# Define the transform for preprocessing the image
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to 32x32 (CIFAR-10 size)
    transforms.ToTensor(),         # Convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
])

# Load a single image
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Ensure 3 channels (RGB)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Function to analyze the image
def analyze_image():
    global image_tk, confidence_label, prediction_label, probability_label

    # Set the initial directory to the user's home directory (or any other preferred directory)
    initial_dir = os.path.expanduser("~")  # Defaults to the user's home directory
    # Alternatively, you can set it to the "Pictures" folder:
    # initial_dir = os.path.join(os.path.expanduser("~"), "Pictures")

    # Ask the user to select an image file
    file_path = filedialog.askopenfilename(
        initialdir=initial_dir,  # Set the initial directory
        title="Select an Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return

    try:
        # Load and preprocess the image
        image_tensor = load_image(file_path)

        # Get the confidence score for the image
        model.eval()
        with torch.no_grad():
            output = model(image_tensor.to(device))
            probabilities = torch.softmax(output, dim=1)
            confidence_score = probabilities.max().item()

        # Predict membership using the attack model
        confidence_score = np.array([confidence_score]).reshape(-1, 1)
        prediction = attack_model.predict(confidence_score)
        prediction_proba = attack_model.predict_proba(confidence_score)[:, 1]

        # Display the results
        confidence_label.config(text=f'Confidence Score: {confidence_score[0][0]:.4f}')
        prediction_label.config(text=f'Prediction: {"Member" if prediction[0] == 1 else "Non-Member"}')
        probability_label.config(text=f'Membership Probability: {prediction_proba[0]:.4f}')

        # Display the uploaded image
        image = Image.open(file_path)
        #image = image.resize((200, 200), Image.ANTIALIAS)
        image = image.resize((200, 200), Image.Resampling.LANCZOS)
        image_tk = ImageTk.PhotoImage(image)
        image_label.config(image=image_tk)
        image_label.image = image_tk

    except Exception as e:
        print(f"Error: {e}")

# Main function for the GUI
def main():
    global model, attack_model, device, image_label, confidence_label, prediction_label, probability_label

    # Set the device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the trained model
    model_path = 'cifar10_simplecnn.pth'
    model = load_model(model_path, device)

    # Load the attack model (logistic regression)
    attack_model_path = 'attack_model.pkl'
    attack_model = joblib.load(attack_model_path)

    # Create the main window
    root = tk.Tk()
    root.title("Membership Inference Attack on CIFAR-10 Model")

    # Create a button to upload an image
    upload_button = tk.Button(root, text="Upload Image", command=analyze_image)
    upload_button.pack(pady=10)

    # Create a label to display the uploaded image
    image_label = tk.Label(root)
    image_label.pack(pady=10)

    # Create labels to display the results
    confidence_label = tk.Label(root, text="Confidence Score: ")
    confidence_label.pack(pady=5)

    prediction_label = tk.Label(root, text="Prediction: ")
    prediction_label.pack(pady=5)

    probability_label = tk.Label(root, text="Membership Probability: ")
    probability_label.pack(pady=5)

    # Run the GUI
    root.mainloop()

# Run the GUI
if __name__ == "__main__":
    main()