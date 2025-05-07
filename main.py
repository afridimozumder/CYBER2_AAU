import torch  # Add this line to import the torch module
from train_target import train_model, load_model, evaluate_mia
from data_loader import load_cifar10

if __name__ == "__main__":
    # Load the dataset
    trainloader, valloader, testloader = load_cifar10()

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Step 1: Train and save the model (only do this once)
    # Uncomment the following lines to train and save the model:
    # model = train_model(trainloader, valloader, epochs=100)

    # Step 2: Load the trained model
    model = load_model('cifar10_simplecnn.pth', device)

    # Step 3: Evaluate Membership Inference Attack based on confidence scores
    evaluate_mia(model, trainloader, testloader, device)