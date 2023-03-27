import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

batch_size = 64
num_epochs = 10
random_seed = 1
learning_rate = 0.01


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print(f"Time: {end - start}")

    return wrapper


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(MultiLayerPerceptron, self).__init__()
        self.linear_1 = torch.nn.Linear(num_features, 256)
        self.linear_2 = torch.nn.Linear(256, 128)
        self.linear_3 = torch.nn.Linear(128, 64)
        self.linear_4 = torch.nn.Linear(64, 32)
        self.linear_out = torch.nn.Linear(32, num_classes)

    def forward(self, x):
        out = F.relu(self.linear_1(x))
        out = F.relu(self.linear_2(out))
        out = F.relu(self.linear_3(out))
        out = F.relu(self.linear_4(out))
        logits = self.linear_out(out)
        probas = F.log_softmax(logits, dim=1)

        return logits, probas


@timer
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = datasets.MNIST(
        root="data", train=True, transform=transforms.ToTensor(), download=True
    )

    test_dataset = datasets.MNIST(
        root="data", train=False, transform=transforms.ToTensor()
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    torch.manual_seed(random_seed)

    model = MultiLayerPerceptron(num_features=784, num_classes=10)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()

        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.view(-1, 28 * 28).to(device)
            targets = targets.to(device)

            logits, _ = model(features)

            loss = F.cross_entropy(logits, targets)

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, norm_type=2)

            optimizer.step()

            if not batch_idx % 100:
                print(
                    f"Epoch: {epoch+1:03d}/{num_epochs:03d} | "
                    f"Batch {batch_idx:03d}/{len(train_loader):03d} | "
                    f"Loss: {loss:.4f}"
                )

        # save the model
        torch.save(model.state_dict(), "mlp_grad_clip.pt")

        model.eval()

        with torch.no_grad():
            correct_pred, num_examples = 0, 0

            for features, targets in test_loader:
                features = features.view(-1, 28 * 28).to(device)
                targets = targets.to(device)

                logits, probas = model(features)

                _, predicted_labels = torch.max(probas, 1)

                num_examples += targets.size(0)

                correct_pred += (predicted_labels == targets).sum()

            print(f"Accuracy: {correct_pred.float()/num_examples*100:.2f}%")


# python main function
if __name__ == "__main__":
    train()
