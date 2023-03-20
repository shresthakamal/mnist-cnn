import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# define a convolutional neural network
class ConvNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()

        # batch x 1 x 28 x 28 -> batch x 8 x 28 x 28
        self.conv1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        # batch x 8 x 28 x 28 -> batch x 8 x 14 x 14
        self.pool1 = torch.nn.MaxPool2d(
            kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)
        )

        # batch x 8 x 14 x 14 -> batch x 16 x 14 x 14
        self.conv2 = torch.nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        # batch x 16 x 14 x 14 -> batch x 16 x 7 x 7
        self.pool2 = torch.nn.MaxPool2d(
            kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)
        )

        self.linear1 = torch.nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = F.relu(out)
        out = self.pool2(out)

        # flatten
        logits = self.linear1(out.view(-1, 16 * 7 * 7))
        probas = F.softmax(logits, dim=1)

        return logits, probas


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    random_seed = 1
    learning_rate = 0.05
    num_epochs = 5
    batch_size = 128

    num_classes = 10

    # load MNIST dataset
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
    model = ConvNet(num_classes=num_classes).to(device)
    model = model.to(device)

    # print number of model trainiable parameters
    print(
        f"Trainable Prameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print("Training the model...")

    model.zero_grad()

    for epoch in range(num_epochs):
        model.train()

        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(device)
            targets = targets.to(device)

            # forward pass
            logits, probas = model(features)
            loss = F.cross_entropy(logits, targets)

            optimizer.zero_grad()

            # calculate gradients
            loss.backward()

            # update model parameters
            optimizer.step()

            ### LOGGING
            if not batch_idx % 50:
                print(
                    "Epoch: %03d/%03d | Batch %03d/%03d | loss: %.4f"
                    % (epoch + 1, num_epochs, batch_idx, len(train_loader), loss)
                )

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (features, targets) in enumerate(test_loader):
                features = features.to(device)
                targets = targets.to(device)

                logits, probas = model(features)

                _, predicted = torch.max(probas, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum()

            print("Test accuracy: %.2f %%" % (100 * correct / total))


if __name__ == "__main__":
    train()
