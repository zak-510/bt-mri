import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision import models
from torchvision.models import ResNet18_Weights
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


data_transform = v2.Compose([
    v2.RandomResizedCrop((224,224)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


eval_transform = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = datasets.ImageFolder(root=r"C:/Users/celre/Downloads/dataset/Training", transform=data_transform)
val_size = int(0.2 * len(train_dataset))


train_subset, val_subset = torch.utils.data.random_split(
    train_dataset, [len(train_dataset) - val_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

test_dataset = datasets.ImageFolder(root=r"C:/Users/celre/Downloads/dataset/Testing", transform=eval_transform)


train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16*61*61, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )
   
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(x, 2))
        x = F.relu(self.conv2(x))
        x = F.relu(F.max_pool2d(x, 2))
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
   
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 4)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print(model)


loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True, min_lr=1e-6)


def train(dataloader, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)

    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss = loss.item()
            curr = batch * batch_size + len(X)
            print(f"loss: {loss:0.2f} [{curr:}/{size:0.2f}]")



def test(dataloader, model, loss_fn, batch_size):
    model.eval()
    size = len(dataloader.dataset)

    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg Loss: {test_loss:.2f} \n")

    return correct


best_acc = 0.0
epochs = 50
count = 0
for e in range(epochs):
    print(f"Epoch {e+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer, batch_size=32)
    acc = test(val_loader, model, loss_fn, batch_size=32)
    scheduler.step(acc)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"Best model saved with accuracy: {best_acc:.2f}")
        count = 0
    else:
        count += 1
        if count == 5:
            print("Early stopping. Model not improving.")
            break


model.load_state_dict(torch.load("best_model.pth"))
test_acc = test(test_loader, model, loss_fn, batch_size=32)


model.load_state_dict(torch.load("best_model.pth"))
model.eval()


y_true, y_pred = [], []
with torch.no_grad():
    for X, y in test_loader:
        preds = model(X.to(device)).argmax(1).cpu()
        y_true.extend(y.tolist())
        y_pred.extend(preds.tolist())


print(classification_report(y_true, y_pred, target_names=test_dataset.classes, digits=3))


ConfusionMatrixDisplay.from_predictions(
    y_true, y_pred, display_labels=test_dataset.classes, xticks_rotation=45, cmap="Blues"
)
plt.tight_layout()
plt.show()





