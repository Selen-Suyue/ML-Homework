import torch
import torchvision
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import torchvision.transforms as transforms
from tqdm import tqdm

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

train_data, train_labels = next(iter(train_loader))
test_data, test_labels = next(iter(test_loader))

train_data = train_data.numpy().reshape(train_data.shape[0], -1)
test_data = test_data.numpy().reshape(test_data.shape[0], -1)
train_labels = train_labels.numpy()
test_labels = test_labels.numpy()

# 2. Train SVM
svm_clf = SVC(kernel='sigmoid', C=10, gamma='scale')  

# Train on a subset for demonstration (adjust as needed)
svm_clf.fit(train_data[:1000], train_labels[:1000])  

y_pred = svm_clf.predict(test_data)
accuracy = accuracy_score(test_labels, y_pred)

print(f"Test accuracy: {accuracy * 100:.2f}%")