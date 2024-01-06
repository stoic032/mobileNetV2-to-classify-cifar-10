import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# dataset_path = "./dataset/cifar-10-batches-py"
# dict = unpickle("./dataset/cifar-10-batches-py/data_batch_1")



# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the dataset
train_dataset = datasets.CIFAR10(root='./dataset', train=True, download=False, transform=transform)
test_dataset = datasets.CIFAR10(root='./dataset', train=False, download=False, transform=transform)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

if __name__ == '__main__':
    classes = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    (data, label) = train_dataset[100]

    print(classes[label], "\t", data.shape)
    img = ((data + 1) / 2)
    img = img.permute(1,2,0).contiguous()

    import matplotlib.pyplot as plt
    plt.imshow(img.numpy(), cmap='gray')
    plt.title(classes[label])
    plt.show()
