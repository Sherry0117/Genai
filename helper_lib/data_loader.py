import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loader(data_dir, batch_size=32, train=True, num_workers=2, pin_memory=True):
    """Create a DataLoader for images organized in ImageFolder format.

    Expected directory structure:
      data_dir/
        train/ class_x/ ..., class_y/ ...
        test/  class_x/ ..., class_y/ ...

    For training, light augmentation is applied; for evaluation, only resize/normalize.
    Images are resized to 224 so they are compatible with `resnet18` if used.
    """

    split_dir = "train" if train else "test"
    dataset_root = os.path.join(data_dir, split_dir)

    if train:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    dataset = datasets.ImageFolder(root=dataset_root, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
