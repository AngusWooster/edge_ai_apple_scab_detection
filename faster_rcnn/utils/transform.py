from torchvision import transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define the training transforms
def training_augmentation():
    pass


def eval_transform():
    transform = A.Compose([ToTensorV2(p=1.0)],
                          bbox_params={'format': 'pascal_voc',
                                       'label_fields': ['labels']})
    return transform

def infer_transforms(image):
    # Define the torchvision image transforms.
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    return transform(image)