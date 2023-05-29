import glob
import os
import torch
import cv2
import numpy as np
from xml.etree import ElementTree as et
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms

from utils.transform import *

# the dataset class
class CustomPascalVocDataset(Dataset):
    def __init__(
        self, images_path, labels_path,
        width, height, classes, transforms=None,
        use_train_aug=False,
        train=False, mosaic=False
    ):
        self.transforms = transforms
        self.use_train_aug = use_train_aug
        self.images_path = images_path
        self.labels_path = labels_path
        self.height = height
        self.width = width
        self.classes = classes
        self.train = train
        self.image_file_types = ['*.jpg', '*.jpeg', '*.JPG']
        self.all_image_paths = []

        # get all the image paths in sorted order
        for file_type in self.image_file_types:
            self.all_image_paths.extend(glob.glob(os.path.join(self.images_path, file_type)))
        self.all_annot_paths = glob.glob(os.path.join(self.labels_path, '*.xml'))
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.all_image_paths]
        self.all_images = sorted(self.all_images)
        # Remove all annotations and images when no object is present.
        self.read_and_clean()

    def read_and_clean(self):
        # Discard any images and labels when the XML
        # file does not contain any object.
        for annot_path in self.all_annot_paths:
            tree = et.parse(annot_path)
            root = tree.getroot()
            object_present = False
            for member in root.findall('object'):
                if member.find('bndbox'):
                    object_present = True
            if object_present == False:
                image_name = annot_path.split(os.path.sep)[-1].split('.xml')[0]
                image_root = self.all_image_paths[0].split(os.path.sep)[:-1]
                # remove_image = f"{'/'.join(image_root)}/{image_name}.jpg"
                for image_type in self.image_file_types:
                    ext = image_type.split('*')[-1]
                    try:
                        remove_image = os.path.join(os.sep.join(image_root), image_name+ext)
                    except:
                        pass
                print(f"Removing {annot_path} and corresponding {remove_image}")
                self.all_annot_paths.remove(annot_path)
                self.all_image_paths.remove(remove_image)

        # Discard any image file when no annotation file
        # is not found for the image.
        for image_name in self.all_images:
            for image_type in self.image_file_types:
                ext = image_type.split('*')[-1]
                # Only compare if the extension of files are belong image file type
                if image_name.split(".")[-1] == ext.split(".")[-1]:
                    try:
                        possible_xml_name = os.path.join(self.labels_path, image_name.split(ext)[0]+'.xml')
                    except:
                        pass
                    if possible_xml_name not in self.all_annot_paths:
                        print(f"{possible_xml_name} not found...")
                        print(f"Removing {image_name} image")
                        self.all_images = [image_instance for image_instance in self.all_images if image_instance != image_name]
                        # self.all_images.remove(image_name)
                    break
    def check_image_and_annotation(self, xmax, ymax, width, height):
        """
        Check that all x_max and y_max are not more than the image
        width or height.
        """
        if ymax > height:
            ymax = height
        if xmax > width:
            xmax = width
        return ymax, xmax

    def load_image_and_labels(self, index):
        image_name = self.all_images[index]
        image_path = os.path.join(self.images_path, image_name)
        # Read the image.
        image = cv2.imread(image_path)
        # Convert BGR to RGB color format.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0
        # Capture the corresponding XML file for getting the annotations.
        annot_filename = image_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.labels_path, annot_filename)

        boxes = []
        orig_boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        # Get the height and width of the original image.
        image_width = image.shape[1]
        image_height = image.shape[0]
        # Box coordinates for xml files are extracted and corrected for image size given.
        for member in root.findall('object'):
            # Map the current object name to `classes` list to get
            # the label index and append to `labels` list.
            labels.append(self.classes.index(member.find('name').text))

            # xmin = left corner x-coordinates
            xmin = int(member.find('bndbox').find('xmin').text)
            # xmax = right corner x-coordinates
            xmax = int(member.find('bndbox').find('xmax').text)
            # ymin = left corner y-coordinates
            ymin = int(member.find('bndbox').find('ymin').text)
            # ymax = right corner y-coordinates
            ymax = int(member.find('bndbox').find('ymax').text)

            ymax, xmax = self.check_image_and_annotation(
                xmax, ymax, image_width, image_height
            )

            orig_boxes.append([xmin, ymin, xmax, ymax])

            # Resize the bounding boxes according to the
            # desired `width`, `height`.
            xmin_final = (xmin/image_width)*self.width
            xmax_final = (xmax/image_width)*self.width
            ymin_final = (ymin/image_height)*self.height
            ymax_final = (ymax/image_height)*self.height

            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

        # Bounding box to tensor.
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Area of the bounding boxes.
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # No crowd instances.
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # Labels to tensor.
        labels = torch.as_tensor(labels, dtype=torch.int64)
        return image, image_resized, orig_boxes, \
               boxes, labels, area, iscrowd, (image_width, image_height)

    def __getitem__(self, idx):
        # Capture the image name and the full image path.
        image, image_resized, orig_boxes, boxes, \
        labels, area, iscrowd, dims = self.load_image_and_labels(index=idx)
        # Prepare the final `target` dictionary.
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        if False: #self.use_train_aug: # Use train augmentation if argument is passed.
            train_aug = get_train_aug()
            sample = train_aug(image=image_resized,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
        else:
            sample = self.transforms(image=image_resized,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])


        return image_resized, target

    def __len__(self):
        return len(self.all_images)


def eval_dataset_create(valid_dir_images, valid_dir_labels,
                         resize_width, resize_height, classes):
    valid_dataset = CustomPascalVocDataset(valid_dir_images,
                                           valid_dir_labels,
                                           resize_width,
                                           resize_height,
                                           classes,
                                           eval_transform(),
                                           train=False)
    return valid_dataset



def _collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

def eval_loader_create(valid_dataset, batch_size, num_workers=0):
    valid_loader = DataLoader(valid_dataset,
                              batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              collate_fn=_collate_fn)
    return valid_loader