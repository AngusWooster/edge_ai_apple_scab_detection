import os
import numpy as np
import cv2
import math

def validation_results(images, detections, counter, out_dir, classes, colors, threshold=0.5):
    """
    Function to save validation results.
    :param images: All the images from the current batch.
    :param detections: All the detection results.
    :param counter: Step counter for saving with unique ID.
    """
    # IMG_MEAN = [0.485, 0.456, 0.406]
    # IMG_STD = [0.229, 0.224, 0.225]
    image_list = [] # List to store predicted images to return.
    for i, detection in enumerate(detections):
        image_c = images[i].clone()
        # image_c = denormalize(image_c, IMG_MEAN, IMG_STD)
        image_c = image_c.detach().cpu().numpy().astype(np.float32)
        image = np.transpose(image_c, (1, 2, 0))

        image = np.ascontiguousarray(image, dtype=np.float32)

        scores = detection['scores'].cpu().numpy()
        labels = detection['labels']
        bboxes = detection['boxes'].detach().cpu().numpy()
        boxes = bboxes[scores >= threshold].astype(np.int32)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Get all the predicited class names.
        pred_classes = [classes[i] for i in labels.cpu().numpy()]
        for j, box in enumerate(boxes):
            class_name = pred_classes[j]
            color = colors[classes.index(class_name)]
            cv2.rectangle(
                image,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                color, 2, lineType=cv2.LINE_AA
            )
            cv2.putText(image, class_name,
                    (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,
                    2, lineType=cv2.LINE_AA)
        if out_dir != None:
            cv2.imwrite(f"{out_dir}/image_{i}_{counter}.jpg", image*255.)

        image_list.append(image[:, :, ::-1])
    return image_list

def evaluation_dir_create(dir_name = "eval") -> str:
    """
    This functions counts the number of evaluation directories already present
    and creates a new one in `outputs/evaluation/`.
    And returns the directory path.
    """
    dir_path = 'outputs/evaluation' + '/' + dir_name
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    num_eval_dirs_present = len(os.listdir(dir_path))
    next_dir_num = num_eval_dirs_present + 1
    new_dir_path = f"{dir_path}/res_{next_dir_num}"
    os.makedirs(new_dir_path, exist_ok=True)
    return new_dir_path


def inference_dir_create() -> str:
    """
    This functions counts the number of evaluation directories already present
    and creates a new one in `outputs/inference/`.
    And returns the directory path.
    """
    dir_path = 'outputs/inference'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    num_eval_dirs_present = len(os.listdir(dir_path))
    next_dir_num = num_eval_dirs_present + 1
    new_dir_path = f"{dir_path}/res_{next_dir_num}"
    os.makedirs(new_dir_path, exist_ok=True)
    return new_dir_path