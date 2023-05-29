import torch
import time
from torch.utils.data import Dataset, DataLoader
from utils.general import (validation_results)
from utils_pytorch.coco_utils import (convert_to_coco_api)
from utils_pytorch.coco_eval import (CocoEvaluator)


@torch.inference_mode()
def evaluate(model,
             data_loader:DataLoader,
             device,
             save_valid_preds=False,
             out_dir=None,
             classes=None,
             colors=None):
    model.eval()
    coco = convert_to_coco_api(data_loader.dataset)
    iou_types = ["bbox"]

    coco_evaluator = CocoEvaluator(coco, iou_types)
    '''
    output format of data loader:
        images: (tensor([[[2.2434e-01, 2.1739e-01, 2.2312e-01,  ..., 1.6339e-01, 2.2247e-01, 2.6782e-01],
                            ....
                        ]]))
        targets: ({'boxes': tensor([[192.4267, 283.0400, 323.2000, 345.7600],
                                    [193.2800, 364.4800, 235.0934, 390.2400],
                                    [496.6400, 139.0400, 541.2267, 164.1600]]),
                'labels': tensor([1, 1, 1]),
                'area': tensor([8202.1045, 1077.1107, 1120.0176]),
                'image_id': tensor([0])},)
    '''
    inference_times = []
    cnt = 0
    for images, targets in data_loader:
        cnt += 1
        images = list(img.to(device) for img in images)

        model_time = time.time()
        outputs = model(images)
        model_time = time.time() - model_time
        outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
        print(f"num of images: {len(images)}\tmodel time: {model_time}")
        inference_times.append(model_time)

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time

        if save_valid_preds:
            validation_results(images, outputs, cnt, out_dir, classes, colors)

    coco_evaluator.accumulate()
    stats = coco_evaluator.summarize()
    print(f"final stats = {stats}")
    return stats, inference_times