from models import *

def fasterrcnn_resnet50_fpn_v2_create(num_classes, pretrained=True, coco_model=False):
    model = fasterrcnn_resnet50_fpn_v2.create_model(
        num_classes, pretrained=pretrained, coco_model=coco_model
    )
    return model

def fasterrcnn_resnet50_fpn_create(num_classes, pretrained=True, coco_model=False):
    model = fasterrcnn_resnet50_fpn.create_model(
        num_classes, pretrained=pretrained, coco_model=coco_model
    )
    return model

create_model = {
    'fasterrcnn_resnet50_fpn': fasterrcnn_resnet50_fpn_create,
    'fasterrcnn_resnet50_fpn_v2': fasterrcnn_resnet50_fpn_v2_create
}