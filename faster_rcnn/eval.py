import torch
import argparse
import yaml
import numpy as np
from datasets import (eval_dataset_create, eval_loader_create)
from utils.general import (evaluation_dir_create, validation_results)
from utils.logging import (coco_log)
from utils_pytorch.engine import (evaluate)
from models.models_create import (create_model)

def parse_opt():
    # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None,
                        help='path to the data config file')

    parser.add_argument('-m', '--model', default='fasterrcnn_resnet50_fpn_v2',
                        help='name of the model')

    parser.add_argument('--eval_dataset', default='AppleScabFDs',
                        help='choose a dataset for evaluation')

    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='number of workers for data processing/transforms/augmentations')

    parser.add_argument('-d', '--device',
                        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                        help='computation/training device, default is GPU if GPU present')

    parser.add_argument('-w', '--weights', default=None, type=str,
                        help='path to model weights if using pretrained weights')

    parser.add_argument('-b', '--batch-size', dest='batch_size', default=1, type=int,
                        help='batch size to load the data')

    parser.add_argument('-ims', '--img-size', dest='img_size', default=640, type=int,
                        help='image size to feed to the network')


    print(f"parser.parse_args() = {parser.parse_args()}")
    args = vars(parser.parse_args())
    return args

def main(args):
    print(args['config'])
    with open(args['config']) as file:
        data_configs = yaml.safe_load(file)


    VALID_DIR_IMAGES = data_configs['DATASET'][args['eval_dataset']]['VALID_DIR_IMAGES']
    VALID_DIR_LABELS = data_configs['DATASET'][args['eval_dataset']]['VALID_DIR_LABELS']
    print(f"VALID_DIR_IMAGES = {VALID_DIR_IMAGES} / VALID_DIR_LABELS = {VALID_DIR_LABELS}")

    CLASSES = data_configs['CLASSES']
    DEVICE = args['device']
    BATCH_SIZE = args['batch_size']
    IMAGE_WIDTH = args['img_size']
    IMAGE_HEIGHT = args['img_size']
    NUM_WORKERS = args['workers']
    COLORS = np.random.uniform(0, 1, size=(len(CLASSES), 3))

    eval_dataset = eval_dataset_create(VALID_DIR_IMAGES, VALID_DIR_LABELS, IMAGE_WIDTH, IMAGE_HEIGHT, CLASSES)
    eval_loader = eval_loader_create(eval_dataset, BATCH_SIZE, NUM_WORKERS)
    # Build the new model with number of classes
    build_model = create_model[args['model']]
    model = build_model(num_classes=2)
    # Load weights.
    checkpoint = torch.load(args['weights'], map_location=DEVICE)
    ckpt_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(ckpt_state_dict)
    model = model.to(DEVICE)
    model.eval()
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")

    eval_dir_name = args['eval_dataset'] + '/' + (args['weights'].split('/')[-1]).split('.')[0]
    print(f'eval dir: {eval_dir_name}')
    valuation_dir_path = evaluation_dir_create(eval_dir_name)

    #####################################
    #   testing eval loader
    #####################################
    # print(f"{eval_loader.dataset}")
    # for i, (img, targets) in enumerate(eval_loader):
    #     print(f'num:{i} {targets}, img size = {len(img)}')
    #     img = list(img.to(DEVICE) for img in img)
    #     with torch.inference_mode():
    #         res = model(img)
    #         print(f"res = {res}")
    #     out_img = validation_results(img, res, i, valuation_dir_path, CLASSES, COLORS)
    #     print("*"*50)
    #     break



    stats, inference_times = evaluate(model,
                                      eval_loader,
                                      device=DEVICE,
                                      save_valid_preds=True,
                                      out_dir=valuation_dir_path,
                                      classes=CLASSES,
                                      colors=COLORS)

    coco_log(valuation_dir_path, stats, 'eval_log', inference_times)

if __name__ == '__main__':
    args = parse_opt()
    main(args)


'''
executed commands:

python3 eval.py \
--config data_configs/apple_scab.yaml \
--weights outputs/training/AppleScabFDs/best_model.pth \
--eval_dataset final_dataset --device cpu

python3 eval.py \
--config data_configs/apple_scab.yaml \
--weights outputs/training/final_dataset/best_model.pth \
--batch-size 4 \
--eval_dataset final_dataset
'''