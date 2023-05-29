
import argparse
import yaml
import glob
import os
from tqdm import tqdm
import time

def parse_opt():
    # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None,
                        help='path to the data config file')
    parser.add_argument('--dataset', default=None,
                        help='path to the data config file')

    args = vars(parser.parse_args())
    return args


def dataset_image_rename(prefix:str, file_types:list, dataset_path:str):
    images = []
    print(f"dataset_path: {dataset_path}")
    for file_type in file_types:
        images.extend(glob.glob(f"{dataset_path}/{file_type}"))

    for i, image in enumerate(tqdm(images)):
        new_image = f"{os.path.dirname(image)}/{prefix}_{i+1:04d}.{image.split('.')[-1]}"
        print(f'{image} -> {new_image}')
        os.rename(image, new_image)
        # #
        # #   change xml file name
        # #
        # xml = image.rsplit( ".", 1 )[ 0 ] + '.xml'
        # new_xml = new_image.rsplit( ".", 1 )[ 0 ] + '.xml'
        # print(f'{xml} ->> {new_xml}')
        # os.rename(xml, new_xml)


def main(args):
    with open(args['config']) as file:
        data_configs = yaml.safe_load(file)

    dataset_num = data_configs["DATASET"][args['dataset']]['NUM']
    train_dir = os.path.join('..', data_configs["DATASET"][args['dataset']]['TRAIN_DIR_IMAGES'])
    valid_dir = os.path.join('..',data_configs["DATASET"][args['dataset']]['VALID_DIR_IMAGES'])
    print(f"dataset_num:{dataset_num}")
    print(f"train_dir:{train_dir}")
    print(f"valid_dir:{valid_dir}")

    timestr = time.strftime("%m%d-%H%M%S")
    train_prefix = f"IMG_t_{int(dataset_num):02d}_{timestr}"
    valid_prefix = f"IMG_v_{int(dataset_num):02d}_{timestr}"

    file_types = ['*.jpg', '*.jpeg', '*.JPG']
    dataset_image_rename(train_prefix, file_types, train_dir)
    dataset_image_rename(valid_prefix, file_types, valid_dir)


if __name__ == '__main__':
    args = parse_opt()
    main(args)