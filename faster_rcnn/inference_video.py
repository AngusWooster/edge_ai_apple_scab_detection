import argparse
import numpy as np
import yaml
import os
import cv2
import torch
import time

from utils.general import inference_dir_create
from utils.annotations import inference_annotations, annotate_fps
from utils.transform import infer_transforms
from models.models_create import (create_model)


def video_data_info_get(video_path):
    print(f"video_path = {video_path} type = {type(video_path)}")
    assert (video_path != None), 'Please check video path...'

    if video_path.isnumeric() == True:
        video_path = int(video_path)

    cap = cv2.VideoCapture(video_path)
    # Get the video's frame width and height
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    assert (frame_width != 0 and frame_height !=0), 'Please check video path...'
    return cap, frame_width, frame_height


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default=None,
                        help='path to input video')

    parser.add_argument('-c', '--config', default=None,
                        help='path to the data config file')

    # parser.add_argument('-m', '--model', default='fasterrcnn_resnet50_fpn_v2',
    #                     help='name of the model')

    parser.add_argument('-d', '--device',
                        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                        help='computation/training device, default is GPU if GPU present')

    parser.add_argument('-w', '--weights', default=None, type=str,
                        help='path to model weights if using pretrained weights')

    parser.add_argument('-th', '--threshold', default=0.3, type=float,
                        help='detection threshold')

    parser.add_argument('--save_video', default=False, type=bool, action=argparse.BooleanOptionalAction,
                        help='saving inference video steam')

    parser.add_argument('--show_img', default=False, type=bool, action=argparse.BooleanOptionalAction,
                        help='show inference video steam')

    parser.add_argument('--print_dbg_msg', default=False, type=bool, action=argparse.BooleanOptionalAction,
                        help='show inference video steam')

    args = vars(parser.parse_args())
    return args

def dbg_print(msg:str):
    if args['print_dbg_msg']:
        print(msg)

def main(args):
    # For same annotation colors each time.
    np.random.seed(42)

    if args['weights'] == None or args['input'] == None:
        print(f"invalid arguments")
        return

    # data_configs = None
    # if args['config'] is not None:
    #     with open(args['config']) as file:
    #         data_configs = yaml.safe_load(file)


    INPUT_VIDEO_SRC = args['input']
    DEVICE = args['device']
    SAVE_VIDEO = args['save_video']

    # load weights
    checkpoint = torch.load(args['weights'], map_location=DEVICE)
    CLASSES = checkpoint['config']['CLASSES']
    print(f"checkpoint['model_name'] = {checkpoint['model_name']}")
    # build model
    build_model = create_model[checkpoint['model_name']]
    model = build_model(num_classes=checkpoint['config']['NC'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    # Define the detection threshold any detection having
    # score below this will be discarded.
    detection_threshold = args['threshold']

    cap, frame_width, frame_height = video_data_info_get(INPUT_VIDEO_SRC)
    print(f"video frame: W = {frame_width}, H = {frame_height}")


    video_stream_out = None
    if SAVE_VIDEO:
        OUT_DIR = inference_dir_create()
        save_name = INPUT_VIDEO_SRC.split(os.path.sep)[-1].split('.')[0]
        video_stream_out = cv2.VideoWriter(f"{OUT_DIR}/{save_name}.mp4",
                        cv2.VideoWriter_fourcc(*'mp4v'), 30,
                        (frame_width, frame_height))
        print(f"save name: {save_name}")

    RESIZE_TO = (frame_width, frame_height)
    frame_count = 0 # To count total frames.
    total_fps = 0 # To get the final frames per second.

    # read until end of video
    while(cap.isOpened()):
        # capture each frame of the video
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, RESIZE_TO)
            image = frame.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = infer_transforms(image)
            # Add batch dimension.
            image = torch.unsqueeze(image, 0)
            # Get the start time.
            start_time = time.time()
            with torch.no_grad():
                # Get predictions for the current frame.
                outputs = model(image.to(DEVICE))
            forward_end_time = time.time()

            forward_pass_time = forward_end_time - start_time

            dbg_print("fwd: {forward_pass_time}")
            # Get the current fps.
            fps = 1 / (forward_pass_time)
            # Add `fps` to `total_fps`.
            total_fps += fps
            # Increment frame count.
            frame_count += 1

            # Load all detection to CPU for further operations.
            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
            # Carry further only if there are detected boxes.
            if len(outputs[0]['boxes']) != 0:
                frame = inference_annotations(
                    outputs, detection_threshold, CLASSES,
                    COLORS, frame
                )
            frame = annotate_fps(frame, fps)

            final_end_time = time.time()
            forward_and_annot_time = final_end_time - start_time
            print_string = f"Frame: {frame_count}, Forward pass FPS: {fps:.3f}, "
            print_string += f"Forward pass time: {forward_pass_time:.3f} seconds, "
            print_string += f"Forward pass + annotation time: {forward_and_annot_time:.3f} seconds"
            dbg_print(print_string)

            if SAVE_VIDEO:
                video_stream_out.write(frame)

            if args['show_img'] == True:
                cv2.imshow('Prediction', frame)
                # Press `q` to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        else:
            break

    # Release VideoCapture().
    cap.release()
    # Close all frames and video windows.
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = parse_opt()
    main(args)

'''
python3 inference_video.py \
--weights outputs/training/final_dataset/best_model.pth \
--input 2 \
--show_img --print_dbg_msg


python3 inference_video.py \
--weights outputs/training/final_dataset/best_model.pth \
--input 0 \
--device cpu \
--show_img --print_dbg_msg

'''