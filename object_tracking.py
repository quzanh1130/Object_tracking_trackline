import cv2
import torch
import numpy as np
import random
import sys
import os
import argparse
from pathlib import Path
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape
from utils.torch_utils import select_device, smart_inference_mode
from utils.dataloaders import IMG_FORMATS, VID_FORMATS

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

@smart_inference_mode()
def run(
        weights=ROOT / 'weight\yolov9-c-converted.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        mode='0'  # mode show line or not. 0: not show, 1: show line
):

    conf_threshold = 0.5
    tracking_class = 2 # None: track all

    # Khởi tạo DeepSort
    tracker = DeepSort(max_age=30)

    device = select_device(device)
    model  = DetectMultiBackend(weights=weights, device=device, fuse=True )
    model  = AutoShape(model)
    
    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)

    # Load classname từ file classes.names
    with open("data_ext/classes.names") as f:
        class_names = f.read().strip().split('\n')

    colors = np.random.randint(0,255, size=(len(class_names),3 ))
    tracks = []
    
    # Mở video từ webcam hoặc file
    if webcam:
        source = int(source) if source.isnumeric() else source
        cap = cv2.VideoCapture(source)
        assert cap.isOpened(), f'Failed to open {source}'
    else:
        cap = cv2.VideoCapture(source)
        assert cap.isOpened(), f'Failed to open {source}'

    # Initialize the dictionaries before the while loop
    first_locations = {}
    last_known_locations = {}
    frames_since_seen = {}
    track_colors = {}

    # Start of the while loop
    while True:
        # Đọc
        ret, frame = cap.read()
        if not ret:
            continue
        # Đưa qua model để detect
        results = model(frame)

        detect = []
        for detect_object in results.pred[0]:
            label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]
            x1, y1, x2, y2 = map(int, bbox)
            class_id = int(label)

            if tracking_class is None:
                if confidence < conf_threshold:
                    continue
            else:
                if class_id != tracking_class or confidence < conf_threshold:
                    continue

            detect.append([ [x1, y1, x2-x1, y2 - y1], confidence, class_id ])


        # Cập nhật,gán ID băằng DeepSort
        tracks = tracker.update_tracks(detect, frame = frame)

        if mode == '0':
            # Vẽ lên màn hình các khung chữ nhật kèm ID
            for track in tracks:
                if track.is_confirmed():
                    track_id = track.track_id
                    # Lấy toạ độ, class_id để vẽ lên hình ảnh
                    ltrb = track.to_ltrb()
                    class_id = track.get_det_class()
                    x1, y1, x2, y2 = map(int, ltrb)
                    color = colors[class_id]
                    B, G, R = map(int,color)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
                    # Draw object ID on the image
                    label = f"ID: {track_id}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
        elif mode == '1':
            for track in tracks:
                if track.is_confirmed():
                    track_id = track.track_id

                    # Get coordinates, class_id to draw on the image
                    ltrb = track.to_ltrb()
                    class_id = track.get_det_class()
                    x1, y1, x2, y2 = map(int, ltrb)

                    # Draw bounding box
                    color = colors[class_id]
                    B, G, R = map(int,color)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
                    # Draw object ID on the image
                    label = f"ID: {track_id}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # Calculate the center of the bounding box
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    current_location = (center_x, center_y)

                    # Store the first location of the object
                    if track_id not in first_locations:
                        first_locations[track_id] = current_location
                        track_colors[track_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                    # Update the last known location of the object
                    last_known_locations[track_id] = current_location
                    frames_since_seen[track_id] = 0  # Reset the counter for this track_id

            # For each track, draw a line from the first location to the last known location
            for track_id, last_known_location in last_known_locations.items():
                if frames_since_seen.get(track_id, 0) < 3:  # If the object has been seen in the last 5 frames
                    first_location = first_locations[track_id]
                    color = track_colors[track_id]
                    cv2.line(frame, first_location, last_known_location, color, 2)

            # Increment the counter for each track that was not seen in this frame
            for track_id in frames_since_seen.keys():
                frames_since_seen[track_id] += 1

        # Show the image on the screen
        cv2.imshow("OT", frame)
        # Press Q to quit
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weight\yolov9-c-converted.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data_ext/test.mp4', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--mode', default='0', help='mode show line or not. 0: not show, 1: show line')
    opt = parser.parse_args()
    opt.mode = '0' if str(opt.mode) != '1' or str(opt.mode) != '0' else str(opt.mode)  # expand
    return opt


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
