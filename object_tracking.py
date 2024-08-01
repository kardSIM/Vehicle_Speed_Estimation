import cv2
import torch
import numpy as np
import argparse, os
from deep_sort_realtime.deepsort_tracker import DeepSort
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "yolov9"))
from yolov9.models.common import DetectMultiBackend, AutoShape



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=str,
        nargs="?",
        default="content/highway.mp4",
        help="Path to input video"
    )
    parser.add_argument(
        "--output",
        type=str,
        nargs="?",
        help="path to output video",
        default="content/output.mp4"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.50,
        help="confidence threshold",
    )
    parser.add_argument(
        "--blur_id",
        type=int,
        default=None,
        help="class ID to apply Gaussian Blur",
    )
    parser.add_argument(
        "--class_id",
        type=int,
        default=None,
        help="class ID to track",
    )
    parser.add_argument(
        "--click",
        type=bool,
        default=False,
        help="Use mouse to define polygone",
    )
    opt = parser.parse_args()
    return opt



def draw_corner_rect(img, bbox, line_length=30, line_thickness=5, rect_thickness=1,
                     rect_color=(255, 0, 255), line_color=(0, 255, 0)):
    x, y, w, h = bbox
    x1, y1 = x + w, y + h

    if rect_thickness != 0:
        cv2.rectangle(img, bbox, rect_color, rect_thickness)

    # Top Left  x, y
    cv2.line(img, (x, y), (x + line_length, y), line_color, line_thickness)
    cv2.line(img, (x, y), (x, y + line_length), line_color, line_thickness)

    # Top Right  x1, y
    cv2.line(img, (x1, y), (x1 - line_length, y), line_color, line_thickness)
    cv2.line(img, (x1, y), (x1, y + line_length), line_color, line_thickness)

    # Bottom Left  x, y1
    cv2.line(img, (x, y1), (x + line_length, y1), line_color, line_thickness)
    cv2.line(img, (x, y1), (x, y1 - line_length), line_color, line_thickness)

    # Bottom Right  x1, y1
    cv2.line(img, (x1, y1), (x1 - line_length, y1), line_color, line_thickness)
    cv2.line(img, (x1, y1), (x1, y1 - line_length), line_color, line_thickness)

    return img  

def calculate_speed(distance, fps):
    return (distance *fps)*3.6


def calculate_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def read_frames(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame 

def get_clicked_points(frame, num_points=4):
    img = frame.copy()
    cv2.namedWindow("image")
    cv2.imshow("image", img)

    clicked_points = []
    keep_looping = True

    def mouse_callback(event, x, y, flags, param):
        nonlocal clicked_points, keep_looping, img

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(clicked_points) == num_points:
                clicked_points = []  # Reset if already captured 4 points
            clicked_points.append([x, y])
            cv2.circle(img, (x, y), 10, (0, 0, 255), -1)
            cv2.imshow("image", img)         

            if len(clicked_points) == num_points:
                keep_looping = False

    cv2.setMouseCallback("image", mouse_callback)

    while keep_looping:
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    return clicked_points, img if len(clicked_points) == num_points else None


def get_height_width_input(frame):
    img = frame.copy()

    while True:
        height = input("Enter the height: ")
        width = input("Enter the width: ")

        try:
            height = int(height)
            width = int(width)
            break
        except ValueError:
            print("Invalid input. Please enter integers for height and width.")


    return height, width




def main(_argv):
    FRAME_WIDTH=30
    FRAME_HEIGHT=100

    SOURCE_POLYGONE = np.array([[18, 550], [1852, 608],[1335, 370], [534, 343]], dtype=np.float32)
    BIRD_EYE_VIEW = np.array([[0, 0], [FRAME_WIDTH, 0], [FRAME_WIDTH, FRAME_HEIGHT],[0, FRAME_HEIGHT]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(SOURCE_POLYGONE, BIRD_EYE_VIEW)


    # Initialize the video capture
    video_input = opt.video

    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        print('Error: Unable to open video source.')
        return
    
  
    frame_generator = read_frames(cap)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))


    if opt.click:

        
        frame = next(frame_generator)
        selected_points=[]
        selected_points, img = get_clicked_points(frame)
 
        SOURCE_POLYGONE=np.array([], dtype=np.float32)

        SOURCE_POLYGONE=np.array([selected_points], dtype=np.float32)
        x, y= get_height_width_input(img)
        FRAME_HEIGHT,FRAME_WIDTH=x, y
        BIRD_EYE_VIEW = np.array([[0, 0], [FRAME_WIDTH, 0], [FRAME_WIDTH, FRAME_HEIGHT],[0, FRAME_HEIGHT]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(SOURCE_POLYGONE, BIRD_EYE_VIEW)

    pts = SOURCE_POLYGONE.astype(np.int32) 
    pts = pts.reshape((-1, 1, 2))

    polygon_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    cv2.fillPoly(polygon_mask, [pts], 255)
    # video writer objects
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(opt.output, fourcc, fps, (frame_width, frame_height))

    # Initialize the DeepSort tracker
    tracker = DeepSort(max_age=50)
    # select device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load YOLO model
    model = DetectMultiBackend(weights='weights/yolov9-e.pt',device=device, fuse=True)
    model = AutoShape(model)

    # Load the COCO class labels
    classes_path = "configs/coco.names"
    with open(classes_path, "r") as f:
        class_names = f.read().strip().split("\n")

    # Create a list of random colors to represent each class
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3)) 
    # FPS calculation variables
    frame_count = 0
    start_time = time.time()
    prev_positions={}
    speed_accumulator={}
    
    while True:
        try:
            frame = next(frame_generator)
        except StopIteration:
            break
        # Run model on each frame
        with torch.no_grad():
            results = model(frame)
        detect = []
        for det in results.pred[0]:
            label, confidence, bbox = det[5], det[4], det[:4]
            x1, y1, x2, y2 = map(int, bbox)
            class_id = int(label)
            # Filter out weak detections by confidence threshold and class_id
            if opt.class_id is None:
                if confidence < opt.conf:
                    continue
            else:
                if class_id != opt.class_id or confidence < opt.conf:
                    continue
                
            if polygon_mask[(y1 + y2) // 2, (x1 + x2) // 2] == 255:
                detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, int(label)])            
        tracks = tracker.update_tracks(detect, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id    
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            if polygon_mask[(y1+y2)//2,(x1+x2)//2] == 0:
                tracks.remove(track)
            color = colors[class_id]
            B, G, R = map(int, color)
            text = f"{track_id} - {class_names[class_id]}"
            center_pt = np.array([[(x1+x2)//2, (y1+y2)//2]], dtype=np.float32)
            transformed_pt = cv2.perspectiveTransform(center_pt[None, :, :], M)
            if track_id in prev_positions:
                prev_position = prev_positions[track_id]
                distance = calculate_distance(prev_position, transformed_pt[0][0])
                speed = calculate_speed(distance, fps)
                if track_id in speed_accumulator:
                    speed_accumulator[track_id].append(speed)
                    if len(speed_accumulator[track_id]) > 100:
                        speed_accumulator[track_id].pop(0)
                else:
                    speed_accumulator[track_id] = []
                    speed_accumulator[track_id].append(speed)
            prev_positions[track_id] = transformed_pt[0][0]
            # Draw bounding box and text
            frame = draw_corner_rect(frame, (x1, y1, x2 - x1, y2 - y1), line_length=15, line_thickness=3, rect_thickness=1, rect_color=(B, G, R), line_color=(R, G, B))
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 10, y1), (B, G, R), -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if track_id in speed_accumulator :
                avg_speed = sum(speed_accumulator[track_id]) / len(speed_accumulator[track_id])
                #print(avg_speed)
                cv2.rectangle(frame, (x1 - 1, y1-40 ), (x1 + len(f"Speed: {avg_speed:.0f} km/h") * 10, y1-20), (0, 0, 255), -1)
                cv2.putText(frame, f"Speed: {avg_speed:.0f} km/h", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # Apply Gaussian Blur
            if opt.blur_id is not None and class_id == opt.blur_id:
                print("true")
                if 0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]:
                    frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (99, 99), 3)

        cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.putText(frame, f"Height: {FRAME_HEIGHT}", (1500, 900), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Width: {FRAME_WIDTH}", (1530, 930), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow('speed_estimation', frame)
        writer.write(frame)
        frame_count += 1
        if frame_count % 10 == 0:
            elapsed_time = time.time() - start_time
            fps_calc = frame_count / elapsed_time
            print(f"FPS: {fps_calc:.2f}")
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    opt = parse_args()
    main(opt)