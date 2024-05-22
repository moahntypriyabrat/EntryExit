
from django.conf import settings
from .models import Video
import os
import cv2
import numpy as np
from . import Person
from ultralytics import YOLO


def create_video(frames, output_path, fps=30):
    # Get the dimensions of the frames
    height, width, _ = frames[0].shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write frames to the video
    for frame in frames:
        out.write(frame)

    # Release the video writer
    out.release()


def detect_object(video_id):
    video = Video.objects.get(id=video_id)
    input_path = os.path.join(settings.MEDIA_ROOT, video.original_video.name)
    output_v_name = video.original_video.name.split('/')[-1]
    output_path = os.path.join(settings.MEDIA_ROOT, r'videos\processed', f'processed_{output_v_name}')

    # Load YOLO model
    model = YOLO("yolov8n.pt")

    # Video input
    cap = cv2.VideoCapture(input_path)

    # Define the label mapping
    label_mapping = {
        0: 'Person',
        1: 'Bicycle',
        2: 'Car',
        3: 'Motorbike',
        # Add mappings for other classes if needed
    }

    # Capture properties
    w = cap.get(3)
    h = cap.get(4)
    frameArea = h * w
    areaTH = frameArea / 300

    # out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 42.99, (int(w), int(h)))

    # Variables
    font = cv2.FONT_HERSHEY_SIMPLEX
    persons = []
    max_p_age = 1
    pid = 1
    entry_count = 0
    exit_count = 0

    # Lines coordinate for counting
    line_up = int(1.5 * (h / 6))
    line_down = int(3.5 * (h / 6))
    up_limit = int(0.5 * (h / 6))
    down_limit = int(4.5 * (h / 6))

    pt1 = [0, line_down]
    pt2 = [w, line_down]
    pts_L1 = np.array([pt1, pt2], np.int32)
    pts_L1 = pts_L1.reshape((-1, 1, 2))

    pt3 = [0, line_up]
    pt4 = [w, line_up]
    pts_L2 = np.array([pt3, pt4], np.int32)
    pts_L2 = pts_L2.reshape((-1, 1, 2))

    pt5 = [0, up_limit]
    pt6 = [w, up_limit]
    pts_L3 = np.array([pt5, pt6], np.int32)
    pts_L3 = pts_L3.reshape((-1, 1, 2))

    pt7 = [0, down_limit]
    pt8 = [w, down_limit]
    pts_L4 = np.array([pt7, pt8], np.int32)
    pts_L4 = pts_L4.reshape((-1, 1, 2))

    frame_video = []
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:  # If frame could not be read (end of video)
            break
        frame = frame[:, 20:]

        # Object detection with YOLO
        results = model.predict(frame, conf=0.2)

        # Process YOLO detections
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            ids = result.boxes.cls.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy()

            for box, id, score in zip(boxes, ids, scores):
                label_name = label_mapping.get(id, 'Unknown')
                if label_name != 'Person':
                    continue
                x1, y1, x2, y2 = box

                # Draw bounding box
                frame_with_text = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Check if the person crosses the counting lines
                cy = int((y1 + y2) / 2)
                if cy in range(up_limit, down_limit):
                    for i in persons:
                        if abs((x1 + x2) // 2 - i.getX()) <= (x2 - x1) and abs(cy - i.getY()) <= (y2 - y1):
                            i.updateCoords((x1 + x2) // 2, cy)
                            if i.going_UP(line_down, line_up):
                                entry_count += 1
                            elif i.going_DOWN(line_down, line_up):
                                exit_count += 1
                            break
                    else:
                        p = Person.MyPerson(pid, (x1 + x2) // 2, cy, max_p_age)
                        persons.append(p)
                        pid += 1

        # Display entry and exit count
        str_up = 'Exit: ' + str(entry_count)
        str_down = 'Entry: ' + str(exit_count)
        frame_with_text = cv2.putText(frame, str_down, (20, 70), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        frame_with_text = cv2.putText(frame, str_up, (20, 100), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        # Display frame
        # out.write(frame_with_text)
        frame_video.append(frame_with_text)
        cv2.imshow('Counting', frame_with_text)

        # Press ESC to exit
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    create_video(frame_video, output_path)

    # Release resources
    cap.release()
    # out.release()
    cv2.destroyAllWindows()

    video.processed_video.name = f'videos/processed/processed_{output_v_name}'
    video.save()

