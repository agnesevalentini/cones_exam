import cv2
import numpy as np
from ultralytics import YOLO


def plot_points(points, img):
    for point in points:
        cv2.circle(img, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
    return img

def write_line_between_box_centers(img, boxes):
    box_centers = []
    for box in boxes:
        box_center = [(box[2] + box[0]) / 2 , (box[3] + box[1]) / 2 ]
        box_centers.append(box_center)
    # print("box_centers:", box_centers)
    # print(f"box_centers 0: {int(box_centers[0][1])}")
    plot_points(box_centers, img)
    for i in range(len(box_centers) - 1):
        pt1 = (int(box_centers[i][0]), int(box_centers[i][1]))
        pt2 = (int(box_centers[i + 1][0]), int(box_centers[i + 1][1]))
        cv2.line(img, pt1, pt2, (0, 255, 0), 2)
    return img, box_centers

def plot_bboxes(results):
    img = results[0].orig_img # original image
    names = results[0].names # class names dict
    scores = results[0].boxes.conf.numpy() # probabilities
    classes = results[0].boxes.cls.numpy() # predicted classes
    boxes = results[0].boxes.xyxy.numpy().astype(np.int32) # bboxes
    for score, cls, bbox in zip(scores, classes, boxes): # loop over all bboxes
        class_label = names[cls] # class name
        label = f"{class_label} : {score:0.2f}" # bbox label
        lbl_margin = 3 #label margin

        if class_label == 'blue_cone':
            color = (255, 0, 0)
            font_color = (255, 255, 255)
        elif class_label == 'yellow_cone':
            color = (0, 255, 255)
            font_color = (0, 0, 0)
        else:
            color = (0, 0, 255)
            font_color = (0, 0, 0)
        img = cv2.rectangle(img, (bbox[0], bbox[1]),
                            (bbox[2], bbox[3]),
                            color=color,
                            thickness=1)
        label_size = cv2.getTextSize(label, # labelsize in pixels 
                                     fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                     fontScale=1, thickness=1)
        lbl_w, lbl_h = label_size[0] # label w and h
        lbl_w += 2* lbl_margin # add margins on both sides
        lbl_h += 2*lbl_margin
        img = cv2.rectangle(img, (bbox[0], bbox[1]), # plot label background
                             (bbox[0]+lbl_w, bbox[1]-lbl_h),
                             color=color, 
                             thickness=-1) # thickness=-1 means filled rectangle
        cv2.putText(img, label, (bbox[0]+ lbl_margin, bbox[1]-lbl_margin), # write label to the image
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0, color=font_color,
                    thickness=1)
    return img

def calculate_angle_between_lines(line1_points, line2_points):
    """
    Calculate the angle between two lines defined by their endpoints.
    
    Args:
        line1_points: List of two points [(x1, y1), (x2, y2)] defining the first line
        line2_points: List of two points [(x1, y1), (x2, y2)] defining the second line
    
    Returns:
        Angle in degrees between the two lines (0-90 degrees)
    """
    # Calculate direction vectors for both lines
    vector1 = np.array([line1_points[1][0] - line1_points[0][0], 
                       line1_points[1][1] - line1_points[0][1]])
    vector2 = np.array([line2_points[1][0] - line2_points[0][0], 
                       line2_points[1][1] - line2_points[0][1]])
    
    # Calculate the dot product
    dot_product = np.dot(vector1, vector2)
    
    # Calculate the magnitudes
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    
    # Calculate the cosine of the angle
    cos_angle = dot_product / (magnitude1 * magnitude2)
    
    # Clamp the value to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Calculate the angle in radians and convert to degrees
    angle_radians = np.arccos(abs(cos_angle))  # abs() to get the acute angle
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees

cap = cv2.VideoCapture("FSAE2.mp4")

model = YOLO("cones_exam/train/weights/best.pt")

#i: int = 0

p: int = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, save=True, conf=0.8)
    
    #cv2.imwrite(f"output/frame_{i}.jpg", frame)

    # Process the frame (e.g., object detection)
    # ...
    
    img = plot_bboxes(results)
    yellow_box_list = []
    blue_box_list = []
    for r in results:
        for box in r.boxes:
            if box.cls == 1:  # Assuming 1 is the class ID for yellow cones
                yellow_box_list.append(box)
            elif box.cls == 0:  # Assuming 0 is the class ID for blue cones
                blue_box_list.append(box)
    
    sorted_yellow_boxes = sorted(yellow_box_list, key=lambda box: box.xyxy[0][1], reverse=True)
    sorted_blue_boxes = sorted(blue_box_list, key=lambda box: box.xyxy[0][1], reverse=True)

    i = 0

    lines = []

    while i < min(len(sorted_blue_boxes), len(sorted_yellow_boxes)):
        img, box_centers = write_line_between_box_centers(img, [sorted_blue_boxes[i].xyxy[0], sorted_yellow_boxes[i].xyxy[0]])
        #print(f"box_centers: {box_centers}")
        lines.append([box_centers[0], box_centers[1]])
        i += 1

    j = 0
    while j < len(lines) - 1:
        line1 = lines[j]
        line2 = lines[j + 1]
        #print(f"line1: {line1}, line2: {line2}")
        angle = calculate_angle_between_lines(line1, line2)
        #print(f"Angle between the two lines: {angle:.2f} degrees")
        # Also add the angle value near the intersection of the lines (if they intersect)
        # Calculate approximate midpoint between the two lines for display
        mid_x = int((line1[0][0] + line1[1][0] + line2[0][0] + line2[1][0]) / 4)
        mid_y = int((line1[0][1] + line1[1][1] + line2[0][1] + line2[1][1]) / 4)
        cv2.putText(img, f"{angle:.5f} degrees", (mid_x, mid_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        j += 1

    cv2.imwrite(f"output/output_{p}.jpg", img)
    cv2.imshow("Frame", frame)
    cv2.waitKey(5)
    for r in results:
        #r.save_txt(f'output/results_{i}.txt')
        for box in r.boxes:
            with open(f'output/results_{p}.txt', 'a') as f:
                f.write(f"Frame shape: {frame.shape}\n")
                f.write(f"{box.xyxy.numpy()} {box.conf.numpy()} {box.cls.numpy()}\n")
    p += 1

cap.release()