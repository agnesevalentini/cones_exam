import cv2
import math
import numpy as np
import csv
import torch
from ultralytics import YOLO

from scipy import interpolate
from claude import process_stereo_lines

from load_data_as_tensor_v2 import to_tensor
from predict_and_plot import predict
from raceline_prediction_model_v3 import TrackNetConditioned

###### da spostare ######
symmetric=0
also_current_position=0
total_foresight=3 #basically f=total_foresight/2 NOTE must be even number if symmetric
foreward_foresight=3
total_sampling=2 #basically total_sampling
foreward_sampling=2

with_thetas=0 #yes=1 no=0
with_normal_dists=0 #difference between v1 and v2
input_size=(2+with_normal_dists+with_thetas)*(total_foresight+1)
hidden_size1=450
hidden_size2and3=200
#if you want it to be symmetric either put symmetric+1 or let it be (total_sampling)/2
#if you want it to be only foreward (no current position) let it be sampling+1
output_size=total_sampling+(1*also_current_position)
racetrack_model=TrackNetConditioned(input_size,hidden_size1,hidden_size2and3,output_size)
racetrack_model.load_state_dict(torch.load("track_model_very_small.pt", map_location="cpu"))
center=np.array([0])
##############



# IN METERS
CONES_KNOWN_WIDTH = 0.288 

# IN PIXELS
FOCAL_LENGTH = 663.4631958007812

def draw_spline(img, points, color=(255, 0, 255), thickness=2, smoothness=100):
    """
    Disegna una spline sui punti specificati.

    Args:
        img (np.ndarray): immagine su cui disegnare
        points (list of tuple): lista di punti [(x, y), (x, y), ...]
        color (tuple): colore BGR della spline (default viola)
        thickness (int): spessore linea
        smoothness (int): numero di segmenti della curva

    Returns:
        np.ndarray: immagine con spline disegnata
    """
    points = np.array(points, dtype=np.float32)
    x, y = points[:, 0], points[:, 1]

    # Crea spline parametrica
    tck, u = interpolate.splprep([x, y], s=0, k = min(3, len(points)-1))
    unew = np.linspace(0, 1, smoothness)
    out = interpolate.splev(unew, tck)

    # Disegna la spline
    for i in range(len(out[0]) - 1):
        pt1 = (int(out[0][i]), int(out[1][i]))
        pt2 = (int(out[0][i+1]), int(out[1][i+1]))
        cv2.line(img, pt1, pt2, color, thickness)

    return img

def line_pixels_cv(pt1, pt2, thickness=1, shape=None):
    if shape is None:
        # calcolo bounding box minima che contiene la linea
        max_x = max(pt1[0], pt2[0]) + thickness + 1
        max_y = max(pt1[1], pt2[1]) + thickness + 1
        shape = (max_y, max_x)

    mask = np.zeros(shape, dtype=np.uint8)
    cv2.line(mask, pt1, pt2, 255, thickness)

    ys, xs = np.where(mask > 0)
    return list(zip(xs, ys))


def pixel_al_t(line, t):
    """
    line: array/lista di (x,y) che rappresentano la linea (in ordine)
    t: valore normalizzato [0,1] che indica la posizione lungo la linea
    """
    line = np.array(line, dtype=float)

    # lunghezze dei segmenti tra pixel consecutivi
    diffs = np.diff(line, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)

    # lunghezza cumulativa
    cumlen = np.cumsum(segment_lengths)
    total_len = cumlen[-1]

    # lunghezza target
    target_len = t * total_len

    # trova il segmento in cui cade il punto
    i = np.searchsorted(cumlen, target_len)

    # lunghezza fino al punto precedente
    prev_len = 0 if i == 0 else cumlen[i-1]

    # fattore di interpolazione dentro al segmento
    alpha = (target_len - prev_len) / segment_lengths[i]

    # punto interpolato (può non coincidere con un pixel esatto)
    point = (1 - alpha) * line[i] + alpha * line[i+1]

    # se vuoi un vero pixel (arrotondato)
    pixel = tuple(np.round(point).astype(int))

    return pixel


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
    #plot_points(box_centers, img)
    # for i in range(len(box_centers) - 1):
    #     pt1 = (int(box_centers[i][0]), int(box_centers[i][1]))
    #     pt2 = (int(box_centers[i + 1][0]), int(box_centers[i + 1][1]))
    #     cv2.line(img, pt1, pt2, (0, 255, 0), 2)
    return img, box_centers

def plot_bboxes(results):
    img = results[0].orig_img # original image
    names = results[0].names # class names dict
    scores = results[0].boxes.conf.numpy() # probabilities
    classes = results[0].boxes.cls.numpy() # predicted classes
    boxes = results[0].boxes.xyxy.numpy().astype(np.int32) # bboxes
    for score, cls, bbox in zip(scores, classes, boxes): # loop over all bboxes
        class_label = names[cls] # class name
        label = f"{score:0.2f}" # bbox label
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
        # label_size = cv2.getTextSize(label, # labelsize in pixels 
        #                              fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        #                              fontScale=1, thickness=1)
        # lbl_w, lbl_h = label_size[0] # label w and h
        # lbl_w += 2* lbl_margin # add margins on both sides
        # lbl_h += 2*lbl_margin
        # img = cv2.rectangle(img, (bbox[0], bbox[1]), # plot label background
        #                      (bbox[0]+lbl_w, bbox[1]-lbl_h),
        #                      color=color, 
        #                      thickness=-1) # thickness=-1 means filled rectangle
        # cv2.putText(img, label, (bbox[0]+ lbl_margin, bbox[1]-lbl_margin), # write label to the image
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #             fontScale=1.0, color=font_color,
        #             thickness=1)
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

    #print(f"line1_points: {line1_points}, line2_points: {line2_points}")
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

    if line2_points[0][1] < line2_points[1][1]:
        return angle_degrees
    else:
        return -angle_degrees

cap = cv2.VideoCapture("FSAE2L.mp4")

model = YOLO("train/weights/best.pt")

#i: int = 0

p: int = 0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 15
frame_width = int(1280)
frame_height = int(720)
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, save=True, conf=0.7)
    
    frame_count = 0
    cv2.imwrite(f"output/frame_{frame_count}.jpg", frame)

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
    
    if len(lines) < 2:
        # Not enough lines to compare
        with open(f'output/results_{p}.txt', 'w') as f:
            f.write("Not enough lines detected for analysis.")
        continue

    list_of_all_lines_pixels = []

    with open(f'output/results_{p}.txt', 'w') as f:

        frame_data = np.empty((len(lines)+1, 2))  # 2 columns for [line_width, angle]
        Raceline = np.empty((len(lines)+1, 1))    # 1 column for raceline value
        #writer.writerow(["line_width", "angle", "correction", "distance_between_normals"])
        frame_data[0] = np.array([5.0, 0.0])  # First line with fake angle 0.0
        Raceline[0]=np.array(0.5)

        writer = csv.writer(f)

        writer.writerow(["line_width", "angle", "correction", "distance_between_normals"])
        writer.writerow([5.0, 0.0, 0.0, 5.0])  # First line with fake angle 0.0

        left_point_wing = (483, 494)
        right_point_wing = (835, 493)
        #cv2.line(img, pt1, pt2, (0, 255, 0), 1)

        line_car = [left_point_wing, right_point_wing]
        line_car_pixels = []
        for x in range(min(int(line_car[0][0]), int(line_car[1][0])), max(int(line_car[0][0]), int(line_car[1][0])) + 1):
            y = int(line_car[0][1] + (line_car[1][1] - line_car[0][1]) * (x - line_car[0][0]) / (line_car[1][0] - line_car[0][0]))
            line_car_pixels.append((x, y))

        f.write(f"Line 0 pixels: {line_car_pixels}\n")

        #list_of_all_lines_pixels.append(line_car_pixels)

        #pixels = line_pixels_cv(pt1, pt2, thickness=1, shape=img.shape)
        # i need to verify that pixels and line0_pixels are the same
        #point0 = pixel_al_t(line0_pixels, 0.1006)
        first_normal = lines[0]

        angle = calculate_angle_between_lines(line_car, first_normal)

        # cv2.putText(img, f"{angle:.5f} degrees", (left_point_wing[0], left_point_wing[1] - 10), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        writer.writerow([5.0, angle, 0.0, 5.0])

        frame_data[1]=np.array([5.0, angle])
        Raceline[1]=np.array(0)

        points = [] #[point0]

        line1_pixels = []
        for x in range(min(int(first_normal[0][0]), int(first_normal[1][0])), max(int(first_normal[0][0]), int(first_normal[1][0])) + 1):
            y = int(first_normal[0][1] + (first_normal[1][1] - first_normal[0][1]) * (x - first_normal[0][0]) / (first_normal[1][0] - first_normal[0][0]))
            line1_pixels.append((x, y))

        list_of_all_lines_pixels.append(line1_pixels)

        #start_pt = pixel_al_t(line1_pixels, 0.4028)
        #start_pt = pixel_al_t(line1_pixels, 0.5853)

        #processed_stereo_lines = process_stereo_lines(img, line_car_pixels, line1_pixels, start_pt)
        #img = processed_stereo_lines['output_image']
        #transformed_points = processed_stereo_lines['transformed_points']


        j = 0
        while j < len(lines) - 1:
            line1 = lines[j]
            line2 = lines[j + 1]
            #print(f"line1: {line1}, line2: {line2}")
            angle = calculate_angle_between_lines(line1, line2)
            #print(f"Angle between the two lines: {angle:.2f} degrees")
            # Also add the angle value near the intersection of the lines (if they intersect)
            # Calculate approximate midpoint between the two lines for display
            # mid_x = int((line1[0][0] + line1[1][0] + line2[0][0] + line2[1][0]) / 4)
            # mid_y = int((line1[0][1] + line1[1][1] + line2[0][1] + line2[1][1]) / 4)
            # cv2.putText(img, f"{angle:.5f} degrees", (mid_x, mid_y), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            # Also add the angle value to the results file
            #f.write(f"Angle between line {j} and line {j + 1}: {angle:.5f} degrees\n")
            writer.writerow([5.0, angle, 0.0, 5.0])  # Example values

            frame_data[j+2]=np.array([5.0, angle])
            Raceline[j+2]=np.array(0)

            #list all pixel from line
            # line1_pixels = []
            # for x in range(min(int(line1[0][0]), int(line1[1][0])), max(int(line1[0][0]), int(line1[1][0])) + 1):
            #     y = int(line1[0][1] + (line1[1][1] - line1[0][1]) * (x - line1[0][0]) / (line1[1][0] - line1[0][0]))
            #     line1_pixels.append((x, y))
            #point1 = pixel_al_t(line1_pixels, 0.3568)

            #point1 = pixel_al_t(line1_pixels, 0.5853)

            #f.write(f"point1: {point1}\n")
            #f.write(f"Line {j} pixels: {line1_pixels}\n")

            #cv2.circle(img, point1, 5, (255, 0, 255), -1)

            line2_pixels = []
            for x in range(min(int(line2[0][0]), int(line2[1][0])), max(int(line2[0][0]), int(line2[1][0])) + 1):
                y = int(line2[0][1] + (line2[1][1] - line2[0][1]) * (x - line2[0][0]) / (line2[1][0] - line2[0][0]))
                line2_pixels.append((x, y))
            #print(f"Line {j + 1} pixels: {line2_pixels}")
            #point2 = pixel_al_t(line2_pixels, 0.3568)

            list_of_all_lines_pixels.append(line2_pixels)

            #point2 = pixel_al_t(line2_pixels, 0.6629)

            #f.write(f"point2: {point2}\n")

            #cv2.circle(img, point2, 5, (255, 0, 255), -1)

            #points.append(point1)
            #points.append(point2)

            #processed_stereo_lines = process_stereo_lines(img, transformed_points, line2_pixels, point2)
            #img = processed_stereo_lines['output_image']
            #transformed_points = processed_stereo_lines['transformed_points']

            # frame_data[j+1]=np.array([5.0, angle])  # Example values add 0.0, 5.0]
            # Raceline[j+1]=np.array([0])
            j += 1

    
        frame_data = np.array([frame_data[:, 0], frame_data[:, 1]])
        back_foresight = total_foresight-foreward_foresight
        back_sampling = total_sampling-foreward_sampling
        X,Y = to_tensor(frame_data,Raceline,center,back_foresight,foreward_foresight,back_sampling,foreward_sampling-1,len(frame_data))
        current_positions=Y[:,total_sampling-foreward_sampling]
        file = X,Y,current_positions

        predictions, raceline_for_plot = predict(file,racetrack_model)

        f.write(f"Predictions: {predictions}\n")

        predictions_as_np_array = predictions.detach().numpy()

        predictions_list = predictions_as_np_array.flatten().tolist()

        start_pt_first_normal = pixel_al_t(list_of_all_lines_pixels[0], predictions_list[0])
        start_pt_second_normal = pixel_al_t(list_of_all_lines_pixels[1], predictions_list[1])

        # cv2.circle(img, start_pt_first_normal, 5, (0, 255, 0), -1)
        # cv2.circle(img, start_pt_second_normal, 5, (0, 255, 0), -1)
        
        first_direction = ''

        if predictions_list[0] > 0.5:
            first_direction = 'left'
        else:
            first_direction = 'right'

        second_direction = ''

        if predictions_list[1] > 0.5:
            second_direction = 'left'
        else:
            second_direction = 'right'

        processed_stereo_lines = process_stereo_lines(img, line_car_pixels, list_of_all_lines_pixels[0], start_pt_first_normal, direction=first_direction)
        img = processed_stereo_lines['output_image']
        transformed_points = processed_stereo_lines['transformed_points']

        processed_stereo_lines = process_stereo_lines(img, transformed_points, list_of_all_lines_pixels[1], start_pt_second_normal, direction=second_direction)
        img = processed_stereo_lines['output_image']
        transformed_points = processed_stereo_lines['transformed_points']

    frame_count += 1
    #points = [(482, 496), (349, 411), (356, 359)]
    #img = draw_spline(img, points)

    # tensor1 = torch.tensor(0.0709, dtype=torch.float64)
    # tensor2 = torch.tensor(0.1006, dtype=torch.float64)



    #cv2.imwrite(f"output/output_{p}.jpg", img)
    #cv2.imshow("Frame", frame)
    #cv2.waitKey(1)


    # for r in results:
    #     #r.save_txt(f'output/results_{i}.txt')
    #     for box in r.boxes:
    #         with open(f'output/results_{p}.txt', 'a') as f:
    #             f.write(f"Frame shape: {frame.shape}\n")
    #             f.write(f"{box.xyxy.numpy()} {box.conf.numpy()} {box.cls.numpy()}\n")
    out.write(img)
    p += 1

out.release()
cap.release()