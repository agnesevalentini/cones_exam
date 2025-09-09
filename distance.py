import cv2
import numpy as np
from ultralytics import YOLO

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
        img = cv2.rectangle(img, (bbox[0], bbox[1]),
                            (bbox[2], bbox[3]),
                            color=(0, 0, 255),
                            thickness=1)
        label_size = cv2.getTextSize(label, # labelsize in pixels 
                                     fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                     fontScale=1, thickness=1)
        lbl_w, lbl_h = label_size[0] # label w and h
        lbl_w += 2* lbl_margin # add margins on both sides
        lbl_h += 2*lbl_margin
        img = cv2.rectangle(img, (bbox[0], bbox[1]), # plot label background
                             (bbox[0]+lbl_w, bbox[1]-lbl_h),
                             color=(0, 0, 255), 
                             thickness=-1) # thickness=-1 means filled rectangle
        cv2.putText(img, label, (bbox[0]+ lbl_margin, bbox[1]-lbl_margin), # write label to the image
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0, color=(255, 255, 255 ),
                    thickness=1)
    return img


# Load the YOLO model
model = YOLO("best.pt")

# Load an image
image_path = "image.jpg"
image = cv2.imread(image_path)

# Perform inference
results = model(image, save_txt=True)

# Known width of the object (e.g., a car width in meters)
KNOWN_WIDTH = 0.288  # Example width in meters

# Focal length of the camera (calibrated)
FOCAL_LENGTH = 1114.6426  # Example focal length in pixels

# Iterate through the results and calculate distances
#print(f"results: {results}")
for r in results:
    for box in r.boxes:
        #print(box)
        cls = box.cls
        conf = box.conf
        if conf >= 0.5:
            # Calculate the width of the bounding box in pixels
            box_width = box.xyxy[0][2] - box.xyxy[0][0]
            # Calculate the distance
            distance = (KNOWN_WIDTH * FOCAL_LENGTH) / box_width
            print(f"Object: {model.names[int(cls)]}, Distance: {distance:.2f} meters, box: {box.xyxy} ")

# Display the image with bounding boxes and distances
# for r in results:
#     r.plot()
# cv2.imshow("Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
i: int = 0
for r in results:
    r.save_txt(f'test{i}.txt')
    i+=1

img = plot_bboxes(results)

cv2.imwrite("output.jpg", img)