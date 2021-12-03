import numpy as np
import cv2
import torch

from .model import create_model

from .config import (
    BASE_DIR, NUM_CLASSES, DEVICE, CLASSES, RESIZE_TO
)
def detect(image_name):
    # this will help us create a different color for each class
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


    # load the best model and trained weights
    model = create_model(num_classes=NUM_CLASSES)
    checkpoint = torch.load(BASE_DIR / "outputs/best_model.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    # directory where all the images are present
    DIR_IMAGE = BASE_DIR.parent / "static/uploads"

    # define the detection threshold...
    # ... any detection having score below this will be discarded
    detection_threshold = 0.8

    image = cv2.imread(str(DIR_IMAGE / image_name))
    
    orig_image = image.copy()
    
    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image = cv2.resize(image, (RESIZE_TO, RESIZE_TO))
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float).cuda()
    # add batch dimension
    image = torch.unsqueeze(image, 0)

    with torch.no_grad():
        outputs = model(image.to(DEVICE))

    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # get all the predicited class names
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        
        # draw the bounding boxes and write the class name on top of it
        for j, box in enumerate(draw_boxes):
            class_name = pred_classes[j]
            color = COLORS[CLASSES.index(class_name)]
            
            y1 = int((int(box[1])*orig_image.shape[0])/RESIZE_TO)
            y2 = int((int(box[3])*orig_image.shape[0])/RESIZE_TO)
            x1 = int((int(box[0])*orig_image.shape[1])/RESIZE_TO)
            x2 = int((int(box[2])*orig_image.shape[1])/RESIZE_TO)

            cv2.rectangle(orig_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(orig_image, class_name, 
                        (x1, int((int(box[1]-5)*orig_image.shape[0])/RESIZE_TO)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 
                        2, lineType=cv2.LINE_AA)
        cv2.imwrite(str(DIR_IMAGE / image_name), orig_image)