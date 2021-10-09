# %% Utils

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

def apply_nms(dets, thresh = 0.4):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

## %% Detection

import cv2
import onnxruntime
import numpy as np

def scale_image(img, input_size):
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio>model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros( (input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img
        return det_img, det_scale

def detect(model_path, image, threshold):
    fmc = 3
    use_kps = True
    input_mean = 127.5
    input_std = 128.0
    num_anchors = 2
    feat_stride_fpn = [8, 16, 32]

    input_size = tuple(image.shape[0:2][::-1])
    blob = cv2.dnn.blobFromImage(image, 1.0/input_std, input_size, (input_mean, input_mean, input_mean), swapRB=True)

    session = onnxruntime.InferenceSession(model_path)
    #print(session.get_providers())
    inputs = session.get_inputs()
    input_name = inputs[0].name
    outputs = session.get_outputs()
    output_names = []
    for o in outputs:
        output_names.append(o.name)

    net_outs = session.run(output_names, {input_name : blob})

    input_width = blob.shape[3]
    input_height = blob.shape[2]

    scores_list = []
    bboxes_list = []
    kpss_list = []
    center_cache = {}

    for index, stride in enumerate(feat_stride_fpn):
        scores = net_outs[index]
        bbox_preds = net_outs[index+fmc]
        bbox_preds = bbox_preds * stride
        if use_kps:
            kps_preds = net_outs[index+fmc*2] * stride
        height = input_height // stride
        width = input_width // stride
        K = height * width
        key = (height, width, stride)
        if key in center_cache:
            anchor_centers = center_cache[key]
        else:
            #solution-1, c style:
            #anchor_centers = np.zeros( (height, width, 2), dtype=np.float32 )
            #for i in range(height):
            #    anchor_centers[i, :, 1] = i
            #for i in range(width):
            #    anchor_centers[:, i, 0] = i

            #solution-2:
            #ax = np.arange(width, dtype=np.float32)
            #ay = np.arange(height, dtype=np.float32)
            #xv, yv = np.meshgrid(np.arange(width), np.arange(height))
            #anchor_centers = np.stack([xv, yv], axis=-1).astype(np.float32)

            #solution-3:
            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            #print(anchor_centers.shape)

            anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
            if num_anchors>1:
                anchor_centers = np.stack([anchor_centers]*num_anchors, axis=1).reshape( (-1,2) )
            if len(center_cache)<100:
                center_cache[key] = anchor_centers

        pos_inds = np.where(scores>=threshold)[0]
        bboxes = distance2bbox(anchor_centers, bbox_preds)
        pos_scores = scores[pos_inds]
        pos_bboxes = bboxes[pos_inds]
        scores_list.append(pos_scores)
        bboxes_list.append(pos_bboxes)
        if use_kps:
            kpss = distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)
    return scores_list, bboxes_list, kpss_list

def parse_output(detection_scale, scores_list, bboxes_list, kpss_list):
    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]

    transformed_boxes = np.vstack(bboxes_list) / detection_scale
    all_boxes = np.hstack((transformed_boxes, scores)).astype(np.float32, copy=False)
    all_boxes = all_boxes[order, :]
    valid_boxes = apply_nms(all_boxes)
    detections = all_boxes[valid_boxes, :]

    use_keypoints = kpss_list is not None
    keypoints = None
    if use_keypoints:
        keypoints = np.vstack(kpss_list) / detection_scale
        keypoints = keypoints[order,:,:]
        keypoints = keypoints[valid_boxes,:,:]

    return detections, keypoints

## %% Create faces

class Face(object):

    def __init__(self, box, kps, confidence):
        self.box = box
        self.kps = kps
        self.confidence = confidence

def create_face_data(img, boxes, kpss):
    faces = []
    for i in range(boxes.shape[0]):
        abs_box = boxes[i, 0:4]

        width = img.shape[1]
        height = img.shape[0]

        box = abs_box
        #box = np.array([abs_box[0]/width, abs_box[1]/height, abs_box[2]/width, abs_box[3]/height])

        confidence = boxes[i, 4]
        kps = None
        if kpss is not None:
            kps = kpss[i]
            # kps = []
            # for kp in kpss[i]:
            #     kps.append(np.array([kp[0]/width, kp[1]/height]))
        face = Face(box, kps, confidence)
        faces.append(face)

    return faces
    
## %% Drawing

def draw_faces(image, faces):
    for i in range(len(faces)):
        face = faces[i]
        box = face.box.astype(int)
        color = (0, 0, 255)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
        if face.kps is not None:
            kps = face.kps.astype(int)
            for l in range(kps.shape[0]):
                color = (0, 0, 255)
                if l == 0 or l == 1:
                    color = (0, 255, 0)
                cv2.circle(image, (kps[l][0], kps[l][1]), 1, color, 2)

def save_result(image, faces, img_path):
    extension_split = os.path.splitext(img_path)
    new_file_name = extension_split[0] + "_detected"

    extension = extension_split[1]

    copy_image = image.copy()
    draw_faces(copy_image, faces)

    cv2.imwrite(f'{new_file_name}{extension}', copy_image)

## %% Main

import os

def detect_faces_in_image(image_path, nn_path, nn_input_size):
    image = cv2.imread(image_path)

    scaled_image, detection_scale = scale_image(image, nn_input_size)

    score_data, box_data, keypoint_data = detect(nn_path, scaled_image, 0.5)
    boxes, key_points = parse_output(detection_scale, score_data, box_data, keypoint_data)

    faces = create_face_data(image, boxes, key_points)

    save_result(image, faces, image_path)

    return faces

    
## %% Run

image_folder_path = '../images'
nn_path = "det_10g.onnx"
nn_input_size = (640, 640)

all_files = os.listdir(image_folder_path)
image_files = list(filter(lambda f: f.endswith(('.jpg', 'png')), all_files)) 
print(image_files)

for file in image_files:
    faces = detect_faces_in_image(os.path.join(image_folder_path, file), nn_path, nn_input_size)
    print(f'{len(faces)} boxes:')
    for face in faces:
        print(face.box)
    print('key points:')
    for face in faces:
        print(face.kps)

print()
print('done')
# %%
