# |  ||
# || |_

import keras.backend as K
from model import C, S, B

LAMBDA_COORD=5
LAMBDA_NOOBJ=0.5

def xywh2minmax(xy, wh):
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2
    return xy_min, xy_max

def iou(pred_mins, pred_maxes, true_mins, true_maxes):
    intersect_mins = K.maximum(pred_mins, true_mins)
    intersect_maxes = K.minimum(pred_maxes, true_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_wh = pred_maxes - pred_mins
    true_wh = true_maxes - true_mins
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    return intersect_areas / (union_areas + 1e-10)

def yolo_head(feats):
    conv_dims = K.shape(feats)[1:3]  # (7, 7)
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])

    conv_width_index = K.tile(K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = K.flatten(K.transpose(conv_width_index))
    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = K.cast(conv_index, K.dtype(feats))

    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))

    box_xy = (feats[..., :2] + conv_index) / conv_dims
    box_wh = feats[..., 2:4]

    return box_xy, box_wh

def yolo_loss(y_true, y_pred):
    label_class = y_true[..., :C]                  # batch x 7 x 7 x 2
    label_box = y_true[..., C:C+4]       # batch x 7 x 7 x 4
    response_mask = y_true[..., C+4]               # batch x 7 x 7
    response_mask = K.expand_dims(response_mask)             # batch x 7 x 7 x 1

    predict_class = y_pred[..., :C]                # batch x 7 x 7 x 2
    predict_trust = y_pred[..., C:C+B]   # batch x 7 x 7 x 2
    predict_box = y_pred[..., C+B:]   # batch x 7 x 7 x 8

    _label_box = K.reshape(label_box, [-1, S, S, 1, 4])
    _predict_box = K.reshape(predict_box, [-1, S, S, B, 4])

    label_xy, label_wh = yolo_head(_label_box)
    label_xy = K.expand_dims(label_xy, axis=4)
    label_wh = K.expand_dims(label_wh, axis=4)
    label_xy_min, label_xy_max = xywh2minmax(label_xy, label_wh)

    predict_xy, predict_wh = yolo_head(_predict_box)
    predict_xy = K.expand_dims(predict_xy, 4)
    predict_wh = K.expand_dims(predict_wh, 4)
    predict_xy_min, predict_xy_max = xywh2minmax(predict_xy, predict_wh)

    iou_scores = iou(predict_xy_min, predict_xy_max, label_xy_min, label_xy_max)
    best_ious = K.max(iou_scores, axis=4)
    best_box = K.max(best_ious, axis=3, keepdims=True)

    box_mask = K.cast(best_ious >= best_box, K.dtype(best_ious))  # batch x 7 x 7 x 2

    no_object_loss = LAMBDA_NOOBJ * (1 - box_mask * response_mask) * K.square(0 - predict_trust)
    object_loss = box_mask * response_mask * K.square(1 - predict_trust)
    confidence_loss = K.sum(no_object_loss + object_loss)

    class_loss = response_mask * K.square(label_class - predict_class)
    class_loss = K.sum(class_loss)

    box_mask = K.expand_dims(box_mask)
    response_mask = K.expand_dims(response_mask)

    label_xy, label_wh = yolo_head(_label_box)
    predict_xy, predict_wh = yolo_head(_predict_box)

    box_loss = LAMBDA_COORD * box_mask * response_mask * K.square(label_xy - predict_xy) 
    box_loss += LAMBDA_COORD * box_mask * response_mask * K.square(K.sqrt(K.maximum(label_wh, 1e-10)) - K.sqrt(K.maximum(predict_wh, 1e-10)))
    box_loss = K.sum(box_loss)

    loss = (confidence_loss + class_loss + box_loss) / K.cast(K.shape(y_pred)[0], K.floatx())
    return loss