import math

import tensorflow as tf

from utils import bbox_utils


def get_hyper_params():
    return {
        "img_size": 300,
        "feature_map_shapes": [19, 10, 5, 3, 2, 1],
        "aspect_ratios": [[1., 2., 1. / 2.],
                          [1., 2., 1. / 2., 3., 1. / 3.],
                          [1., 2., 1. / 2., 3., 1. / 3.],
                          [1., 2., 1. / 2., 3., 1. / 3.],
                          [1., 2., 1. / 2.],
                          [1., 2., 1. / 2.]],
        "iou_threshold": 0.5,
        "neg_pos_ratio": 3,
        "loc_loss_alpha": 1,
        "variances": [0.1, 0.1, 0.2, 0.2]
    }


def scheduler(epoch):
    if epoch < 100:
        return 1e-3
    elif epoch < 125:
        return 1e-4
    else:
        return 1e-5


def get_step_size(total_items, batch_size):
    return math.ceil(total_items / batch_size)


def custom_generator(dataset, prior_boxes, hyper_params):
    while True:
        img, gt_boxes, gt_labels = next(dataset)
        img       = tf.constant(img)
        gt_boxes  = tf.constant(gt_boxes)
        gt_labels = tf.constant(gt_labels)

        actual_deltas, actual_labels = calculate_actual_outputs(prior_boxes, gt_boxes, gt_labels, hyper_params)

        print(f'Img shape: {img.shape}')
        print(f'actual_deltas Shape: {actual_deltas.shape}')
        print(f'actual_labels Shape: {actual_labels.shape}')
        yield img, (actual_deltas, actual_labels)

def generator(dataset, prior_boxes, hyper_params):
    """Tensorflow data generator for fit method, yielding inputs and outputs.
    inputs:
        dataset = tf.data.Dataset, PaddedBatchDataset
        prior_boxes = (total_prior_boxes, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
        hyper_params = dictionary

    outputs:
        yield inputs, outputs
    """
    while True:
        for image_data in dataset:
            img, gt_boxes, gt_labels = image_data
            actual_deltas, actual_labels = calculate_actual_outputs(prior_boxes, gt_boxes, gt_labels, hyper_params)

            yield img, (actual_deltas, actual_labels)


def calculate_actual_outputs(prior_boxes, gt_boxes, gt_labels, hyper_params):
    """Calculate ssd actual output values.
    Batch operations supported.
    inputs:
        prior_boxes = (total_prior_boxes, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
        gt_boxes (batch_size, gt_box_size, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
        gt_labels (batch_size, gt_box_size)
        hyper_params = dictionary

    outputs:
        bbox_deltas = (batch_size, total_bboxes, [delta_y, delta_x, delta_h, delta_w])
        bbox_labels = (batch_size, total_bboxes, [0,0,...,0])
    """
    batch_size        = 32 #tf.shape(gt_boxes)[0]
    total_labels      = hyper_params["total_labels"]
    iou_threshold     = hyper_params["iou_threshold"]
    variances         = hyper_params["variances"]
    total_prior_boxes = prior_boxes.shape[0]

    # Calculate iou values between each bboxes and ground truth boxes
    iou_map = bbox_utils.generate_iou_map(prior_boxes, gt_boxes)
    # Get max index value for each row
    max_indices_each_gt_box = tf.argmax(iou_map, axis=2, output_type=tf.int32)
    # IoU map has iou values for every gt boxes and we merge these values column wise
    merged_iou_map = tf.reduce_max(iou_map, axis=2)
    #
    pos_cond = tf.greater(merged_iou_map, iou_threshold)
    #
    gt_boxes_map = tf.gather(gt_boxes, max_indices_each_gt_box, batch_dims=1)
    expanded_gt_boxes = tf.where(tf.expand_dims(pos_cond, -1), gt_boxes_map, tf.zeros_like(gt_boxes_map))
    bbox_deltas = bbox_utils.get_deltas_from_bboxes(prior_boxes, expanded_gt_boxes) / variances
    #
    gt_labels_map = tf.gather(gt_labels, max_indices_each_gt_box, batch_dims=1)
    expanded_gt_labels = tf.where(pos_cond, gt_labels_map, tf.zeros_like(gt_labels_map))
    bbox_labels = tf.one_hot(expanded_gt_labels, total_labels)
    #
    return bbox_deltas, bbox_labels
