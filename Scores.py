import numpy as np

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_coef_multilabel(y_true, y_pred, num_labels, flag=0):
    dice = 0
    for index in range(num_labels):
        if flag:
            print(dice_coef(y_true[:, :, index], y_pred[:, :, index]))
        dice += dice_coef(y_true[:, :, index], y_pred[:, :, index])
    return dice / num_labels  # taking average


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
