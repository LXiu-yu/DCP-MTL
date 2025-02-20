import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))


def calc_loss(prediction, target, bce_weight=0.5):
    """Calculating the loss and metrics
    Args:
        prediction = predicted image
        target = Targeted image
        metrics = Metrics printed
        bce_weight = 0.5 (default)
    Output:
        loss : dice loss of the epoch """
    bce = F.binary_cross_entropy_with_logits(prediction, target)
    prediction = torch.sigmoid(prediction)
    dice = dice_loss(prediction, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss





class log_cosh_dice_loss(nn.Module):
    def __init__(self, num_classes=1, smooth=1, alpha=0.7):
        super(log_cosh_dice_loss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, outputs, targets):
        x = self.dice_loss(outputs, targets)
        return torch.log((torch.exp(x) + torch.exp(-x)) / 2.0)

    def dice_loss(self, y_pred, y_true):
        """[function to compute dice loss]
        Args:
            y_true ([float32]): [ground truth image]
            y_pred ([float32]): [predicted image]
        Returns:
            [float32]: [loss value]
        """
        smooth = 1.
        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(y_pred)
        intersection = torch.sum((y_true * y_pred))
        coeff = (2. * intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)
        return (1. - coeff)


def focal_loss(predict, label, alpha=0.6, beta=2):
    probs = torch.sigmoid(predict)
    # 交叉熵Loss
    ce_loss = nn.BCELoss()
    ce_loss = ce_loss(probs,label)
    alpha_ = torch.ones_like(predict) * alpha
    # 正label 为alpha, 负label为1-alpha
    alpha_ = torch.where(label > 0, alpha_, 1.0 - alpha_)
    probs_ = torch.where(label > 0, probs, 1.0 - probs)
    # loss weight matrix
    loss_matrix = alpha_ * torch.pow((1.0 - probs_), beta)
    # 最终loss 矩阵，为对应的权重与loss值相乘，控制预测越不准的产生更大的loss
    loss = loss_matrix * ce_loss
    loss = torch.sum(loss)
    return loss



class Loss:
    def __init__(self, dice_weight=0.0, class_weights=None, num_classes=1, device=None):
        self.device = device
        if class_weights is not None:
            nll_weight = torch.from_numpy(class_weights.astype(np.float32)).to(
                self.device
            )
        else:
            nll_weight = None
        self.nll_loss = nn.NLLLoss2d(weight=nll_weight)
        self.dice_weight = dice_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)
        if self.dice_weight:
            eps = 1e-7
            cls_weight = self.dice_weight / self.num_classes
            for cls in range(self.num_classes):
                dice_target = (targets == cls).float()
                dice_output = outputs[:, cls].exp()
                intersection = (dice_output * dice_target).sum()
                # union without intersection
                uwi = dice_output.sum() + dice_target.sum() + eps
                loss += (1 - intersection / uwi) * cls_weight
            loss /= (1 + self.dice_weight)
        return loss


class LossMulti:
    def __init__(
            self, jaccard_weight=0.0, class_weights=None, num_classes=1, device=None
    ):
        self.device = device
        if class_weights is not None:
            nll_weight = torch.from_numpy(class_weights.astype(np.float32)).to(
                self.device
            )
        else:
            nll_weight = None

        self.nll_loss = nn.NLLLoss(weight=nll_weight)
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):

        targets = targets.squeeze(1)
        # 计算负对数似然损失
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-7  # 原先是1e-7
            for cls in range(self.num_classes):
                jaccard_target = (targets == cls).float()
                jaccard_output = outputs[:, cls].exp()
                intersection = (jaccard_output * jaccard_target).sum()

                union = jaccard_output.sum() + jaccard_target.sum()
                loss -= (
                        torch.log((intersection + eps) / (union - intersection + eps))
                        * self.jaccard_weight
                )
        return loss

def dice_loss_func(input, target):
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) / (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()


def criterion_bcedice(inputs, target, loss_weight=torch.tensor(1), dice: bool = True, size=512):
    bcecriterion = nn.BCEWithLogitsLoss(pos_weight=loss_weight)
    if size == 512:
        loss = bcecriterion(inputs.squeeze(), target.squeeze().float())
    else:
        if len(target.shape) == 3:
            target = target.unsqueeze(1)
        target = F.interpolate(target, mode='bilinear', size=(size, size))
        loss = bcecriterion(inputs.squeeze(), target.squeeze().float())
    if dice is True:
        loss += dice_loss_func(torch.sigmoid(inputs.squeeze()), target.squeeze().float())
    return loss





class LossBsiNet:
    def __init__(self, weights=[1, 1, 1]):
        self.criterion1 = LossMulti(num_classes=2)   #mask_loss
        self.criterion2 = LossMulti(num_classes=2)   #contour_loss
        self.criterion3 = nn.MSELoss()               #distance_loss
        self.weights = weights

    def __call__(self, outputs0, outputs1, outputs2, outputs3, outputs4, outputs5, outputs6, outputs7, outputs8, outputs9, outputs10, targets1, targets2, targets3):
        criterion = (
                self.weights[0] * self.criterion1(outputs0, targets1)
                + self.weights[1] * self.criterion2(outputs1, targets2)
                + self.weights[1] * self.criterion3(outputs2, targets3)
                + 0.4 * self.criterion1(outputs3, targets1)
                + 0.4 * self.criterion1(outputs4, targets1)
                + 0.2 * self.criterion1(outputs5, targets1)
                + 0.2 * self.criterion1(outputs6, targets1)
                + 0.4 * self.criterion2(outputs7, targets2)
                + 0.4 * self.criterion2(outputs8, targets2)
                + 0.2 * self.criterion2(outputs9, targets2)
                + 0.2 * self.criterion2(outputs10, targets2)
        )


        return criterion


 
class Loss_v9:
    def __init__(self, weights=[1, 1, 1]):
        # self.criterion1 = criterion()                # mask_loss
        # self.criterion2 = criterion()                # contour_loss
        self.criterion3 = nn.MSELoss()               # distance_loss
        self.weights = weights
    

    def __call__(self, outputs0, outputs1, outputs2, outputs3, outputs4, outputs5, outputs6, outputs7, outputs8, outputs9, outputs10, outputs11, outputs12, outputs13, outputs14, targets1, targets2, targets3):
        

        
        criterion = (
                1 * criterion_bcedice(outputs0, targets1, dice=True)
                + 1 * criterion_bcedice(outputs1, targets2, loss_weight=torch.tensor(9), dice=True)
                + 1 * self.criterion3(outputs2, targets3)
                + 0.5 * criterion_bcedice(outputs3, targets1, dice=True)
                + 0.5 * criterion_bcedice(outputs4, targets1, dice=True)
                + 0.3 * criterion_bcedice(outputs5, targets1, dice=True)
                + 0.3 * criterion_bcedice(outputs6, targets1, dice=True)
                + 0.5 * criterion_bcedice(outputs7, targets2, loss_weight=torch.tensor(3),dice=True)
                + 0.5 * criterion_bcedice(outputs8, targets2, loss_weight=torch.tensor(3),dice=True)
                + 0.3 * criterion_bcedice(outputs9, targets2, loss_weight=torch.tensor(3),dice=True)
                + 0.3 * criterion_bcedice(outputs10, targets2,loss_weight=torch.tensor(3),dice=True)
                + 0.5 * self.criterion3(outputs11, targets3)
                + 0.5 * self.criterion3(outputs12, targets3)
                + 0.3 * self.criterion3(outputs13, targets3)
                + 0.3 * self.criterion3(outputs14, targets3)
                
        )

        return criterion

class Loss_v10:
    def __init__(self, weights=[1, 1, 1]):
        # self.criterion1 = criterion()                # mask_loss
        # self.criterion2 = criterion()                # contour_loss
        self.criterion3 = nn.MSELoss()               # distance_loss
        self.weights = weights
    

    def __call__(self, outputs0, outputs1, outputs2, outputs3, outputs4, outputs5, outputs6, outputs7, outputs8, outputs9, outputs10, outputs11, outputs12, outputs13, outputs14, targets1, targets2, targets3):
        

        
        criterion = (
                1 * criterion_bcedice(outputs0, targets1, dice=True)
                + 2 * criterion_bcedice(outputs1, targets2, loss_weight=torch.tensor(9), dice=True)
                + 1 * self.criterion3(outputs2, targets3)
                # + 0.5 * criterion_bcedice(outputs3, targets1, dice=True)
                # + 0.5 * criterion_bcedice(outputs4, targets1, dice=True)
                # + 0.3 * criterion_bcedice(outputs5, targets1, dice=True)
                # + 0.3 * criterion_bcedice(outputs6, targets1, dice=True)
                # + 0.5 * criterion_bcedice(outputs7, targets2, loss_weight=torch.tensor(3),dice=True)
                # + 0.5 * criterion_bcedice(outputs8, targets2, loss_weight=torch.tensor(3),dice=True)
                # + 0.3 * criterion_bcedice(outputs9, targets2, loss_weight=torch.tensor(3),dice=True)
                # + 0.3 * criterion_bcedice(outputs10, targets2,loss_weight=torch.tensor(3),dice=True)
                # + 0.5 * self.criterion3(outputs11, targets3)
                # + 0.5 * self.criterion3(outputs12, targets3)
                # + 0.3 * self.criterion3(outputs13, targets3)
                # + 0.3 * self.criterion3(outputs14, targets3)
                
        )

        return criterion

class Loss_v14:
    def __init__(self, weights=[1, 1, 1]):
        # self.criterion1 = criterion()                # mask_loss
        # self.criterion2 = criterion()                # contour_loss
        self.criterion3 = nn.MSELoss()               # distance_loss
        self.weights = weights
    

    def __call__(self, outputs0, outputs1, outputs2, outputs3, outputs4, outputs5, outputs6, outputs7, outputs8, outputs9, outputs10, outputs11, targets1, targets2, targets3):
        

        
        criterion = (
                1 * criterion_bcedice(outputs0, targets1, dice=True)
                + 2 * criterion_bcedice(outputs1, targets2, loss_weight=torch.tensor(9), dice=True)
                + 1 * self.criterion3(outputs2, targets3)
                + 0.5 * criterion_bcedice(outputs3, targets1, dice=True)
                + 0.5 * criterion_bcedice(outputs4, targets1, dice=True)
                + 0.3 * criterion_bcedice(outputs5, targets1, dice=True)
                + 0.5 * criterion_bcedice(outputs6, targets2, loss_weight=torch.tensor(3),dice=True)
                + 0.5 * criterion_bcedice(outputs7, targets2, loss_weight=torch.tensor(3),dice=True)
                + 0.3 * criterion_bcedice(outputs8, targets2, loss_weight=torch.tensor(3),dice=True)
                + 0.5 * self.criterion3(outputs9, targets3)
                + 0.5 * self.criterion3(outputs10, targets3)
                + 0.3 * self.criterion3(outputs11, targets3)
        )

        return criterion
