import torch
from tqdm import tqdm
import numpy as np
import torchvision
from torch.nn import functional as F
import time
import argparse
from skimage import morphology
import torch.nn as nn

img_size = 512

def fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


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


def evaluate(device, epoch, model, data_loader, writer):

    model.eval()
    losses = []
    hist = 0
    hist_bd = 0
    start = time.perf_counter()
    with torch.no_grad():

        for iter, data in enumerate(tqdm(data_loader)):

            _, inputs, targets, bd_targets,_ = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = F.nll_loss(outputs[0], targets.squeeze(1))
            # loss = criterion_bcedice(outputs[0], targets, dice=True)
            losses.append(loss.item())
            # print("outputs.shape",len(outputs))
            # print("outputs.shape",outputs[0].shape,outputs[1].shape,outputs[3].shape,outputs[4].shape)
            # print("targets.shape",targets.shape)

            # 定义h和w
            h, w = outputs[0].shape[-2:]
            # print('h, w',h, w)

            outputs_mask = outputs[0].detach().cpu().numpy().squeeze()
            targets = targets.detach().cpu().numpy().squeeze()
            res = np.zeros((h, w))
            indices = np.argmax(outputs_mask, axis=0)
            res[indices == 1] = 1
            res[indices == 0] = 0
            result_mask = np.array(res, dtype='uint8')  # 转变为8字节型
            hist+=fast_hist(result_mask.flatten(),targets.flatten(),2)

            # 
            outputs_bd = outputs[0].detach().cpu().numpy().squeeze()
            bd_targets = bd_targets.detach().cpu().numpy().squeeze()
            res_bd = np.zeros((h, w))
            indices_bd = np.argmax(outputs_bd, axis=0)
            res_bd[indices_bd == 1] = 1
            res_bd[indices_bd == 0] = 0
            result_bd = np.array(res_bd, dtype='uint8')  # 转变为8字节型
            hist_bd+=fast_hist(result_bd.flatten(),bd_targets.flatten(),2)

        writer.add_scalar("Dev_Loss", np.mean(losses), epoch)

        IOU=(np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)))[-1]
        acc_global_OA = np.diag(hist).sum() / (hist_bd.sum() + eps)
        acc_R = np.diag(hist) / ((hist_bd.sum(1) * 100) + eps)
        acc_P = np.diag(hist) / ((hist_bd.sum(0) * 100) + eps)
        F1score = 2 * acc_R * acc_P / (acc_R + acc_P)

        print('-----------region-----------')
        print('IOU:', IOU)
        print('OA:', acc_global_OA)
        print('Recall:', acc_R)
        print('Precision:', acc_P)
        print('F1_score:', F1score)    
        print('hist',hist)

        # 
        print('-----------boundary-----------')
        eps = 0.000001
        IOU_bd=(np.diag(hist_bd) / (hist_bd.sum(axis=1) + hist_bd.sum(axis=0) - np.diag(hist_bd)))[-1]
        acc_global_OA_bd = np.diag(hist_bd).sum() / (hist_bd.sum() + eps)
        acc_R_bd = np.diag(hist_bd) / ((hist_bd.sum(1) * 100) + eps)
        acc_P_bd = np.diag(hist_bd) / ((hist_bd.sum(0) * 100) + eps)
        F1score_bd = 2 * acc_R * acc_P / (acc_R + acc_P + eps)
        print('IOU_bd:', IOU_bd)
        print('OA_bd:', acc_global_OA_bd)
        print('Recall_bd:', acc_R_bd)
        print('Precision_bd:', acc_P_bd)
        print('F1_score_bd:', F1score_bd)    
        print('hist_bd',hist_bd)


    return np.mean(losses), time.perf_counter() - start

def evaluate_dcpv9(device, epoch, model, data_loader, writer):

    model.eval()
    losses = []
    hist = 0
    hist_bd = 0
    start = time.perf_counter()
    with torch.no_grad():

        for iter, data in enumerate(tqdm(data_loader)):


            _, inputs, targets, bd_targets,_ = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            
            # loss = F.nll_loss(outputs[0], targets.squeeze(1))
            loss = criterion_bcedice(outputs[0], targets, dice=True)

            losses.append(loss.item())
            # print("outputs.shape",len(outputs))
            # print("outputs.shape",outputs[0].shape,outputs[1].shape,outputs[3].shape,outputs[4].shape)
            # print("targets.shape",targets.shape)

            # 
            h, w = outputs[0].shape[-2:]
            # print('h, w',h, w)

            outputs_mask = outputs[0].detach().cpu().numpy().squeeze()
            targets = targets.detach().cpu().numpy().squeeze()

            outputs_mask = 1 / (1 + np.exp(-outputs_mask))
            result_mask = (outputs_mask > 0.5)
            hist+=fast_hist(result_mask.flatten(),targets.flatten(),2)

            # 
            outputs_bd = outputs[1].detach().cpu().numpy().squeeze()
            bd_targets = bd_targets.detach().cpu().numpy().squeeze()

            outputs_bd = 1 / (1 + np.exp(-outputs_bd))
            result_mask = (outputs_bd > 0.5)

            result_bd = np.array(result_mask, dtype='uint8')  
            hist_bd+=fast_hist(result_bd.flatten(),bd_targets.flatten(),2)

        writer.add_scalar("Dev_Loss", np.mean(losses), epoch)

        eps = 0.000001

        IOU=(np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)))[-1]
        acc_global_OA = np.diag(hist).sum() / (hist_bd.sum() + eps)
        acc_R = np.diag(hist) / ((hist_bd.sum(1) * 100) + eps)
        acc_P = np.diag(hist) / ((hist_bd.sum(0) * 100) + eps)
        F1score = 2 * acc_R * acc_P / (acc_R + acc_P)

        print('-----------region-----------')
        print('IOU:', IOU)
        print('OA:', acc_global_OA)
        print('Recall:', acc_R)
        print('Precision:', acc_P)
        print('F1_score:', F1score)    
        print('hist',hist)

        # 
        print('-----------boundary-----------')
        
        IOU_bd=(np.diag(hist_bd) / (hist_bd.sum(axis=1) + hist_bd.sum(axis=0) - np.diag(hist_bd)))[-1]
        acc_global_OA_bd = np.diag(hist_bd).sum() / (hist_bd.sum() + eps)
        acc_R_bd = np.diag(hist_bd) / ((hist_bd.sum(1) * 100) + eps)
        acc_P_bd = np.diag(hist_bd) / ((hist_bd.sum(0) * 100) + eps)
        F1score_bd = 2 * acc_R * acc_P / (acc_R + acc_P + eps)
        print('IOU_bd:', IOU_bd)
        print('OA_bd:', acc_global_OA_bd)
        print('Recall_bd:', acc_R_bd)
        print('Precision_bd:', acc_P_bd)
        print('F1_score_bd:', F1score_bd)    
        print('hist_bd',hist_bd)


    return np.mean(losses), time.perf_counter() - start





def visualize(device, epoch, model, data_loader, writer, val_batch_size, train=True):
    def save_image(image, tag, val_batch_size):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(
            image, nrow=int(np.sqrt(val_batch_size)), pad_value=0, padding=25
        )
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            _, inputs, targets, _,_ = data

            inputs = inputs.to(device)

            targets = targets.to(device)
            outputs = model(inputs)

            output_mask = outputs[0].detach().cpu().numpy()
            output_final = np.argmax(output_mask, axis=1).astype(float)
            output_final = torch.from_numpy(output_final).unsqueeze(1)

            if train == "True":
                save_image(targets.float(), "Target_train",val_batch_size)
                save_image(output_final, "Prediction_train",val_batch_size)
            else:
                save_image(targets.float(), "Target", val_batch_size)
                save_image(output_final, "Prediction", val_batch_size)

            break


def create_train_arg_parser():

    parser = argparse.ArgumentParser(description="train setup for segmentation")
    parser.add_argument("--train_path", type=str, help="path to img tif files")
    parser.add_argument("--val_path", type=str, help="path to img tif files")
    parser.add_argument(
        "--model_type",
        type=str,
        help=" ",
    )
    parser.add_argument("--object_type", type=str, help="Dataset.")
    parser.add_argument(
        "--distance_type",
        type=str,
        default="dist_contour",
        help="select distance transform type - dist_mask,dist_contour,dist_contour_tif",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="train batch size")
    parser.add_argument(
        "--val_batch_size", type=int, default=4, help="validation batch size"
    )
    parser.add_argument("--num_epochs", type=int, default=400, help="number of epochs")
    parser.add_argument("--cuda_no", type=int, default=0, help="cuda number")
    parser.add_argument(
        "--use_pretrained", type=bool, default=False, help="Load pretrained checkpoint."
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="If use_pretrained is true, provide checkpoint.",
    )
    parser.add_argument("--save_path", type=str, help="Model save path.")
    parser.add_argument("--img_size", type=int, default = 512 ,help="Model save path.")
    return parser


def create_validation_arg_parser():

    parser = argparse.ArgumentParser(description="train setup for segmentation")
    parser.add_argument(
        "--model_type",
        type=str,
        help=" ",
    )
    parser.add_argument("--test_path", type=str, help="path to img tif files")
    parser.add_argument("--model_file", type=str, help="model_file")
    parser.add_argument("--save_path", type=str, help="results save path.")
    parser.add_argument("--cuda_no", type=int, default=0, help="cuda number")

    return parser


