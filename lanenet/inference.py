import time
import os
import sys

from tqdm import tqdm

import torch
from dataloader.data_loaders import LaneDataSet, UnlabelledDataSet
from dataloader.transformers import Rescale
from model.model import LaneNet, compute_loss
from torch.utils.data import DataLoader
from torch.autograd import Variable

from torchvision import transforms

from utils.cli_helper import parse_args
from utils.average_meter import AverageMeter
from test import test

import numpy as np
import cv2
import matplotlib.pyplot as plt

# might want this in the transformer part as well
VGG_MEAN = [103.939, 116.779, 123.68]
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def compose_img(image_data, out, binary_label, pix_embedding, instance_label, i):
    val_gt = (image_data[i].cpu().numpy().transpose(1, 2, 0) + VGG_MEAN).astype(np.uint8)
    val_pred = out[i].squeeze(0).cpu().numpy().transpose(0, 1) * 255
    val_label = binary_label[i].squeeze(0).cpu().numpy().transpose(0, 1) * 255
    val_out = np.zeros((val_pred.shape[0], val_pred.shape[1], 3), dtype=np.uint8)
    val_out[:, :, 0] = val_pred
    val_out[:, :, 1] = val_label
    val_gt[val_out == 255] = 255
    # epsilon = 1e-5
    # pix_embedding = pix_embedding[i].data.cpu().numpy()
    # pix_vec = pix_embedding / (np.sum(pix_embedding, axis=0, keepdims=True) + epsilon) * 255
    # pix_vec = np.round(pix_vec).astype(np.uint8).transpose(1, 2, 0)
    # ins_label = instance_label[i].data.cpu().numpy().transpose(0, 1)
    # ins_label = np.repeat(np.expand_dims(ins_label, -1), 3, -1)
    # val_img = np.concatenate((val_gt, pix_vec, ins_label), axis=0)
    # val_img = np.concatenate((val_gt, pix_vec), axis=0)
    # return val_img
    return val_gt


def save_model(save_path, epoch, model):
    save_name = os.path.join(save_path, f'{epoch}_checkpoint.pth')
    torch.save(model, save_name)
    print("model is saved: {}".format(save_name))


def main():
    args = parse_args()

    save_path = args.save

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # train_dataset_file = os.path.join(args.dataset, 'train.txt')
    val_dataset_file = os.path.join(args.dataset, 'val.txt')

    # train_dataset = LaneDataSet(train_dataset_file, transform=transforms.Compose([Rescale((512, 256))]))
    # train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)

    if args.val:
        val_dataset = UnlabelledDataSet(val_dataset_file, transform=transforms.Compose([Rescale((512, 256))]))
        val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=True)

    # model = LaneNet()
    model = torch.load(args.pretrained)
    model = model.to(DEVICE)
    model.eval()

    print(model)
    for batch_idx, input_data in enumerate(val_loader):
        print(input_data[0].shape)
        img = np.transpose(input_data[0].numpy(), (1, 2, 0))    
        image_data = Variable(input_data[0]).type(torch.FloatTensor).to(DEVICE)
        image_data = image_data.unsqueeze(0)
        net_output = model(image_data)
        instance_seg_logits = net_output['instance_seg_logits'].squeeze()
        instance_seg_pred = np.argmax(instance_seg_logits.cpu().detach().numpy(), axis=0)
        instance_seg_pred = instance_seg_pred.squeeze().astype(np.uint8)

        binary_seg_logits = net_output['binary_seg_logits'].squeeze()
        binary_seg_pred = np.argmax(binary_seg_logits.cpu().detach().numpy(), axis=0)
        binary_seg_pred = binary_seg_pred.squeeze().astype(np.uint8)
        # binary_seg_pred = net_output['binary_seg_pred'].squeeze().cpu().numpy()

        plt.subplot(131)
        orig_img = image_data.squeeze().cpu().numpy()
        plt.imshow(np.transpose(orig_img.astype(np.uint8), (1, 2, 0)))
        plt.title('original')

        plt.subplot(132)
        plt.imshow(binary_seg_pred, 'gray')
        plt.title('binary')

        plt.subplot(133)
        plt.imshow(instance_seg_pred, 'gray')
        plt.title('instance')
        plt.show()




    # for epoch in range(0, args.epochs):
    #     print(f"Epoch {epoch}")
    #     train_iou = train(train_loader, model, optimizer, epoch)
    #     if args.val:
    #         val_iou = test(val_loader, model, epoch)
    #     if (epoch + 1) % 5 == 0:
    #         save_model(save_path, epoch, model)

    #     print(f"Train IoU : {train_iou}")
    #     if args.val:
    #         print(f"Val IoU : {val_iou}")


if __name__ == '__main__':
    main()
