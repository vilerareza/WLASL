import math
import os
import argparse

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torchvision import transforms
import videotransforms

import numpy as np

import torch.nn.functional as F
from pytorch_i3d import InceptionI3d

# from nslt_dataset_all import NSLT as Dataset
from datasets.nslt_dataset_all import NSLT as Dataset
import cv2

from sklearn.metrics import classification_report

import csv


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str)
parser.add_argument('--mode', type=str, help='rgb or flow')
parser.add_argument('--train_split', type=str, default='./preprocess/nslt_100.json')
parser.add_argument('--config_file', type=str, default='./configfiles/asl100.ini')
parser.add_argument('--save_model', type=str, default='./checkpoints/')
parser.add_argument('--weights_dir', type=str, default = './weights/')
parser.add_argument('--weights', type=str, default='archived/asl100/FINAL_nslt_100_iters=896_top1=65.89_top5=84.11_top10=89.92.pt')
parser.add_argument('--num_classes', type=int, default=100)

args = parser.parse_args()


def load_rgb_frames_from_video(video_path, start=0, num=-1):
    vidcap = cv2.VideoCapture(video_path)
    # vidcap = cv2.VideoCapture('/home/dxli/Desktop/dm_256.mp4')

    frames = []

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
    if num == -1:
        num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    for offset in range(num):
        
        success, img = vidcap.read()

        if not success:
            continue

        w, h, c = img.shape
        sc = 224 / w
        img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

        img = (img / 255.) * 2 - 1

        frames.append(img)

    return torch.Tensor(np.asarray(frames, dtype=np.float32))


def run(init_lr=0.1,
        num_classes=100,
        max_steps=64e3,
        mode='rgb',
        root='/ssd/Charades_v1_rgb',
        train_split='charades/charades.json',
        batch_size=3 * 15,
        save_model='',
        weights=None,
        weights_dir=''):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    val_dataset = Dataset(train_split, 'test', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                 shuffle=False, num_workers=2,
                                                 pin_memory=False)

    dataloaders = {'test': val_dataloader}
    datasets = {'test': val_dataset}

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        # RV: Change with CLI weight arg
        # i3d.load_state_dict(torch.load('weights/flow_imagenet.pt'))
        i3d.load_state_dict(torch.load(os.path.join(weights_dir, 'flow_imagenet.pt')))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        # RV: Disabled. change with CLI weight arg
        # i3d.load_state_dict(torch.load('weights/rgb_imagenet.pt'))
        i3d.load_state_dict(torch.load(os.path.join(weights_dir, 'rgb_imagenet.pt')))
        
    i3d.replace_logits(num_classes)
    i3d.load_state_dict(torch.load(weights))  # nslt_2000_000700.pt nslt_1000_010800 nslt_300_005100.pt(best_results)  nslt_300_005500.pt(results_reported) nslt_2000_011400
    
    # RV: Add option for CPU
    if device == 'cuda':
        i3d.cuda()
    else:
        i3d.cpu()

    i3d = nn.DataParallel(i3d)
    i3d.eval()

    correct = 0
    correct_5 = 0
    correct_10 = 0

    top1_fp = np.zeros(num_classes, dtype=int)
    top1_tp = np.zeros(num_classes, dtype=np.int32)

    top5_fp = np.zeros(num_classes, dtype=np.int32)
    top5_tp = np.zeros(num_classes, dtype=np.int32)

    top10_fp = np.zeros(num_classes, dtype=np.int32)
    top10_tp = np.zeros(num_classes, dtype=np.int32)

    predictions_all = []
    labels_all = []

    for data in dataloaders["test"]:
        inputs, labels, video_id = data  # inputs: b, c, t, h, w

        per_frame_logits = i3d(inputs)

        predictions = torch.max(per_frame_logits, dim=2)[0]
        out_labels = np.argsort(predictions.cpu().detach().numpy()[0])
        out_probs = np.sort(predictions.cpu().detach().numpy()[0])

        if labels[0].item() in out_labels[-5:]:
            correct_5 += 1
            top5_tp[labels[0].item()] += 1
        else:
            top5_fp[labels[0].item()] += 1
        if labels[0].item() in out_labels[-10:]:
            correct_10 += 1
            top10_tp[labels[0].item()] += 1
        else:
            top10_fp[labels[0].item()] += 1
        if torch.argmax(predictions[0]).item() == labels[0].item():
            correct += 1
            top1_tp[labels[0].item()] += 1
        else:
            top1_fp[labels[0].item()] += 1
        # print(video_id, float(correct) / len(dataloaders["test"]), float(correct_5) / len(dataloaders["test"]),
        #       float(correct_10) / len(dataloaders["test"]))

        predictions_all.append(torch.argmax(predictions[0]).item())
        labels_all.append(labels[0].item())

    # RV: Eliminate divide by 0
    # top1_per_class = np.mean(top1_tp / (top1_tp + top1_fp), where=(top1_tp + top1_fp)!=0)
    # top5_per_class = np.mean(top5_tp / (top5_tp + top5_fp), where=(top5_tp + top5_fp)!=0)
    # top10_per_class = np.mean(top10_tp / (top10_tp + top10_fp), where=(top10_tp + top10_fp)!=0)

    # print('top-k average per class acc: {}, {}, {}'.format(top1_per_class, top5_per_class, top10_per_class))

    print_metrics(predictions_all, labels_all)


# RV: Print metrics
def print_metrics(predictions, labels):

    print(classification_report(labels, predictions, zero_division=0))
    
    # Create csv report
    report = classification_report(labels, predictions, zero_division=0, output_dict=True)
    
    with open('report_i3d_test.csv', 'w', newline='') as csvfile:
        fieldnames = ['class_id', 'precision', 'recall', 'f1-score', 'support']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for class_id in report.keys():

            if class_id == 'accuracy': 
                writer.writerow({})
                writer.writerow({'class_id': class_id, 
                                'f1-score': round(report[class_id], 2)})
            
            else:
                writer.writerow({'class_id': class_id, 
                                'precision': round(report[class_id]['precision'], 2),
                                'recall': round(report[class_id]['recall'], 2),
                                'f1-score': round(report[class_id]['f1-score'], 2),
                                'support': round(report[class_id]['support'], 2)})
    


def ensemble(mode, root, train_split, weights, num_classes):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    # test_transforms = transforms.Compose([])

    val_dataset = Dataset(train_split, 'test', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                 shuffle=False, num_workers=2,
                                                 pin_memory=False)

    dataloaders = {'test': val_dataloader}
    datasets = {'test': val_dataset}

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('weights/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('weights/rgb_imagenet.pt'))
    i3d.replace_logits(num_classes)
    i3d.load_state_dict(torch.load(weights))  # nslt_2000_000700.pt nslt_1000_010800 nslt_300_005100.pt(best_results)  nslt_300_005500.pt(results_reported) nslt_2000_011400
    
    # RV: Add option for CPU
    if device == 'cuda':
        i3d.cuda()
    else:
        i3d.cpu()

    i3d = nn.DataParallel(i3d)
    i3d.eval()

    correct = 0
    correct_5 = 0
    correct_10 = 0
    # confusion_matrix = np.zeros((num_classes,num_classes), dtype=np.int32)

    top1_fp = np.zeros(num_classes, dtype=np.int32)
    top1_tp = np.zeros(num_classes, dtype=np.int32)

    top5_fp = np.zeros(num_classes, dtype=np.int32)
    top5_tp = np.zeros(num_classes, dtype=np.int32)

    top10_fp = np.zeros(num_classes, dtype=np.int32)
    top10_tp = np.zeros(num_classes, dtype=np.int32)

    for data in dataloaders["test"]:
        inputs, labels, video_id = data  # inputs: b, c, t, h, w

        t = inputs.size(2)
        num = 64
        if t > num:
            num_segments = math.floor(t / num)

            segments = []
            for k in range(num_segments):
                segments.append(inputs[:, :, k*num: (k+1)*num, :, :])

            segments = torch.cat(segments, dim=0)
            per_frame_logits = i3d(segments)

            predictions = torch.mean(per_frame_logits, dim=2)

            if predictions.shape[0] > 1:
                predictions = torch.mean(predictions, dim=0)

        else:
            per_frame_logits = i3d(inputs)
            predictions = torch.mean(per_frame_logits, dim=2)[0]

        out_labels = np.argsort(predictions.cpu().detach().numpy())

        if labels[0].item() in out_labels[-5:]:
            correct_5 += 1
            top5_tp[labels[0].item()] += 1
        else:
            top5_fp[labels[0].item()] += 1
        if labels[0].item() in out_labels[-10:]:
            correct_10 += 1
            top10_tp[labels[0].item()] += 1
        else:
            top10_fp[labels[0].item()] += 1
        if torch.argmax(predictions).item() == labels[0].item():
            correct += 1
            top1_tp[labels[0].item()] += 1
        else:
            top1_fp[labels[0].item()] += 1
        #print(video_id, float(correct) / len(dataloaders["test"]), float(correct_5) / len(dataloaders["test"]),
        #      float(correct_10) / len(dataloaders["test"]))

    # RV: Eliminate divide by 0
    top1_per_class = np.mean(top1_tp / (top1_tp + top1_fp), where=(top1_tp + top1_fp)!=0)
    top5_per_class = np.mean(top5_tp / (top5_tp + top5_fp), where=(top5_tp + top5_fp)!=0)
    top10_per_class = np.mean(top10_tp / (top10_tp + top10_fp), where=(top10_tp + top10_fp)!=0)

    print('top-k average per class acc: {}, {}, {}'.format(top1_per_class, top5_per_class, top10_per_class))


def run_on_tensor(weights, ip_tensor, num_classes):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    i3d = InceptionI3d(400, in_channels=3)
    # i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))

    i3d.replace_logits(num_classes)
    i3d.load_state_dict(torch.load(weights))  # nslt_2000_000700.pt nslt_1000_010800 nslt_300_005100.pt(best_results)  nslt_300_005500.pt(results_reported) nslt_2000_011400
    
    # RV: Add option for CPU
    if device == 'cuda':
        i3d.cuda()
    else:
        i3d.cpu()

    i3d = nn.DataParallel(i3d)
    i3d.eval()

    t = ip_tensor.shape[2]

    # RV: Add option for CPU
    if device == 'cuda':
        ip_tensor.cuda()
    else:
        ip_tensor.cpu()
        
    per_frame_logits = i3d(ip_tensor)

    predictions = F.interpolate(per_frame_logits, t, mode='linear')

    predictions = predictions.transpose(2, 1)
    out_labels = np.argsort(predictions.cpu().detach().numpy()[0])

    arr = predictions.cpu().detach().numpy()[0,:,0].T

    plt.plot(range(len(arr)), F.softmax(torch.from_numpy(arr), dim=0).numpy())
    plt.show()

    return out_labels


def get_slide_windows(frames, window_size, stride=1):
    indices = torch.arange(0, frames.shape[0])
    window_indices = indices.unfold(0, window_size, stride)

    return frames[window_indices, :, :, :].transpose(1, 2)


if __name__ == '__main__':
    # ================== test i3d on a dataset ==============
    # need to add argparse
    mode = 'rgb'

    # RV: Changed to CLI arg
    # num_classes = 2000
    num_classes = args.num_classes

    # RV: Changed to CLI arg
    # root = '../../data/WLASL2000'
    root = args.root

    # RV: Changed to CLI arg
    # save_model = './checkpoints/'
    save_model = args.save_model

    # RV: Changed to CLI arg
    # train_split = 'preprocess/nslt_{}.json'.format(num_classes)
    train_split = args.train_split

    # RV: Changed to CLI arg
    # weights = 'archived/asl100/FINAL_nslt_100_iters=896_top1=65.89_top5=84.11_top10=89.92.pt'
    weights = args.weights

    # RV: Changed to CLI arg
    weights_dir = args.weights_dir

    run(mode=mode, num_classes=num_classes, root=root, save_model=save_model, train_split=train_split, weights=weights, weights_dir=weights_dir)
