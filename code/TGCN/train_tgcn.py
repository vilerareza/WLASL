import logging
import os

import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset

import utils
from configs import Config
from tgcn_model import GCN_muti_att
from sign_dataset import Sign_Dataset
from train_utils import train, validation

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str)
parser.add_argument('--subset', type=str, default='asl100')
parser.add_argument('--config_file', type=str, default='./configs/asl100.ini')
parser.add_argument('--split_file', type=str)
parser.add_argument('--pose_data', type=str)
parser.add_argument('--output_dir', type=str)

args = parser.parse_args()


def run(split_file, pose_data_root, configs, save_model_to=None, output_dir='./output'):


    epochs = configs.max_epochs
    log_interval = configs.log_interval
    num_samples = configs.num_samples
    hidden_size = configs.hidden_size
    drop_p = configs.drop_p
    num_stages = configs.num_stages

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # setup dataset
    train_dataset = Sign_Dataset(index_file_path=split_file, split=['train', 'val'], pose_root=pose_data_root,
                                 img_transforms=None, video_transforms=None, num_samples=num_samples)

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                                    shuffle=True)

    val_dataset = Sign_Dataset(index_file_path=split_file, split='test', pose_root=pose_data_root,
                               img_transforms=None, video_transforms=None,
                               num_samples=num_samples,
                               sample_strategy='k_copies')
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=configs.batch_size,
                                                  shuffle=True)

    logging.info('\n'.join(['Class labels are: '] + [(str(i) + ' - ' + label) for i, label in
                                                     enumerate(train_dataset.label_encoder.classes_)]))

    # setup the model

    # RV: Add option for CPU
    if device == 'cuda':
        model = GCN_muti_att(input_feature=num_samples*2, hidden_feature=num_samples*2,
                            num_class=len(train_dataset.label_encoder.classes_), p_dropout=drop_p, num_stage=num_stages).cuda()
    else:
        model = GCN_muti_att(input_feature=num_samples*2, hidden_feature=num_samples*2,
                            num_class=len(train_dataset.label_encoder.classes_), p_dropout=drop_p, num_stage=num_stages).cpu()

    # setup training parameters, learning rate, optimizer, scheduler
    lr = configs.init_lr
    # optimizer = optim.SGD(vgg_gru.parameters(), lr=lr, momentum=0.00001)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=configs.adam_eps, weight_decay=configs.adam_weight_decay)

    # record training process
    epoch_train_losses = []
    epoch_train_scores = []
    epoch_val_losses = []
    epoch_val_scores = []

    best_test_acc = 0
    # start training
    for epoch in range(int(epochs)):
        # train, test model

        print('start training.')
        train_losses, train_scores, train_gts, train_preds = train(log_interval, model,
                                                                   train_data_loader, optimizer, epoch)
        print('start testing.')
        val_loss, val_score, val_gts, val_preds, incorrect_samples = validation(model,
                                                                                val_data_loader, epoch,
                                                                                save_to=save_model_to)
        # print('start testing.')
        # val_loss, val_score, val_gts, val_preds, incorrect_samples = validation(model,
        #                                                                         val_data_loader, epoch,
        #                                                                         save_to=save_model_to)

        logging.info('========================\nEpoch: {} Average loss: {:.4f}'.format(epoch, val_loss))
        logging.info('Top-1 acc: {:.4f}'.format(100 * val_score[0]))
        logging.info('Top-3 acc: {:.4f}'.format(100 * val_score[1]))
        logging.info('Top-5 acc: {:.4f}'.format(100 * val_score[2]))
        logging.info('Top-10 acc: {:.4f}'.format(100 * val_score[3]))
        logging.info('Top-30 acc: {:.4f}'.format(100 * val_score[4]))
        logging.debug('mislabelled val. instances: ' + str(incorrect_samples))

        # save results
        epoch_train_losses.append(train_losses)
        epoch_train_scores.append(train_scores)
        epoch_val_losses.append(val_loss)
        epoch_val_scores.append(val_score[0])

        # save all train test results
        np.save(f'{output_dir}/epoch_training_losses.npy', np.array(epoch_train_losses))
        np.save(f'{output_dir}/epoch_training_scores.npy', np.array(epoch_train_scores))
        np.save(f'{output_dir}/epoch_test_loss.npy', np.array(epoch_val_losses))
        np.save(f'{output_dir}/epoch_test_score.npy', np.array(epoch_val_scores))


        if val_score[0] > best_test_acc:
            best_test_acc = val_score[0]
            best_epoch_num = epoch

            torch.save(model.state_dict(), os.path.join('checkpoints', subset, 'gcn_epoch={}_val_acc={}.pth'.format(
                best_epoch_num, best_test_acc)))

    utils.plot_curves()

    class_names = train_dataset.label_encoder.classes_

    # RV: Set output according to CLI
    utils.plot_confusion_matrix(train_gts, train_preds, classes=class_names, normalize=False,
                                save_to=f'{output_dir}/train-conf-mat')
    utils.plot_confusion_matrix(val_gts, val_preds, classes=class_names, normalize=False, save_to=f'{output_dir}/val-conf-mat')


if __name__ == "__main__":
    
    # RV: Change to CLI
    # root = '/media/anudisk/github/WLASL'
    root = args.root

    # RV: Change to CLI
    # subset = 'asl100'
    subset = args.subset

    # RV: Change to CLI
    # split_file = os.path.join(root, 'data/splits/{}.json'.format(subset))
    split_file = args.split_file

    # RV: Change to CLI
    # pose_data_root = os.path.join(root, 'data/pose_per_individual_videos')
    pose_data_root = args.pose_data

    # RV: Change to CLI
    # config_file = os.path.join(root, 'code/TGCN/configs/{}.ini'.format(subset))
    config_file = args.config_file
    configs = Config(config_file)

    # RV: Change to CLI
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(filename=f'{output_dir}/{os.path.basename(config_file)[:-4]}.log', level=logging.DEBUG, filemode='w+')

    logging.info('Calling main.run()')
    run(split_file=split_file, configs=configs, pose_data_root=pose_data_root, output_dir=output_dir)
    logging.info('Finished main.run()')
    # utils.plot_curves()
