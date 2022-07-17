from __future__ import print_function

import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model

from deepsense_model import MyUTDmodel
import data_pre as data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=1,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')

    # model dataset
    parser.add_argument('--model', type=str, default='MyUTDmodel')
    parser.add_argument('--dataset', type=str, default='UTD-MHAD',
                        choices=['USC-HAR', 'UTD-MHAD', 'ours'], help='dataset')
    parser.add_argument('--num_class', type=int, default=27,
                        help='num_class')
    parser.add_argument('--num_train_basic', type=int, default=1,
                        help='num_train_basic')
    parser.add_argument('--num_test_basic', type=int, default=8,
                        help='num_test_basic')
    parser.add_argument('--label_rate', type=int, default=5,
                        help='label_rate')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='3',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.model_path = './save/deepsense/{}_models'.format(opt.dataset)
    opt.tb_path = './save/deepsense/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'deepsense_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):

    # load data (already normalized)
    num_of_train = (opt.num_train_basic * opt.label_rate * np.ones(opt.num_class)).astype(int)
    num_of_test = (opt.num_test_basic * np.ones(opt.num_class)).astype(int)

    #load labeled train and test data
    print("train labeled data:")
    x_train_label_1, x_train_label_2, y_train = data.load_data(opt.num_class, num_of_train, 1, opt.label_rate)
    print("test data:")
    x_test_1, x_test_2, y_test = data.load_data(opt.num_class, num_of_test, 2, opt.label_rate)

    train_dataset = data.Multimodal_dataset(x_train_label_1, x_train_label_2, y_train)
    test_dataset = data.Multimodal_dataset(x_test_1, x_test_2, y_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)

    return train_loader, val_loader


def set_model(opt):

    model = MyUTDmodel(input_size=1, num_classes=opt.num_class)
    criterion = torch.nn.CrossEntropyLoss()

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()


    for idx, (input_data1, input_data2, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            input_data1 = input_data1.cuda()
            input_data2 = input_data2.cuda()
            labels = labels.cuda()
        bsz = labels.shape[0]

        output = model(input_data1, input_data2)
        
        loss = criterion(output, labels)
        acc, _ = accuracy(output, labels, topk=(1, 5))

        # update metric
        losses.update(loss.item(), bsz)
        top1.update(acc[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()


    return losses.avg, top1.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    confusion = np.zeros((opt.num_class, opt.num_class))

    label_list = []
    pred_list = []

    with torch.no_grad():
        end = time.time()
        for idx, (input_data1, input_data2, labels) in enumerate(val_loader):

            if torch.cuda.is_available():
                input_data1 = input_data1.float().cuda()
                input_data2 = input_data2.float().cuda()
                labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = model(input_data1, input_data2)

            label_list.extend(labels.cpu().numpy())
            pred_list.extend(output.max(1)[1].cpu().numpy())


            loss = criterion(output, labels)

            rows = labels.cpu().numpy()
            cols = output.max(1)[1].cpu().numpy()

            for label_index in range(labels.shape[0]):
                confusion[rows[label_index], cols[label_index]] += 1

            acc, _ = accuracy(output, labels, topk=(1, 5))

            losses.update(loss.item(), bsz)
            top1.update(acc[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))


    F1score_test = f1_score(label_list, pred_list, average="macro") # macro sees all class with the same importance

    print(' * Acc@1 {top1.avg:.3f}\t'
        'F1-score {F1score_test:.3f}\t'.format(top1=top1, F1score_test=F1score_test))

    return losses.avg, top1.avg, confusion, F1score_test


def main():

    
    opt = parse_option()

    result_record = np.zeros((opt.trial, 3))

    for trial_id in range(opt.trial):

        # build data loader
        train_loader, val_loader = set_loader(opt)

        # build model and criterion
        model, criterion = set_model(opt)

        # build optimizer
        optimizer = set_optimizer(opt, model)

        # tensorboard
        logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

        record_acc = np.zeros(opt.epochs)

        best_acc = 0

        # training routine
        for epoch in range(1, opt.epochs + 1):
            adjust_learning_rate(opt, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, opt)
            time2 = time.time()

            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

            # tensorboard logger
            logger.log_value('train_loss', loss, epoch)
            logger.log_value('train_acc', train_acc, epoch)
            logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

            # evaluation
            loss, val_acc, confusion, val_F1score = validate(val_loader, model, criterion, opt)
            logger.log_value('val_loss', loss, epoch)
            logger.log_value('val_acc', val_acc, epoch)
            logger.log_value('val_f1', val_F1score, epoch)

            if val_acc > best_acc:
                best_acc = val_acc

            record_acc[epoch-1] = val_acc
            if epoch % opt.save_freq == 0:
                save_file = os.path.join(
                    opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                save_model(model, optimizer, opt, epoch, save_file)

        # save the last model
        save_file = os.path.join(
            opt.save_folder, 'last.pth')
        save_model(model, optimizer, opt, opt.epochs, save_file)

        result_record[trial_id, 0] = best_acc
        result_record[trial_id, 1] = val_acc
        result_record[trial_id, 2] = val_F1score

        print('best accuracy: {:.3f}'.format(best_acc))
        print('last accuracy: {:.3f}'.format(val_acc))
        print('final F1:{:.3f}'.format(val_F1score))
        print("deepsense_result_{:,}:".format(opt.label_rate),confusion)
        
        np.savetxt("deepsense_result_labelrate_{:,}_{}.txt".format(opt.label_rate, trial_id), confusion)
        np.savetxt("deepsense_record_acc_labelrate_{:,}_{}.txt".format(opt.label_rate, trial_id), record_acc)

    print("mean accuracy:", np.mean(result_record[:, 0]))
    print("std accuracy:", np.std(result_record[:, 0]))
    print("mean accuracy:", np.mean(result_record[:, 1]))
    print("std accuracy:", np.std(result_record[:, 1]))
    print("mean accuracy:", np.mean(result_record[:, 2]))
    print("std accuracy:", np.std(result_record[:, 2]))
    

if __name__ == '__main__':
    main()
