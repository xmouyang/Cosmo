from __future__ import print_function

import sys
import argparse
import time
import math
import numpy as np
import torch.optim as optim

import torch
import torch.backends.cudnn as cudnn

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer
from cmc_model import MyUTDModelFeature, LinearClassifierAttn
import data_pre as data
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
    parser.add_argument('--batch_size', type=int, default=27,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs')
    parser.add_argument('--iterative_epochs', type=int, default=20,
                        help='number of iterative training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='MyUTDmodel')
    parser.add_argument('--dataset', type=str, default='UTD-MHAD',
                        choices=['USC-HAR', 'UTD-MHAD'], help='dataset')
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
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')


    parser.add_argument('--ckpt', type=str, default='./save/CMC/UTD-MHAD_models/CMC_UTD-MHAD_MyUTDmodel_label_',
                        help='path to pre-trained model')
    parser.add_argument('--trial', type=int, default='3',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # set the path according to the environment
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
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

    return opt



def set_loader(opt):

    # load data (already normalized)
    num_of_train = (opt.num_train_basic * opt.label_rate / 5 * np.ones(opt.num_class)).astype(int)
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
    model = MyUTDModelFeature(input_size=1)
    classifier = LinearClassifierAttn(num_classes=opt.num_class)
    criterion = torch.nn.CrossEntropyLoss()

    ## load the pretrained feature encoders
    ckpt_path = opt.ckpt + str(opt.label_rate) + '_lr_0.01_decay_0.9_bsz_27_temp_0.07_trial_0_epoch_300/last.pth'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    model.load_state_dict(state_dict)

    #freeze the MLP in pretrained feature encoders
    for name, param in model.named_parameters():
        if "head" in name:
            param.requires_grad = False
        
    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""

    model.train()
    classifier.train() 

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

        # compute loss
        feature1, feature2 = model.encoder(input_data1, input_data2)
        output, weight1, weight2 = classifier(feature1, feature2)
        loss = criterion(output, labels)


        # update metric
        losses.update(loss.item(), bsz)
        acc, _ = accuracy(output, labels, topk=(1, 5))
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
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

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
                input_data1 = input_data1.cuda()
                input_data2 = input_data2.cuda()
                labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            feature1, feature2 = model.encoder(input_data1, input_data2)
            output, weight1, weight2 = classifier(feature1, feature2)
            loss = criterion(output, labels)

            # calculate and store confusion matrix
            label_list.extend(labels.cpu().numpy())
            pred_list.extend(output.max(1)[1].cpu().numpy())

            rows = labels.cpu().numpy()
            cols = output.max(1)[1].cpu().numpy()

            for label_index in range(labels.shape[0]):
                confusion[rows[label_index], cols[label_index]] += 1

            # update metric
            losses.update(loss.item(), bsz)
            acc, _ = accuracy(output, labels, topk=(1, 5))
            top1.update(acc[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
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
        model, classifier, criterion = set_model(opt)

        # build optimizer for feature extractor and classifier
        optimizer = optim.SGD([ 
                    {'params': model.parameters(), 'lr': 1e-4},   # 0
                    {'params': classifier.parameters(), 'lr': opt.learning_rate}],
                    momentum=opt.momentum,
                    weight_decay=opt.weight_decay)

        record_acc = np.zeros(opt.epochs)

        best_acc = 0

        # training routine
        for epoch in range(1, opt.epochs + 1):
            adjust_learning_rate(opt, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            loss, acc = train(train_loader, model, classifier, criterion,
                              optimizer, epoch, opt)
            time2 = time.time()
            print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
                epoch, time2 - time1, acc))

            # eval for one epoch
            loss, val_acc, confusion, val_F1score = validate(val_loader, model, classifier, criterion, opt)
            if val_acc > best_acc:
                best_acc = val_acc

            record_acc[epoch-1] = val_acc

        result_record[trial_id, 0] = best_acc
        result_record[trial_id, 1] = val_acc
        result_record[trial_id, 2] = val_F1score

        print('best accuracy: {:.2f}'.format(best_acc))
        print('last accuracy: {:.3f}'.format(val_acc))
        print('final F1:{:.3f}'.format(val_F1score))
        print("CMC_result_labelrate_{:,}:".format(opt.label_rate),confusion)

        np.savetxt("CMC_result_labelrate_{:,}_finetune.txt".format(opt.label_rate), confusion)
        np.savetxt("CMC_record_acc_labelrate_{:,}_finetune.txt".format(opt.label_rate), record_acc)

    print("mean accuracy:", np.mean(result_record[:, 0]))
    print("std accuracy:", np.std(result_record[:, 0]))
    print("mean accuracy:", np.mean(result_record[:, 1]))
    print("std accuracy:", np.std(result_record[:, 1]))
    print("mean accuracy:", np.mean(result_record[:, 2]))
    print("std accuracy:", np.std(result_record[:, 2]))


if __name__ == '__main__':
    main()

    
