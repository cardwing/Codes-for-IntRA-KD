import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import cv2
import utils.transforms as tf
import numpy as np
import models
from models import sync_bn
import dataset as ds
from options.options import parser

best_mIoU = 0


def main():
    global args, best_mIoU
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu) for gpu in args.gpus)
    args.gpus = len(args.gpus)

    if args.no_partialbn:
        sync_bn.Synchronize.init(args.gpus)

    if args.dataset == 'VOCAug' or args.dataset == 'VOC2012' or args.dataset == 'COCO':
        num_class = 21
        ignore_label = 255
        scale_series = [10, 20, 30, 60]
    elif args.dataset == 'Cityscapes':
        num_class = 19
        ignore_label = 255 # 0
        scale_series = [15, 30, 45, 90]
    elif args.dataset == 'ApolloScape':
        num_class = 37 # merge the noise and ignore labels
        ignore_label = 255 # 0
    else:
        raise ValueError('Unknown dataset ' + args.dataset)

    model = models.ERFNet(num_class, partial_bn=not args.no_partialbn)
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    model = torch.nn.DataParallel(model, device_ids=range(args.gpus)).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mIoU = checkpoint['best_mIoU']
            torch.nn.Module.load_state_dict(model, checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))


    cudnn.benchmark = True
    cudnn.fastest = True

    # Data loading code

    test_loader = torch.utils.data.DataLoader(
        getattr(ds, args.dataset.replace("ApolloScape", "VOCAug") + 'DataSet')(data_list=args.val_list, transform=torchvision.transforms.Compose([
            tf.GroupRandomScale(size=(0.5, 0.5), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
            # tf.GroupRandomScaleRatio(size=(3384, 3384, 1010, 1010), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
            tf.GroupNormalize(mean=(input_mean, (0, )), std=(input_std, (1, ))),
        ])), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    # define loss function (criterion) optimizer and evaluator
    weights = [1.0 for _ in range(37)]
    weights[0] = 0.05
    weights[36] = 0.05
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = torch.nn.NLLLoss(ignore_index=ignore_label, weight=class_weights).cuda()
    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
    optimizer = torch.optim.SGD(policies, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    evaluator = EvalSegmentation(num_class, ignore_label)

    ### evaluate ###
    validate(test_loader, model, criterion, 0, evaluator)
    return


def validate(val_loader, model, criterion, iter, evaluator, logger=None):

    batch_time = AverageMeter()
    losses = AverageMeter()
    IoU = AverageMeter()
    mIoU = 0

    # switch to evaluate mode
    model.eval()

    img_w = 1692
    img_h = 505

    end = time.time()
    for i, (input, target, img_name) in enumerate(val_loader):
        # target = target.cuda(async=True)
        with torch.no_grad():
            input_var = input # torch.autograd.Variable(input, volatile=True)        
            input_var_1 = input_var[:, :, :int(args.test_size / 3), :args.test_size]
            input_var_2 = input_var[:, :, :int(args.test_size / 3), (img_w - args.test_size):]
            input_var_3 = input_var[:, :, (img_h - int(args.test_size / 3)):, :args.test_size]
            input_var_4 = input_var[:, :, (img_h - int(args.test_size / 3)):, (img_w - args.test_size):]

            if i == 0:
                freq_mat = np.zeros((img_h, img_w))
                freq_mat[:int(args.test_size / 3), :args.test_size] += np.ones((int(args.test_size / 3), args.test_size))
                freq_mat[:int(args.test_size / 3), (img_w - args.test_size):] += np.ones((int(args.test_size / 3), args.test_size))
                freq_mat[(img_h - int(args.test_size / 3)):, :args.test_size] += np.ones((int(args.test_size / 3), args.test_size))
                freq_mat[(img_h - int(args.test_size / 3)):, (img_w - args.test_size):] += np.ones((int(args.test_size / 3), args.test_size))
        
            # target_var = torch.autograd.Variable(target)

            # compute output
            output_1 = model(input_var_1)
            output_2 = model(input_var_2)
            output_3 = model(input_var_3)
            output_4 = model(input_var_4)

            # measure accuracy and record loss

            pred_1 = output_1.data.cpu().numpy()#.transpose(0, 2, 3, 1)
            pred_2 = output_2.data.cpu().numpy()#.transpose(0, 2, 3, 1)
            pred_3 = output_3.data.cpu().numpy()#.transpose(0, 2, 3, 1)
            pred_4 = output_4.data.cpu().numpy()#.transpose(0, 2, 3, 1)

            pred = np.zeros((args.batch_size, 37, img_h, img_w))
            pred[:, :, :int(args.test_size / 3), :args.test_size] += pred_1
            pred[:, :, :int(args.test_size / 3), (img_w - args.test_size):] += pred_2
            pred[:, :, (img_h - int(args.test_size / 3)):, :args.test_size] += pred_3
            pred[:, :, (img_h - int(args.test_size / 3)):, (img_w - args.test_size):] += pred_4

            pred = pred / freq_mat
            pred = pred.transpose(0, 2, 3, 1)

        pred = np.argmax(pred, axis=3).astype(np.uint8)
        pred = pred + 1
        for cnt in range(len(img_name)):
            np.save('road05_new_erfnet/' + img_name[cnt].split('/')[5].replace('jpg', 'npy'), pred[cnt])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time)))

    print('finished, #test:{}'.format(i) )

    return mIoU


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    filename = os.path.join('outputs', '_'.join((args.snapshot_pref, args.method.lower(), filename)))
    torch.save(state, filename)
    if is_best:
        best_name = os.path.join('outputs', '_'.join((args.snapshot_pref, args.method.lower(), 'model_best.pth.tar')))
        shutil.copyfile(filename, best_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def update(self, val, n=1):
        if self.val is None:
            self.val = val
            self.sum = val * n
            self.count = n
            self.avg = self.sum / self.count
        else:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


class EvalSegmentation(object):
    def __init__(self, num_class, ignore_label=None):
        self.num_class = num_class
        self.ignore_label = ignore_label

    def __call__(self, pred, gt):
        assert (pred.shape == gt.shape)
        gt = gt.flatten().astype(int)
        pred = pred.flatten().astype(int)
        locs = (gt != self.ignore_label)
        sumim = gt + pred * self.num_class
        hs = np.bincount(sumim[locs], minlength=self.num_class**2).reshape(self.num_class, self.num_class)
        return hs


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # decay = 0.1**(sum(epoch >= np.array(lr_steps)))
    decay = ((1 - float(epoch) / args.epochs)**(0.9))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


if __name__ == '__main__':
    main()
