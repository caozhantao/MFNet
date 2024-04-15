import argparse
import os
import numpy as np
from skimage import morphology, measure
from tqdm import tqdm

from dataloaders.datasets.medical import MedicalSegmentDataset, MedicalSegmentDataset1
from modeling.unet import UNet
from modeling.unet2d import Unet2D
from mypath import Path
from medpy import metric
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.dice_score import dice_loss, multiclass_dice_coeff
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator


def repeat_dataloader(iterable):
    """ repeat dataloader """
    while True:
        for x in iterable:
            yield x
class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Define Dataloader
        # kwargs = {'num_workers': args.workers, 'pin_memory': True}
        # self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
        train_dataset = MedicalSegmentDataset(args.data_path,
                                              train=True,
                                              )

        train_dataset1 = MedicalSegmentDataset1(args.data_path,
                                              train=True,
                                              )

        val_dataset = MedicalSegmentDataset(args.data_path,
                                            train=False,
                                            )

        num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   num_workers=num_workers,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   )

        train_loader1 = torch.utils.data.DataLoader(train_dataset1,
                                                   batch_size=args.batch_size,
                                                   num_workers=num_workers,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   )

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.batch_size,
                                                 num_workers=num_workers,
                                                 pin_memory=True,
                                                 )
        self.train_loader = train_loader
        self.train_loader1 = train_loader1
        self.val_loader = val_loader
        self.nclass = 2
        # Define network
        # model = DeepLab(num_classes=self.nclass,
        #                 backbone=args.backbone,
        #                 output_stride=args.out_stride,
        #                 sync_bn=args.sync_bn,
        #                 freeze_bn=args.freeze_bn)
        model = UNet(in_channels=3, n_classes=self.nclass, base_c=64)
        # model = Unet2D(n_classes=self.nclass)
        # train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
        #                 {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        train_params = [p for p in model.parameters() if p.requires_grad]
        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum, lr=args.lr,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset + '_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

        dataloader_train = []
        dataloader_train.append(train_loader)
        dataloader_train.append(train_loader1)
        self.dataloader_train = dataloader_train


    def training1(self,epoch):
        data_iter = [repeat_dataloader(self.dataloader_train[i]) for i in range(2)]
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.dataloader_train[0])
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            # sample = [sample]
            sample1 = [next(data_iter[i]) for i in range(1, 2)]
            image, target = sample['image'], sample['label']
            image1, target1 = sample1[0]['image'], sample1[0]['label']
            image = image.to(dtype=torch.float32)
            image1 = image1.to(dtype=torch.float32)
            target = target.to(dtype=torch.long)
            target1 = target1.to(dtype=torch.long)
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
                image1, target1 = image1.cuda(), target1.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            # inner loop
            output = self.model(image)
            output1 = self.model(image1)
            loss1 = self.criterion(output, target).cuda() \
                   + dice_loss(F.softmax(output, dim=1).float(),
                               F.one_hot(target, self.model.module.n_classes).permute(0, 3, 1, 2).float(),
                               multiclass=False)
            loss2 = self.criterion(output1, target1).cuda() \
                   + dice_loss(F.softmax(output1, dim=1).float(),
                               F.one_hot(target1, self.model.module.n_classes).permute(0, 3, 1, 2).float(),
                               multiclass=False)
            loss = loss1 + loss2
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # # Show 10 * 3 inference results each epoch
            # if i % (num_img_tr // 10) == 0:
            #     global_step = i + num_img_tr * epoch
            #     self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            image = image.to(dtype=torch.float32)
            target = target.to(dtype=torch.long)
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            # loss = self.criterion(output, target)
            loss = self.criterion(output, target).cuda() \
                   + dice_loss(F.softmax(output, dim=1).float(),
                               F.one_hot(target, self.model.module.n_classes).permute(0, 3, 1, 2).float(),
                               multiclass=False)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # # Show 10 * 3 inference results each epoch
            # if i % (num_img_tr // 10) == 0:
            #     global_step = i + num_img_tr * epoch
            #     self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        dice_score = 0
        test_total = 0.0
        asd = 0
        count = 0
        dsc = 0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            target = target.to(dtype=torch.long)
            mask_true = F.one_hot(target, self.model.module.n_classes).permute(0, 3, 1, 2).float()
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
                mask_true = mask_true.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target).cuda()
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))

            test_total += target.size(0)

            mask_pred = F.one_hot(output.argmax(dim=1), self.model.module.n_classes).permute(0, 3, 1, 2).float()
            dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

            np.set_printoptions(threshold=np.inf)
            pred = output.cpu().numpy()
            target = target.cpu().numpy()

            # asd = metric.binary.asd(pred, target)
            pred = np.argmax(pred, axis=1)
            if pred.sum() == 0:
                asd += 100
            else:
                asd += metric.binary.asd(pred, target)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)
            dsc += metric.binary.dc(pred, target)
            tbar.set_description('Dice: {},ASD: {}'.format(
                round(100 * dsc / (i + 1), 2),
                round(asd / (i + 1), 2)
            ))
            count = count + 1
        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        asd = asd / count
        dsc = 100 * dsc / count
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        # self.writer.add_scalar('val/dice', 100 * dsc / (i + 1), epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        self.writer.add_scalar('val/ASD', asd, epoch)
        self.writer.add_scalar('val/dsc', dsc, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{},fwIoU: {}, asd: {}, dsc: {}".format(Acc, Acc_class, mIoU,
                                                                                           FWIoU, asd, dsc))
        print('Loss: %.3f' % test_loss)

        new_pred = dsc
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    # parser.add_argument("--data-path", default="../dataset/ct-ich", help="DRIVE root")
    parser.add_argument("--data-path", default="/home/temp58/dataset/seg-CQ500/", help="DRIVE root")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='medical',
                        choices=['pascal', 'coco', 'cityscapes', 'medical'],
                        help='dataset name (default: medical)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=6,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=4,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
    False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1337, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
            'medical': 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
            'medical': 0.01,
        }
        args.lr = lrs[args.dataset.lower()] * 1.5

    if args.checkname is None:
        args.checkname = 'unet-' + str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training1(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()


if __name__ == "__main__":
    main()
