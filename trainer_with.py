import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import models

import time
import os
import math
import pdb
from utils import logger_setting, Timer
from tensorboard_logger import log_value
import torchvision.models 
from torchvision.utils import save_image


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * 0.5

def grad_reverse(x):
    return GradReverse.apply(x)



class Trainer(object):
    def __init__(self, option):
        self.option = option

        self._build_model()
        self._set_optimizer()
        self.logger = logger_setting(option.exp_name, option.save_dir, option.debug)

    def _build_model(self):
        self.n_color_cls = 1
        # self.n_color_cls = 2

        self.net = models.convnet(num_classes=self.option.n_class)
        # self.net = torchvision.models.resnet18(pretrained=True, num_classes=self.option.n_class)

        self.pred_net_r = models.Predictor(input_ch=32, num_classes=self.n_color_cls)
        # self.pred_net_g = models.Predictor(input_ch=32, num_classes=self.n_color_cls)
        # self.pred_net_b = models.Predictor(input_ch=32, num_classes=self.n_color_cls)

        # self.loss = nn.CrossEntropyLoss(ignore_index=255)
        # self.loss = nn.BCEWithLogitsLoss()
        self.loss = nn.BCELoss()
        # self.color_loss = nn.BCEWithLogitsLoss()
        self.color_loss = nn.BCELoss()


        if self.option.cuda:
            self.net.cuda()
            self.pred_net_r.cuda()
            # self.pred_net_g.cuda()
            # self.pred_net_b.cuda()
            self.loss.cuda()
            # self.color_loss.cuda()

    def _set_optimizer(self):
        self.optim = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.option.lr, momentum=self.option.momentum, weight_decay=self.option.weight_decay)
        self.optim_r = optim.SGD(self.pred_net_r.parameters(), lr=self.option.lr, momentum=self.option.momentum, weight_decay=self.option.weight_decay)
        # self.optim_g = optim.SGD(self.pred_net_g.parameters(), lr=self.option.lr, momentum=self.option.momentum, weight_decay=self.option.weight_decay)
        # self.optim_b = optim.SGD(self.pred_net_b.parameters(), lr=self.option.lr, momentum=self.option.momentum, weight_decay=self.option.weight_decay)

        #TODO: last_epoch should be the last step of loaded model
        lr_lambda = lambda step: self.option.lr_decay_rate ** (step // self.option.lr_decay_period)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lr_lambda, last_epoch=-1)
        self.scheduler_r = optim.lr_scheduler.LambdaLR(self.optim_r, lr_lambda=lr_lambda, last_epoch=-1)
        # self.scheduler_g = optim.lr_scheduler.LambdaLR(self.optim_g, lr_lambda=lr_lambda, last_epoch=-1)
        # self.scheduler_b = optim.lr_scheduler.LambdaLR(self.optim_b, lr_lambda=lr_lambda, last_epoch=-1)

    @staticmethod
    def _weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def _initialization(self):
        self.net.apply(self._weights_init)


        if self.option.is_train and self.option.use_pretrain:
            if self.option.checkpoint is not None:
                self._load_model()
            else:
                print("Pre-trained model not provided")



    def _mode_setting(self, is_train=True):
        if is_train:
            self.net.train()
            self.pred_net_r.train()
            # self.pred_net_g.train()
            # self.pred_net_b.train()
        else:
            self.net.eval()
            self.pred_net_r.eval()
            # self.pred_net_g.eval()
            # self.pred_net_b.eval()



    def _train_step(self, data_loader, step):
        _lambda = 0.01

        for i, (images,color_labels,labels) in enumerate(data_loader):
            # pdb.set_trace()
            images = self._get_variable(images)
            color_labels = self._get_variable(color_labels)
            labels = self._get_variable(labels)

            self.optim.zero_grad()
            self.optim_r.zero_grad()
            # self.optim_g.zero_grad()
            # self.optim_b.zero_grad()
            # pdb.set_trace()
            
            feat_label, pred_label = self.net(images)
            # sigmoid    = nn.Sigmoid()
            # pred_label = sigmoid(pred_label)
            
            
            # predict colors from feat_label. Their prediction should be uniform.
            _,pseudo_pred_r = self.pred_net_r(feat_label)
            # _,pseudo_pred_g = self.pred_net_g(feat_label)
            # _,pseudo_pred_b = self.pred_net_b(feat_label)


            # loss for self.net
            labels = torch.reshape(labels, (-1,1))
            # import pdb; pdb.set_trace()
            # loss for self.net
            labels= labels.float()
            # labels= labels.long()
            # pdb.set_trace()
            
            loss_pred = self.loss(pred_label, labels)

            loss_pseudo_pred_r = torch.mean(torch.sum(pseudo_pred_r*torch.log(pseudo_pred_r),1))
            # loss_pseudo_pred_g = torch.mean(torch.sum(pseudo_pred_g*torch.log(pseudo_pred_g),1))
            # loss_pseudo_pred_b = torch.mean(torch.sum(pseudo_pred_b*torch.log(pseudo_pred_b),1))

            loss_pred_ps_color = loss_pseudo_pred_r

            loss = loss_pred + loss_pred_ps_color*_lambda

            loss.backward()
            self.optim.step()
            

            ## # # # # # # # # #  color loss 
            self.optim.zero_grad()
            self.optim_r.zero_grad()
            # self.optim_g.zero_grad()
            # self.optim_b.zero_grad()
            # pdb.set_trace()
            feat_label, pred_label = self.net(images)
            # sigmoid    = nn.Sigmoid()
            # pred_label = sigmoid(pred_label)

            feat_color = grad_reverse(feat_label)
            _, pred_r = self.pred_net_r(feat_color)
            # pred_g,_ = self.pred_net_g(feat_color)
            # pred_b,_ = self.pred_net_b(feat_color)

            # loss for rgb predictors
            
            color_labels = color_labels.float()
            color_labels = color_labels.view(color_labels.shape[0], 1, 1, 1)
            color_labels = color_labels.repeat(1,1, pred_r.shape[2], pred_r.shape[3])
            loss_pred_r = self.color_loss(pred_r, color_labels)


            loss_pred_color = loss_pred_r 

            loss_pred_color.backward()
            self.optim.step()
            self.optim_r.step()
            # self.optim_g.step()
            # self.optim_b.step()
            log_value('Class loss', loss_pred, step*data_loader.__len__()+i)
            log_value('RGB loss', loss_pred_r, step*data_loader.__len__()+i)
            log_value('MI loss', loss_pred_ps_color, step*data_loader.__len__()+i)
            log_value('total loss', loss_pred+loss_pred_r+loss_pred_ps_color, step*data_loader.__len__()+i)

            if i % self.option.log_step == 0:
                msg = "[TRAIN] cls loss : %.6f, rgb : %.6f, MI : %.6f  (epoch %d.%02d)" \
                       % (loss_pred,loss_pred_color,loss_pred_ps_color,step,int(100*i/data_loader.__len__()))
                self.logger.info(msg)

    def get_cam(self, img):
        # get fc weights
        fc_weights = self.net.fc.weight 
        feat = self.net.get_feature(img)
        
        # multiply fc_weights and feature
        fc_weights = fc_weights.view(fc_weights.size(0), fc_weights.size(1), 1, 1)
        
        fc_weights = fc_weights.repeat(feat.size(0), 1, feat.size(2), feat.size(3))
        out = feat*fc_weights
        # average pooling
        out_mean = out.mean(dim=1).unsqueeze(1)
        # import pdb; pdb.set_trace()
        # min max normalization
        # import pdb; pdb.set_trace()
        out_mean_shape = out_mean.shape
        out_mean = out_mean.view(out_mean_shape[0], 1, -1)
        out_mean_min = out_mean.min(dim=2)[0]
        out_mean_min = out_mean_min.unsqueeze(2).expand_as(out_mean)
        out_mean_max = out_mean.max(dim=2)[0]
        out_mean_max = out_mean_max.unsqueeze(2).expand_as(out_mean)
        out_norm = (out_mean - out_mean_min) / (out_mean_max - out_mean_min)
        out_norm[torch.isnan(out_norm)] = 0

        # interpolation
        cam = torch.nn.functional.interpolate(out_norm.view(out_mean_shape), size=img.size(2))
        

        for i in range(img.size(0)):
            # if self.option.save_dir == './Ratio_Reverse':
            save_image(torch.cat((img[i],cam[i].expand_as(img[i]),cam[i].expand_as(img[i])*img[i]),dim=2), './cam_oneone_wo_GRL/img_cam_{i}.jpg'.format(i=i))
            # elif self.option.save_dir == './Ratio_oneone':
            #     save_image(torch.cat((img[i],cam[i].expand_as(img[i]),cam[i].expand_as(img[i])*img[i]),dim=2), './cam_test_oneone/img_cam_{i}.jpg'.format(i=i))
            # elif self.option.save_dir == './Ratio_same':
            #     save_image(torch.cat((img[i],cam[i].expand_as(img[i]),cam[i].expand_as(img[i])*img[i]),dim=2), './cam_test_same/img_cam_{i}.jpg'.format(i=i))
            
        
 
    def _validate(self, step, data_loader):
        self._mode_setting(is_train=False)
        self._initialization()
        if self.option.checkpoint is None:
            self._load_model()
        # else:
        #     print("No trained model for evaluation provided")
        #     import sys
        #     sys.exit()

        num_test = 10000

        total_num_correct = 0.
        total_num_test = 0.
        total_loss = 0.
        for i, (images,color_labels,labels) in enumerate(data_loader):
            
            start_time = time.time()
            images = self._get_variable(images)
            color_labels = self._get_variable(color_labels)
            labels = self._get_variable(labels)

            self.optim.zero_grad()
            _, pred_label = self.net(images)
            
            labels = torch.reshape(labels, (-1,1))
            # import pdb; pdb.set_trace()
            labels= labels.float()
            # labels= labels.long()
            loss = self.loss(pred_label, labels)
            
            # pdb.set_trace()
            batch_size = images.shape[0]
            total_num_correct += self._num_correct(pred_label,labels)
            total_loss += loss.data*batch_size
            total_num_test += batch_size
        
        self.get_cam(images)
               
        avg_loss = total_loss/total_num_test
        avg_acc = float(total_num_correct)/float(total_num_test)

        # log_value('Accuracy', avg_acc, step*data_loader.__len__())

        msg = "EVALUATION LOSS  %.4f, ACCURACY : %.4f (%d/%d)" % \
                        (avg_loss,avg_acc,int(total_num_correct),total_num_test)
        self.logger.info(msg)



    def _num_correct(self,outputs,labels):
        # pdb.set_trace()
        k1 = torch.zeros_like(outputs)
        k1[outputs>0.5] = 1
        k1 = k1.squeeze()
        k2 = labels.squeeze()
        correct = k1.eq(k2.expand_as(k1))
        correct = correct.view(-1).sum()
        return correct
        
    


    # def _accuracy(self, outputs, labels):
    #     batch_size = labels.size(0)
    #     _, preds = outputs.topk(k=1, dim=1)
    #     preds = preds.t()
    #     preds = preds.float()
    #     correct = preds.eq(labels.view(1, -1).expand_as(preds))
    #     correct = correct.view(-1).float().sum(0, keepdim=True)
    #     accuracy = correct.mul_(100.0 / batch_size)
    #     return accuracy

    def _save_model(self, step):
        torch.save({
            'step': step,
            'optim_state_dict': self.optim.state_dict(),
            'net_state_dict': self.net.state_dict()
        }, os.path.join(self.option.save_dir,self.option.exp_name, 'checkpoint_step_%04d.pth' % step))
        print('checkpoint saved. step : %d'%step)

    def _load_model(self):
        
        if self.option.checkpoint is None:
            # import pdb; pdb.set_trace()
            ckpt_dir = os.path.join(self.option.save_dir,self.option.exp_name)
            ckpt_files = os.listdir(ckpt_dir)

            if len(ckpt_files) == 2:
                return 0
            ckpt_files.sort()
            ckpt = torch.load(os.path.join(ckpt_dir, ckpt_files[-3]))
            step = int(ckpt_files[-3][-8:-4])
        else:
            ckpt = torch.load(self.option.checkpoint)
            step = 0


        self.net.load_state_dict(ckpt['net_state_dict'])
        self.optim.load_state_dict(ckpt['optim_state_dict'])
        return step

    def train(self, train_loader, val_loader=None):
        self._initialization()
        # if self.option.checkpoint is not None:
        #     self._load_model()

        load_step = self._load_model()

        self._mode_setting(is_train=True)
        timer = Timer(self.logger, self.option.max_step)
        start_epoch = load_step
        for step in range(start_epoch, self.option.max_step):

            if self.option.train_baseline:
                self._train_step_baseline(train_loader, step)
            else:
                self._train_step(train_loader,step)
            self.scheduler.step()
            self.scheduler_r.step()
            # self.scheduler_g.step()d
            # self.scheduler_b.step()

            if step == 1 or step % self.option.save_step == 0 or step == (self.option.max_step-1):
                self._save_model(step)
                if val_loader is not None:
                    self._validate(step, val_loader)
                # self._save_model(step)


    def _get_variable(self, inputs):
        if self.option.cuda:
            return Variable(inputs.cuda())
        return Variable(inputs)



