import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision as tv

from time import time
from model import Model
from attack import FastGradientSignUntargeted
from utils import makedirs, create_logger, tensor2cuda, numpy2cuda, evaluate, save_model

from argument import parser, print_args

class Trainer():
    def __init__(self, args, logger, attack):
        self.args = args
        self.logger = logger
        self.attack = attack

    def standard_train(self, model, tr_loader, va_loader=None):
        self.train(model, tr_loader, va_loader, False)

    def adversarial_train(self, model, tr_loader, va_loader=None):
        self.train(model, tr_loader, va_loader, True)

    def train(self, model, tr_loader, va_loader=None, adv_train=False):
        args = self.args
        logger = self.logger

        opt = torch.optim.Adam(model.parameters(), args.learning_rate)

        _iter = 0

        begin_time = time()

        for epoch in range(1, args.max_epoch+1):
            for data, label in tr_loader:
                data, label = tensor2cuda(data), tensor2cuda(label)

                if adv_train:
                    # When training, the adversarial example is created from a random 
                    # close point to the original data point. If in evaluation mode, 
                    # just start from the original data point.
                    adv_data = self.attack.perturb(data, label, 'mean', True)
                    output = model(adv_data, _eval=False)
                else:
                    output = model(data, _eval=False)

                loss = F.cross_entropy(output, label)

                opt.zero_grad()
                loss.backward()
                opt.step()

                if _iter % args.n_eval_step == 0:

                    if adv_train:
                        with torch.no_grad():
                            stand_output = model(data, _eval=True)
                        pred = torch.max(stand_output, dim=1)[1]

                        # print(pred)
                        std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                        pred = torch.max(output, dim=1)[1]
                        # print(pred)
                        adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                    else:
                        adv_data = self.attack.perturb(data, label, 'mean', False)

                        with torch.no_grad():
                            adv_output = model(adv_data, _eval=True)
                        pred = torch.max(adv_output, dim=1)[1]
                        # print(label)
                        # print(pred)
                        adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                        pred = torch.max(output, dim=1)[1]
                        # print(pred)
                        std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                    # only calculating the training time
                    logger.info('epoch: %d, iter: %d, spent %.2f s, tr_loss: %.3f' % (
                        epoch, _iter, time() - begin_time, loss.item()))

                    logger.info('standard acc: %.3f %%, robustness acc: %.3f %%' % (
                        std_acc, adv_acc))

                    if va_loader is not None:
                        va_acc, va_adv_acc = self.test(model, va_loader, True)
                        va_acc, va_adv_acc = va_acc * 100.0, va_adv_acc * 100.0

                        logger.info('\n' + '='*30 + ' evaluation ' + '='*30)
                        logger.info('test acc: %.3f %%, test adv acc: %.3f %%' % (
                            va_acc, va_adv_acc))
                        logger.info('='*28 + ' end of evaluation ' + '='*28 + '\n')

                    begin_time = time()

                if _iter % args.n_store_image_step == 0:
                    tv.utils.save_image(torch.cat([data.cpu(), adv_data.cpu()], dim=0), 
                                        os.path.join(args.log_folder, 'images_%d.jpg' % _iter), 
                                        nrow=16)
                    

                if _iter % args.n_checkpoint_step == 0:
                    file_name = os.path.join(args.model_folder, 'checkpoint_%d.pth' % _iter)
                    save_model(model, file_name)

                _iter += 1

    def test(self, model, loader, adv_test=False):
        # adv_test is False, return adv_acc as -1 

        total_acc = 0.0
        num = 0
        total_adv_acc = 0.0

        with torch.no_grad():
            for data, label in loader:
                data, label = tensor2cuda(data), tensor2cuda(label)

                output = model(data, _eval=True)

                pred = torch.max(output, dim=1)[1]
                te_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                
                total_acc += te_acc
                num += output.shape[0]

                if adv_test:
                    # use predicted label as target label
                    # with torch.enable_grad():
                    adv_data = self.attack.perturb(data, pred, 'mean', False)

                    adv_output = model(adv_data, _eval=True)

                    adv_pred = torch.max(adv_output, dim=1)[1]
                    adv_acc = evaluate(adv_pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                    total_adv_acc += adv_acc
                else:
                    total_adv_acc = -num

        return total_acc / num , total_adv_acc / num

def main(args):

    save_folder = '%s_%s' % (args.dataset, args.affix)

    log_folder = os.path.join(args.log_root, save_folder)
    model_folder = os.path.join(args.model_root, save_folder)

    makedirs(log_folder)
    makedirs(model_folder)

    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', model_folder)

    logger = create_logger(log_folder, args.todo, 'info')

    print_args(args, logger)

    model = Model(i_c=1, n_c=10)

    attack = FastGradientSignUntargeted(model, 
                                        args.epsilon, 
                                        args.alpha, 
                                        min_val=0, 
                                        max_val=1, 
                                        max_iters=args.k, 
                                        _type=args.perturbation_type)

    if torch.cuda.is_available():
        model.cuda()

    trainer = Trainer(args, logger, attack)

    if args.todo == 'train':
        tr_dataset = tv.datasets.MNIST(args.data_root, 
                                       train=True, 
                                       transform=tv.transforms.ToTensor(), 
                                       download=True)

        tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        # evaluation during training
        te_dataset = tv.datasets.MNIST(args.data_root, 
                                       train=False, 
                                       transform=tv.transforms.ToTensor(), 
                                       download=True)

        te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        trainer.train(model, tr_loader, te_loader, args.adv_train)
    elif args.todo == 'test':
        pass
    else:
        raise NotImplementedError
    



if __name__ == '__main__':
    args = parser()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(args)