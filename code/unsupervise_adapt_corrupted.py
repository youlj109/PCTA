# coding=utf-8
import argparse
import os
import sys
import numpy as np
import math
import torchvision.models as models
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from sklearn.metrics import confusion_matrix
from alg.opt import *
from alg import alg
from utils.util import (set_random_seed, save_checkpoint, print_args,
                        train_valid_target_eval_names,alg_loss_dict,                        
                        Tee, img_param_init, print_environ, load_ckpt)
from datautil.getdataloader import get_img_dataloader
from adapt_algorithm import collect_params,configure_model
from adapt_algorithm import PseudoLabel,T3A,BN,ERM,Tent,TSD,Energy,SAR,SAM,EATA,TIPI,PCTA
from datautil.getdataloader import CustomCIFAR100Dataset


def get_args():
    parser = argparse.ArgumentParser(description='Test time adaptation')   
    parser.add_argument('--alpha', type=float,
                        default=1, help='DANN dis alpha')
    parser.add_argument('--anneal_iters', type=int,
                        default=500, help='Penalty anneal iters used in VREx')    
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam hyper-param')
    parser.add_argument('--checkpoint_freq', type=int,
                        default=3, help='Checkpoint every N epoch')
    parser.add_argument('--classifier', type=str,
                        default="linear", choices=["linear", "wn"])
    parser.add_argument('--data_file', type=str, default='',
                        help='root_dir')   
    parser.add_argument('--dis_hidden', type=int,
                        default=256, help='dis hidden dimension')
    parser.add_argument('--gpu_id', type=str, nargs='?',
                        default='0', help="device id to run")
    parser.add_argument('--groupdro_eta', type=float,
                        default=1, help="groupdro eta")
    parser.add_argument('--inner_lr', type=float,
                        default=1e-2, help="learning rate used in MLDG")
    parser.add_argument('--lam', type=float,
                        default=1, help="tradeoff hyperparameter used in VREx")
    parser.add_argument('--lr_decay', type=float, default=0.75, help='for sgd')
    parser.add_argument('--lr_decay1', type=float,
                        default=1.0, help='for pretrained featurizer')
    parser.add_argument('--lr_decay2', type=float, default=1.0,
                        help='inital learning rate decay of network')
    parser.add_argument('--lr_gamma', type=float,
                        default=0.0003, help='for optimizer')
    parser.add_argument('--max_epoch', type=int,
                        default=120, help="max epoch")
    parser.add_argument('--mixupalpha', type=float,
                        default=0.2, help='mixup hyper-param')
    parser.add_argument('--mldg_beta', type=float,
                        default=1, help="mldg hyper-param")
    parser.add_argument('--mmd_gamma', type=float,
                        default=1, help='MMD, CORAL hyper-param')
    parser.add_argument('--momentum', type=float,
                        default=0.9, help='for optimizer')    
    parser.add_argument('--N_WORKERS', type=int, default=4)
    parser.add_argument('--rsc_f_drop_factor', type=float,
                        default=1/3, help='rsc hyper-param')
    parser.add_argument('--rsc_b_drop_factor', type=float,
                        default=1/3, help='rsc hyper-param')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--schuse', action='store_true')
    parser.add_argument('--schusech', type=str, default='cos')
    parser.add_argument('--split_style', type=str, default='strat',
                        help="the style to split the train and eval datasets")
    parser.add_argument('--task', type=str, default="img_dg",
                        choices=["img_dg"], help='now only support image tasks')
    parser.add_argument('--tau', type=float, default=1, help="andmask tau")    
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--batch_size', type=int,
                        default=128, help='batch_size of **test** time')
    parser.add_argument('--dataset', type=str, default='PACS',help='office-home,PACS,VLCS,DomainNet,CIFAR-10,CIFAR-100')
    parser.add_argument('--data_dir', type=str, default='./dataset/PACS', help='data dir')
    parser.add_argument('--lr', type=float, default=1e-5, 
                         help="learning rate of **test** time adaptation,important")
    parser.add_argument('--net', type=str, default='resnet50',
                        help="featurizer: vgg16, resnet18,resnet50, resnet101,DTNBase,ViT-B16,resnext50")
    parser.add_argument('--test_envs', type=int, nargs='+',default=[0], help='target domains')
    parser.add_argument('--output', type=str,default="./tta_output", help='result output path')
    parser.add_argument('--adapt_alg',type=str,default='PCTA',help='[Tent,PL,PLC,T3A,BN,ETA,LAME,ERM,Energy,TSD,SAR,SAM,EATA,TIPI,PCTA]')
    parser.add_argument('--beta',type=float,default=0.9,help='threshold for pseudo label(PL)')
    parser.add_argument('--episodic',action='store_true',help='is episodic or not,default:False')
    parser.add_argument('--steps', type=int, default=1,help='steps of test time, default:1')
    parser.add_argument('--filter_K',type=int,default=100,help='M in T3A/TSD, \in [1,5,20,50,100,-1],-1 denotes no selectiion')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--source_seed',type=int,default=0,help='source model seed')
    parser.add_argument('--update_param',type=str,default='all',help='all / affine / body / head')
    parser.add_argument('--ENERGY_cond',type=str,default='uncond',help='ENERGY_cond Parameter')
    #hpyer-parameters for EATA (ICML22)
    parser.add_argument('--e_margin', type=float, default=math.log(7)*0.40, help='entropy margin E_0 in Eqn. (3) for filtering reliable samples')
    parser.add_argument('--d_margin', type=float, default=0.05, help='\epsilon in Eqn. (5) for filtering redundant samples')
    #hpyer-parameters for PCTA
    parser.add_argument('--PCTA_lam', type=float, default=1)
    parser.add_argument('--PCTA_k', type=float, default=0.9)
    parser.add_argument('--PCTA_tao', type=float, default=1)
    parser.add_argument('--PCTA_k_EM', type=int, default=1)
    parser.add_argument('--PCTA_use_true_class_p', type=bool, default=False)

    parser.add_argument('--pretrain_dir',type=str,default='./model.pkl',help='pre-train model path')      
    args = parser.parse_args()
    args.steps_per_epoch = 100
    args.data_dir = args.data_file+args.data_dir
    
    os.environ['CUDA_VISIBLE_DEVICS'] = args.gpu_id
    os.makedirs(args.output, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.output, 'out.txt'))
    sys.stderr = Tee(os.path.join(args.output, 'err.txt'))
    args = img_param_init(args)
    print_environ()
    return args


if __name__ == '__main__':    
    args = get_args()
    pretrain_model_path = args.pretrain_dir
    set_random_seed(args.seed)
    
    if args.dataset in ['CIFAR-10', 'CIFAR-100', 'ImageNet']:
        class Divided_module(nn.Module):
            def __init__(self, args):
                super(Divided_module, self).__init__()
                self.args = args
                self.algorithm_res = self._load_algorithm()
                num_ftrs = self.algorithm_res.fc.in_features
                self.algorithm_res.fc = nn.Linear(num_ftrs, args.num_classes)
                self.featurizer = nn.Sequential(*list(self.algorithm_res.children())[:-1],nn.Flatten())
                self.classifier = nn.Linear(self.algorithm_res.fc.in_features, self.args.num_classes)
                self.network = nn.Sequential(self.featurizer, self.classifier)
            
            def _load_algorithm(self):
                if self.args.net == 'resnet50':
                    return models.resnet50()
                elif self.args.net == 'resnet18':
                    return models.resnet18()
                else:
                    print('Net selected wrong!')
                    return None

            def forward(self, x):
                return self.network(x)
            
            def predict(self, x):
                return self.network(x)
        
        algorithm = Divided_module(args)
    else:
        algorithm_class = alg.get_algorithm_class(args.algorithm)
        algorithm = algorithm_class(args)
    algorithm.train()
    algorithm = load_ckpt(algorithm,pretrain_model_path)

    #set adapt model and optimizer  
    if args.adapt_alg=='Tent':
        algorithm = configure_model(algorithm)
        params,_ = collect_params(algorithm)
        optimizer = torch.optim.Adam(params,lr=args.lr)        
        adapt_model = Tent(algorithm,optimizer,steps=args.steps, episodic=args.episodic)
    elif args.adapt_alg=='ERM':
        adapt_model = ERM(algorithm)
    elif args.adapt_alg=='PL':
        optimizer = torch.optim.Adam(algorithm.parameters(),lr=args.lr)
        adapt_model = PseudoLabel(algorithm,optimizer,args.beta,steps=args.steps, episodic=args.episodic)
    elif args.adapt_alg=='PLC':
        optimizer = torch.optim.Adam(algorithm.classifier.parameters(),lr=args.lr)
        adapt_model = PseudoLabel(algorithm,optimizer,args.beta,steps=args.steps, episodic=args.episodic)
    elif args.adapt_alg=='T3A':
        adapt_model = T3A(algorithm,filter_K=args.filter_K,steps=args.steps, episodic=args.episodic)
    elif args.adapt_alg=='BN':
        adapt_model = BN(algorithm)  
    elif args.adapt_alg=='PCTA':
        optimizer = torch.optim.Adam(algorithm.parameters(),lr=args.lr)
        sum_params = sum([p.nelement() for p in algorithm.parameters()])
        adapt_model = PCTA(algorithm,optimizer,args.PCTA_lam, args.PCTA_k, args.PCTA_tao, args.PCTA_k_EM, args.PCTA_use_true_class_p)
    elif args.adapt_alg=='ENERGY':
        optimizer = torch.optim.Adam(algorithm.parameters(),lr=args.lr)
        adapt_model = Energy(algorithm,optimizer,steps=args.steps,episodic=args.episodic,
                            im_sz=224,n_ch=3,buffer_size=args.batch_size,n_classes=args.num_classes,
                            sgld_steps=20, sgld_lr=1, sgld_std=0.01, reinit_freq=0.05,
                            if_cond='uncond') #if_cond= 'cond' or 'uncond'
    elif args.adapt_alg=='SAR':
        optimizer = SAM(algorithm.parameters(),torch.optim.SGD,lr=args.lr, momentum=0.9)
        adapt_model = SAR(algorithm, optimizer, steps=args.steps, episodic=args.episodic)
    elif args.adapt_alg=='EATA':
        optimizer = torch.optim.Adam(algorithm.parameters(),lr=args.lr)
        adapt_model = EATA(algorithm, optimizer, steps=args.steps, episodic=args.episodic,fishers=None)
    elif args.adapt_alg=='TIPI':
        adapt_model = TIPI(algorithm,lr_per_sample=args.lr/args.batch_size, optim='Adam', epsilon=2/255,
                           random_init_adv=False,tent_coeff=4.0, use_test_bn_with_large_batches=True)
    elif args.adapt_alg=='TSD':
        if args.update_param=='all':
            optimizer = torch.optim.Adam(algorithm.parameters(),lr=args.lr)
            sum_params = sum([p.nelement() for p in algorithm.parameters()])
        elif args.update_param=='affine':
            algorithm.train()
            algorithm.requires_grad_(False)
            params,_ = collect_params(algorithm)
            optimizer = torch.optim.Adam(params,lr=args.lr)
            for m in algorithm.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
            sum_params = sum([p.nelement() for p in params])
        elif args.update_param=='body':
            #only update encoder
            optimizer = torch.optim.Adam(algorithm.featurizer.parameters(),lr=args.lr)
            print("Update encoder")
        elif args.update_param=='head':
            #only update classifier
            optimizer = torch.optim.Adam(algorithm.classifier.parameters(),lr=args.lr)
            print("Update classifier")
        else:
            raise Exception("Do not support update with %s manner." % args.update_param)
        adapt_model = TSD(algorithm,optimizer,filter_K=args.filter_K,steps=args.steps, episodic=args.episodic)
    
    
    adapt_model.cuda() 
    corruptions = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",  "glass_blur", "motion_blur", "zoom_blur",
                     "snow", "frost", "fog", "brightness",  "contrast", "elastic_transform", "pixelate", "jpeg_compression",  ]
    
    accuracies = []
    accuracy_dict = {}

    for corruption in corruptions:
        data_root=os.path.join(args.data_dir,corruption)
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize])
        if args.dataset in ['CIFAR-100']:
            testset = CustomCIFAR100Dataset(root_dir=data_root,transform=test_transform)
        else:
            testset = ImageFolder(root=data_root,transform=test_transform)
        dataloader = DataLoader(testset,batch_size=args.batch_size,shuffle=True,num_workers=args.N_WORKERS,pin_memory=True)

        total,correct = 0,0
        acc_arr = []
        outputs_arr,labels_arr = [],[]

        for idx,sample in enumerate(dataloader):
            image,label = sample
            image = image.cuda()
            logits = adapt_model(image)
            outputs_arr.append(logits.detach().cpu())
            labels_arr.append(label)
        
        outputs_arr = torch.cat(outputs_arr,0).numpy()
        labels_arr = torch.cat(labels_arr).numpy()
        outputs_arr = outputs_arr.argmax(1)
        matrix = confusion_matrix(labels_arr, outputs_arr)
        acc_per_class = (matrix.diagonal() / matrix.sum(axis=1) * 100.0).round(2)
        avg_acc = 100.0*np.sum(matrix.diagonal()) / matrix.sum()

        print(f"Accuracy for {corruption}: %f"% float(avg_acc))
        accuracy_dict[f'{corruption}_accuracy'] = avg_acc
        accuracies.append(avg_acc)

    mean_accuracy = np.mean(accuracies)
    print(f"Mean accuracy across all corruptions: {mean_accuracy:.4f}%")

    print('\t Hyper-parameter')
    print(f'\t Dataset: {args.dataset}')
    print(f'\t Net: {args.net}')
    print(f'\t Algorithm: {args.adapt_alg}')
    print(f'\t Accuracy: {mean_accuracy:.4f}%')
    print(f'\t lr: {args.lr}')
    print(f'\t PCTA_lam: {args.PCTA_lam}')
    print(f'\t PCTA_k: {args.PCTA_k}')
    print(f'\t PCTA_tao: {args.PCTA_tao}')
    print(f'\t PCTA_k_EM: {args.PCTA_k_EM}')
