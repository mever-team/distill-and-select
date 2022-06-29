import os
import copy
import utils
import torch
import argparse
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets.generators import StudentPairGenerator
from model.students import FineGrainedStudent, CoarseGrainedStudent

    
def main(args):
    print('\nInput Arguments')
    print('---------------')
    utils.pprint_args(args)

    print('\n> Create generator of video pairs')
    print('>> loading pairs...')
    dataset = StudentPairGenerator(args)
    
    print('\n> Building network')
    if 'fine' in args.student_type:
        model = FineGrainedStudent(**vars(args))
    else:
        model = CoarseGrainedStudent(**vars(args))
    model = model.to(args.gpu_id)
    model.train()
    print(model)
    
    distil_criterion = nn.L1Loss()
    params = [v for v in filter(lambda p: p.requires_grad, model.parameters())]
    optimizer = torch.optim.Adam(params,
                                 lr=args.learning_rate, 
                                 weight_decay=args.weight_decay)
    global_step = torch.zeros((1,))
    
    if args.load_model:
        print('>> loading network')
        d = torch.load(os.path.join(args.experiment_path, 'model_{}.pth'.format(
            model.get_network_name())), map_location='cpu')
        model.load_state_dict(d['model'])
        if 'optimizer' in d:
            optimizer.load_state_dict(d['optimizer'])
        if 'global_step' in d:
            global_step = d.pop('global_step')

    if args.val_hdf5 is not None:
        from datasets import FIVR
        from evaluation_student import query_vs_target, queries_vs_database
        if 'fine' in args.student_type:
            eval_function = query_vs_target
        else:
            eval_function = queries_vs_database
        val_dataset = FIVR(version='5k')
        val_args = copy.deepcopy(args)
        val_args.dataset_hdf5 = val_args.val_hdf5
        val_max = .0

    print('\n> Start training')
    for epoch in range(args.epochs):
        dataset.next_epoch(np.random.randint(np.iinfo(np.int32).max))
        loader = DataLoader(dataset, num_workers=args.workers, 
                            shuffle=True, batch_size=args.batch_sz//2,
                            collate_fn=utils.collate_student)
        
        tloss, dloss, rloss = [], [], []
        pbar = tqdm(loader, desc='epoch {}'.format(epoch), unit='iter')
        for pairs in pbar:
            optimizer.zero_grad()

            videos = pairs[0].to(args.gpu_id)
            masks = pairs[1].to(args.gpu_id)
            similarities = pairs[2].to(args.gpu_id)
            
            features = model.index_video(videos, masks)
            anchors, positives, negatives = torch.chunk(features, 3, dim=0)
            anchors_mk, positives_mk, negatives_mk = torch.chunk(masks, 3, dim=0)

            pos_pairs, neg_pairs, regularization_loss = model(
                anchors, positives, negatives, anchors_mk, positives_mk, negatives_mk)
            
            teacher_similarities = similarities.view(-1)
            student_similarities = torch.cat([pos_pairs, neg_pairs], 1).view(-1)
            distillation_loss = distil_criterion(student_similarities, teacher_similarities)
            loss = distillation_loss
            if regularization_loss is not None:
                loss += args.r_parameter*regularization_loss

            loss.backward()
            optimizer.step()
            global_step += 1
            
            tloss.append(loss.detach().cpu().numpy())
            dloss.append(distillation_loss.detach().cpu().numpy())
            if regularization_loss is not None:
                rloss.append(regularization_loss.detach().cpu().numpy())

            if global_step % 5 == 0:
                p = {'total_loss': '{:.3f} ({:.3f})'.format(tloss[-1], np.mean(tloss)),
                     'distillation_loss': '{:.3f} ({:.3f})'.format(dloss[-1], np.mean(dloss))}
                if regularization_loss is not None:
                    p['regularization_loss'] ='{:.3f} ({:.3f})'.format(rloss[-1], np.mean(rloss))
                pbar.set_postfix(p)

        if args.val_hdf5 is None:
            utils.save_model(args, model, optimizer, global_step)
        elif (epoch + 1) % args.val_step == 0:
            model.eval()
            res = eval_function(model, val_dataset, val_args)
            model.train()
            if res[args.val_set] > val_max:
                val_max = res[args.val_set]
                utils.save_model(args, model, optimizer, global_step)
        
                
if __name__ == '__main__':
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=80)
    parser = argparse.ArgumentParser(description='This is the training code for knowledge distillation of video similarity', formatter_class=formatter)
    # Experiment options
    parser.add_argument('--experiment_path', type=str, required=True, 
                        help='Path of the experiment where the weights of the trained networks will be stored.')
    parser.add_argument('--trainset_hdf5', type=str, required=True, 
                        help='Path to hdf5 file containing the features of the DnS-100K trainset.')
    parser.add_argument('--trainset_percentage', type=float, default=100, 
                        help='Percentage of videos in the training dataset used for the training of the students.')
    parser.add_argument('--teacher', type=str, default='teacher', choices=['teacher', 'fg_att_student_iter1', 'fg_att_student_iter2'],
                        help='Teacher network used for the training of the students.')    
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='ID of the GPU used for the student training.')
    parser.add_argument('--workers', default=8, type=int,
                        help='Number of workers used for the student training.')
    parser.add_argument('--load_model', type=utils.bool_flag, default=False, 
                        help='Boolean flag indicating that the weights from a existing will be loaded.')
    parser.add_argument('--dims', type=int, default=512, 
                        help='Dimensionality of the input features.')
    
    # Student network options
    parser.add_argument('--student_type', type=str, default='fine-grained', choices=['fine-grained', 'coarse-grained'], 
                        help='Type of the student network.')
    parser.add_argument('--attention', type=utils.bool_flag, default=True,
                        help='Boolean flag indicating whether an Attention layer will be used. Applicable for both Student types.')
    parser.add_argument('--binarization', type=utils.bool_flag, default=False, 
                        help='Boolean flag indicating whether a Binarization layer will be used. Applicable only for Fine-grained Students.')
    parser.add_argument('--binary_bits', type=int, default=512, 
                        help='Number of bits used in the Binarization layer. Applicable only for Fine-grained Students when --binarization flag is true.')
    parser.add_argument('--transformer', type=utils.bool_flag, default=True, 
                        help='Boolean flag indicating whether a Transformer layer will be used. Applicable only for Coarse-grained Students.')
    parser.add_argument('--transformer_heads', type=int, default=8, 
                        help='Number of heads used in the multi-head attention layer of the Transformer. Applicable only for Coarse-grained Students when --transformer flag is true.')
    parser.add_argument('--transformer_feedforward_dims', type=int, default=2048, 
                        help='Number of dimensions of the feedforward network of the Transformer. Applicable only for Coarse-grained Students when --transformer flag is true.')
    parser.add_argument('--transformer_layers', type=int, default=1, 
                        help='Number of layers of the Transformer. Applicable only for Coarse-grained Students when --transformer flag is true.')
    parser.add_argument('--netvlad', type=utils.bool_flag, default=True,
                        help='Boolean flag indicating whether a NetVLAD layer will be used. Applicable only for Coarse-grained Students.')
    parser.add_argument('--netvlad_clusters', type=int, default=64,
                        help='Number of clusters used in NetVLAD. Applicable only for Coarse-grained Students when --netvlad flag is true.')
    parser.add_argument('--netvlad_outdims', type=int, default=1024,
                        help='Number of outdims used in output linear layer of the NetVLAD. Applicable only for Coarse-grained Students when --netvlad flag is true.')
    
    # Training process options
    parser.add_argument('--batch_sz', type=int, default=64,
                        help='Number of video pairs in each training batch.')
    parser.add_argument('--augmentation', type=utils.bool_flag, default=True,
                        help='Boolean flag indicating whether video temporal augmentations will be used.')
    parser.add_argument('--epochs', type=int, default=300, 
                        help='Number of epochs to train the student network.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                        help='Learning rate used during training')
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='Weight decay used during training')
    parser.add_argument('--r_parameter', type=float, default=1e-3, 
                        help='Parameter that determines the impact of similarity regularization loss')

    # Validation process options
    parser.add_argument('--val_hdf5', type=str, default=None,
                        help='Path to hdf5 file containing the features of the FIVR-5K dataset')
    parser.add_argument('--val_set', type=str, default="DSVR", choices=["DSVR", "CSVR", "ISVR"],
                        help='Set of the FIVR-5K used for validation. Applicable only when --val_hdf5 is provided.')
    parser.add_argument('--val_step', type=int, default=5,
                        help='Number of epochs to perform validation. Applicable only when --val_hdf5 is provided.')
    args = parser.parse_args()
    
    main(args)
