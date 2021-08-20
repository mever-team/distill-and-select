import os
import utils
import torch
import argparse

from tqdm import tqdm
from datasets.generators import *
from torch.utils.data import DataLoader
from model.selector import SelectorNetwork
from sklearn.metrics import accuracy_score, f1_score


def main(args):
    print('\nInput Arguments')
    print('---------------')
    for k, v in sorted(dict(vars(args)).items()):
        print('%s: %s' % (k, str(v)))
    
    print('\n> Create generator of video pairs')
    print('>> loading pairs...')
    X_tr, X_val, y_tr, y_val = utils.generate_selector_dataset(threshold=args.threshold)
    train_dataset = SelectorPairGenerator(X_tr, y_tr, args)
    val_dataset = SelectorPairGenerator(X_val, y_val, args)
    
    print('\n> Building network')
    model = SelectorNetwork(**vars(args))
    model = model.to(args.gpu_id)
    model.train()
    print(model)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=args.learning_rate, 
                                 weight_decay=args.weight_decay)
    global_step = torch.zeros((1,))
    
    if args.load_model:
        print('>> loading network')
        d = torch.load(os.path.join(args.experiment_name, 'model_{}.pth'.format(
            model.get_name())), map_location='cpu')
        model.load_state_dict(d['model'])
        if 'optimizer' in d:
            optimizer.load_state_dict(d['optimizer'])
        if 'global_step' in d:
            global_step = d.pop('global_step')

    print('\n> Start training')
    acc, f1 = 0., 0.
    for epoch in range(args.epochs):
        train_dataset.next_epoch(size=args.pairs_per_class)
        loader = DataLoader(train_dataset, num_workers=args.workers, 
                            shuffle=True, batch_size=args.batch_sz,
                            collate_fn=utils.collate_selector)
        model.train()
        losses = []
        pbar = tqdm(loader, desc='epoch {}'.format(epoch), unit='iter')
        for videos, masks, similarities, y_true in pbar:
            optimizer.zero_grad()

            self_sims = model.index_video(videos.to(args.gpu_id), masks.to(args.gpu_id))
            queries, targets = torch.chunk(self_sims, 2, dim=0)
            y_pred = model(torch.cat([queries, targets, similarities.unsqueeze(1).to(args.gpu_id)], 1))
            y_true = y_true.unsqueeze(1).to(args.gpu_id)

            loss = criterion(y_pred, y_true)

            loss.backward()
            optimizer.step()
            global_step += 1

            losses.append(loss.detach().cpu().numpy())
            if len(losses) % 5 == 0:
                pbar.set_postfix(loss=np.mean(losses), accuracy=acc, f1_score=f1)

        with torch.no_grad():
            loader = DataLoader(val_dataset, num_workers=args.workers, 
                                shuffle=False, batch_size=args.batch_sz,
                                collate_fn=utils.collate_selector)
            model.eval()
            y_pred, y_true = [], []
            for videos, masks, similarities, y in loader:
                self_sims = model.index_video(videos.to(args.gpu_id), masks.to(args.gpu_id))
                queries, targets = torch.chunk(self_sims, 2, dim=0)
                preds = model(torch.cat([queries, targets, similarities.unsqueeze(1).to(args.gpu_id)], 1))
                
                y_true.extend(y.cpu().numpy().tolist())
                y_pred.extend((preds.squeeze(1).cpu().numpy() > 0.5).tolist())

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        utils.save_model(args, model, optimizer, global_step)

            
if __name__ == '__main__':    
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=80)
    parser = argparse.ArgumentParser(description='This is the training code for knowledge distillation of video similarity', formatter_class=formatter)
    # Experiment options
    parser.add_argument('--experiment_path', type=str, required=True, 
                        help='Path of the experiment where the weights of the trained networks will be stored.')
    parser.add_argument('--trainset_hdf5', type=str, required=True, 
                        help='Path to hdf5 file containing the features of the DnS-100K trainset.')
    parser.add_argument('--fine_student', type=str, default='fg_att_student_iter2', choices=['fg_att_student_iter2', 'fg_bin_student_iter2'],
                        help='Fine-grained Student used for the training of the selector network.')
    parser.add_argument('--coarse_student', type=str, default='cg_student_iter2', choices=['cg_student_iter2'],
                        help='Coarse-grained Student used for the training of the selector network.') 
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='ID of the GPU used for the student training.')
    parser.add_argument('--workers', default=8, type=int,
                        help='Number of workers used for the student training.')
    parser.add_argument('--load_model', type=utils.bool_flag, default=False, 
                        help='Boolean flag indicating that the weights from a existing will be loaded.')
    parser.add_argument('--dims', type=int, default=512, 
                        help='Dimensionality of the input features.')
    
    # Student network options
    parser.add_argument('--hidden_size', type=int, default=100,
                        help='Number of dimensions of the hidden layers of the selector network.')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of hidden layers used for the selector network.')

    # Training process options
    parser.add_argument('--threshold', type=float, default=0.2,
                        help='Threshold value used to calculate the binary label function.')
    parser.add_argument('--batch_sz', type=int, default=64,
                        help='Number of video pairs in each training batch.')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of epochs to train the student network.')
    parser.add_argument('--pairs_per_class', type=int, default=5000,
                        help='Number of video pairs sampled each epoch for each class.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                        help='Learning rate used during training')
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='Weight decay used during training')
    args = parser.parse_args()
    
    main(args)
