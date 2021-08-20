import torch
import utils
import argparse
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets.generators import DatasetGenerator
from model.students import FineGrainedStudent, CoarseGrainedStudent


@torch.no_grad()
def calculate_similarities_to_queries(model, queries, target, args):
    similarities = []
    for i, query in enumerate(queries):
        if query.device.type == 'cpu':
            query = query.to(args.gpu_id)
        sim = []
        for b in range(target.shape[0]//args.batch_sz_sim + 1):
            batch = target[b*args.batch_sz_sim: (b+1)*args.batch_sz_sim]
            if batch.shape[0] >= 4:
                sim.append(model.calculate_video_similarity(query, batch))
        sim = torch.mean(torch.cat(sim, 0))
        similarities.append(sim.cpu().numpy())
    return similarities 
    
    
@torch.no_grad()
def query_vs_target(model, dataset, args):
    # Create a video generator for the queries
    generator = DatasetGenerator(args.dataset_hdf5, dataset.get_queries())
    loader = DataLoader(generator, num_workers=args.workers, collate_fn=utils.collate_eval)

    # Extract features of the queries
    all_db, queries, queries_ids = set(), [], []
    print('\n> Extract features of the query videos')
    for video in tqdm(loader):
        video_features = video[0][0]
        video_id = video[2][0]
        if video_id:
            features = model.index_video(video_features.to(args.gpu_id))
            if not args.load_queries: features = features.cpu()
            all_db.add(video_id)
            queries.append(features)
            queries_ids.append(video_id)

    # Create a video generator for the database video
    generator = DatasetGenerator(args.dataset_hdf5, dataset.get_database())
    loader = DataLoader(generator, num_workers=args.workers, collate_fn=utils.collate_eval)
    
    # Calculate similarities between the queries and the database videos
    similarities = dict({query: dict() for query in queries_ids})
    print('\n> Calculate query-target similarities')
    for video in tqdm(loader):
        video_features = video[0][0]
        video_id = video[2][0]
        if video_id:
            features = model.index_video(video_features.to(args.gpu_id))
            sims = calculate_similarities_to_queries(model, queries, features, args)
            all_db.add(video_id)
            for i, s in enumerate(sims):
                similarities[queries_ids[i]][video_id] = float(s)
    
    print('\n> Evaluation on {}'.format(dataset.name))
    dataset.evaluate(similarities, all_db)

    
@torch.no_grad()
def queries_vs_database(model, dataset, args):
    # Create a video generator for the queries
    generator = DatasetGenerator(args.dataset_hdf5, dataset.get_queries())
    loader = DataLoader(generator, batch_size=args.batch_sz, num_workers=args.workers, collate_fn=utils.collate_eval)

    # Extract features of the queries
    all_db, queries, queries_ids = set(), [], []
    print('\n> Extract features of the query videos')
    for video in tqdm(loader):
        video_id = np.array(video[2])
        video_features = video[0][video_id != '']
        video_mask = video[1][video_id != '']
        video_id = video_id[video_id != '']
        if len(video_id) > 0:
            video_features = model.index_video(video_features.to(args.gpu_id), video_mask.to(args.gpu_id))
            all_db.update(video_id)
            queries.append(video_features)
            queries_ids.extend(video_id)
    queries = torch.cat(queries, 0)
    
    # Create a video generator for the database video
    generator = DatasetGenerator(args.dataset_hdf5, dataset.get_database())
    loader = DataLoader(generator, batch_size=args.batch_sz, num_workers=args.workers, collate_fn=utils.collate_eval)
    
    # Extract features of the targets
    targets, targets_ids = [], []
    print('\n> Extract features of the target videos')
    for video in tqdm(loader):
        video_id = np.array(video[2])
        video_features = video[0][video_id != '']
        video_mask = video[1][video_id != '']
        video_id = video_id[video_id != '']
        if len(video_id) > 0:
            video_features = model.index_video(video_features.to(args.gpu_id), video_mask.to(args.gpu_id))
            all_db.update(video_id)
            targets.append(video_features)
            targets_ids.extend(video_id)
    targets = torch.cat(targets, 0)
    
    # Calculate similarities between the queries and the database videos
    print('\n> Calculate query-target similarities')
    sims = model.calculate_video_similarity(queries, targets).cpu().numpy()
    similarities = dict({query: dict() for query in queries_ids})
    for i in range(sims.shape[0]):
        for j in range(sims.shape[1]):
            similarities[queries_ids[i]][targets_ids[j]] = float(sims[i, j])

    print('\n> Evaluation on {}'.format(dataset.name))
    dataset.evaluate(similarities, all_db)

    
if __name__ == '__main__':
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=80)
    parser = argparse.ArgumentParser(description='This is the code for the evaluation of the trained student on five datasets.', formatter_class=formatter)
    parser.add_argument('--dataset', type=str, required=True, choices=["FIVR-200K", "FIVR-5K", "CC_WEB_VIDEO", "SVD", "EVVE"],
                        help='Name of evaluation dataset.')
    parser.add_argument('--dataset_hdf5', type=str, required=True, 
                        help='Path to hdf5 file containing the features of the evaluation dataset')
    parser.add_argument('--student_path', type=str, default=None,
                        help='Path to a trained student network. If it is not provided, then the pretrained weights are used with the default architecture.')
    parser.add_argument('--student_type', type=str, default='fine-grained', choices=['fine-grained', 'coarse-grained'], 
                        help='Type of the student network.')
    parser.add_argument('--attention', type=utils.bool_flag, default=False,
                        help='Boolean flag indicating whether a Fine-grained Attention Student will be used.')
    parser.add_argument('--binarization', type=utils.bool_flag, default=False, 
                        help='Boolean flag indicating whether a Fine-grained Binarization Student will be used.')
    parser.add_argument('--batch_sz', type=int, default=32,
                        help='Number of videos processed in each batch. Aplicable only with Coarse-greained Students.')
    parser.add_argument('--batch_sz_sim', type=int, default=2048,
                        help='Number of feature tensors in each batch during similarity calculation.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='ID of the GPU used for the student evaluation.')
    parser.add_argument('--load_queries', type=utils.bool_flag, default=True,
                        help='Boolean flag indicating whether the query features will be loaded to the GPU memory. Aplicable only for Fine-grained Students.')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of workers used for video loading.')
    args = parser.parse_args()

    if 'CC_WEB' in args.dataset:
        from datasets import CC_WEB_VIDEO
        dataset = CC_WEB_VIDEO()
    elif 'FIVR' in args.dataset:
        from datasets import FIVR
        dataset = FIVR(version=args.dataset.split('-')[1].lower())
    elif 'EVVE' in args.dataset:
        from datasets import EVVE
        dataset = EVVE()
    elif 'SVD' in args.dataset:
        from datasets import SVD
        dataset = SVD()
    
    print('\n> Loading network')
    if args.student_path is not None:
        d = torch.load(args.student_path, map_location='cpu')
        student_args = d['args']
        if student_args.student_type == 'fine-grained':
            model = FineGrainedStudent(**vars(student_args))
            eval_function = query_vs_target
        elif student_args.student_type == 'coarse-grained':
            model = CoarseGrainedStudent(**vars(student_args))
            eval_function = queries_vs_database
        model.load_state_dict(d['model'])
    else:
        if args.student_type == 'fine-grained':
            if not args.attention and not args.binarization:
                raise Exception('No pretrained network for the given inputs. Provide either `--attention` or `--binarization` arguments as true for the pretrained fine-grained students.')
            model = FineGrainedStudent(attention=args.attention, 
                                       binarization=args.binarization,
                                       pretrained=True)
            eval_function = query_vs_target
        elif args.student_type == 'coarse-grained':
            model = CoarseGrainedStudent(pretrained=True)
            eval_function = queries_vs_database        
    model = model.to(args.gpu_id)
    model.eval()
    
    print(model)
    
    eval_function(model, dataset, args)
