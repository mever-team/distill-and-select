import time
import torch
import utils
import argparse
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from model.selector import SelectorNetwork
from datasets.generators import DatasetGenerator
from model.students import FineGrainedStudent, CoarseGrainedStudent


@torch.no_grad()
def calculate_similarities_to_queries(model, queries, target, args):
    similarities = []
    total_time = 0.
    for i, query in enumerate(queries):
        if query.device.type == 'cpu':
            query = query.to(args.gpu_id)
        sim = []
        start_time = time.time()
        for b in range(target.shape[0]//args.batch_sz_sim + 1):
            batch = target[b*args.batch_sz_sim: (b+1)*args.batch_sz_sim]
            if batch.shape[0] >= 4:
                sim.append(model.calculate_video_similarity(query, batch))
        sim = torch.mean(torch.cat(sim, 0))
        total_time += time.time() - start_time
        similarities.append(sim.cpu().numpy())
    return similarities, total_time


def get_similarities_for_percentage(coarse_similarities, fine_similarities, selector_scores, percentage, mask):
    similarities = dict()
    for query, query_sims in coarse_similarities.items():
        similarities[query] = np.copy(query_sims)
        idx = selector_scores[query][:int(percentage*np.sum(mask))]
        if len(idx):
            sims = np.copy(query_sims)[mask]
            sims[idx] = fine_similarities[query][mask][idx]
            similarities[query][mask] = sims
    return similarities


@torch.no_grad()
def main(fine_student, coarse_student, selector_network, dataset, args):
    # Create a video generator for the queries
    generator = DatasetGenerator(args.dataset_hdf5, dataset.get_queries())
    loader = DataLoader(generator, num_workers=args.workers, collate_fn=utils.collate_eval)

    # Extract features of the queries
    all_db, queries_ids, queries_fs, queries_cs, queries_sn = set(), [], [], [], []
    print('\n> Extract features of the query videos')
    for video in tqdm(loader):
        video_features = video[0][0].to(args.gpu_id)
        video_id = video[2][0]
        
        if len(video_id) == 0:
            continue
        queries_ids.append(video_id)
        all_db.add(video_id)
        
        # Extract features of the query video
        fine_features = fine_student.index_video(video_features)
        if not args.load_queries: fine_features = fine_features.cpu()
        coarse_features = coarse_student.index_video(video_features).cpu()
        selector_features = selector_network.index_video(video_features).cpu()
        
        queries_fs.append(fine_features)
        queries_cs.append(coarse_features)
        queries_sn.append(selector_features)
            
    # Initialize similarities of the two students
    fine_similarities = dict({query: np.zeros(len(dataset.get_database())) for query in queries_ids})
    coarse_similarities = dict({query: np.zeros(len(dataset.get_database())) for query in queries_ids})
    
    # Create a video generator for the database video
    generator = DatasetGenerator(args.dataset_hdf5, dataset.get_database())
    loader = DataLoader(generator, num_workers=args.workers, collate_fn=utils.collate_eval)

    targets_ids, targets_cs, targets_sn, available_video_mask = [], [], [], []
    storage_fs, storage_cs, storage_sn = [], [], []
    time_fs, time_cs, time_sn = [], [], []
    print('\n> Extract features of the target videos and calculate fine-grained similarities')
    for j, video in enumerate(tqdm(loader)):
        video_features = video[0][0].to(args.gpu_id)
        video_id = video[2][0]
        
        if len(video_id) == 0:
            available_video_mask.append(False)
            continue
        available_video_mask.append(True)
        targets_ids.append(video_id)
        all_db.add(video_id)
        
        # Extract features of the target video
        fine_features = fine_student.index_video(video_features)
        coarse_features = coarse_student.index_video(video_features).cpu()
        selector_features = selector_network.index_video(video_features).cpu()

        targets_cs.append(coarse_features)
        targets_sn.append(selector_features)
        
        storage_fs.append(fine_features.nelement() * (1/8 if args.binarization else 4))
        storage_cs.append(coarse_features.nelement() * 4)
        storage_sn.append(selector_features.nelement() * 4)
        
        # Calculate fine-grained similarities
        similarities, total_time = calculate_similarities_to_queries(fine_student, queries_fs, fine_features, args)
        time_fs.append(total_time)
        for i, s in enumerate(similarities):
            fine_similarities[queries_ids[i]][j] = float(s / 2. + 0.5)
    targets_sn = torch.cat(targets_sn, 0)

    print('\n> Calculate coarse-grained similarities and selector\'s scores')
    selector_scores = dict()
    for i, query in enumerate(tqdm(queries_ids)):
        start_time = time.time()
        # Calculate coarse-grained similarities
        similarities = [coarse_student.calculate_video_similarity(queries_cs[i], t) for t in targets_cs]
        time_cs.append(time.time() - start_time)
        similarities = torch.cat(similarities)
        coarse_similarities[query][available_video_mask] = similarities.squeeze(-1).numpy()
        
        start_time = time.time()
        # Calculate query-target scores based on the selector network
        selector_features = queries_sn[i].repeat(len(targets_ids), 1)
        selector_features = torch.cat([selector_features, targets_sn, similarities], 1).to(args.gpu_id)
        scores = selector_network(selector_features).squeeze(-1)
        selector_scores[query] = torch.argsort(scores, descending=True).cpu().numpy()
        time_sn.append(time.time() - start_time)
    
    print('\n> Calculate results')
    print('\n---Storage requirements---')
    storage_fs = np.sum(storage_fs) / len(targets_ids)  / 1024
    print('Fine-grained Student: {} KB per video'.format(np.round(storage_fs)))
    storage_cs = np.sum(storage_cs) / len(targets_ids)  / 1024
    print('Coarse-grained Student: {} KB per video'.format(np.round(storage_cs, 4)))
    storage_dns = storage_fs + storage_cs + np.sum(storage_sn) / len(targets_ids)  / 1024
    print('DnS framework: {} KB per video'.format(np.round(storage_dns)))
    
    print('\n---Time requirements---')
    time_fs = np.sum(time_fs) / len(queries_ids)
    print('Fine-grained Student: {} sec per query'.format(np.round(time_fs, 4)))
    time_cs = np.sum(time_cs) / len(queries_ids)
    print('Coarse-grained Student: {} sec per query'.format(np.round(time_cs, 4)))
    
    if args.percentage == 'all':
        percentages = [np.round(i * 0.05, 2) for i in range(21)]
        time_dns = np.round([time_fs * p + time_cs + np.sum(time_sn) / len(queries_ids) for p in percentages], 4)
        print('DnS framework: {} sec per query'.format('\t'.join(map(str, time_dns))))
        
        print('\n---Retrieval results---')
        mAPs = dict()
        for p in percentages:
            similarities = get_similarities_for_percentage(coarse_similarities, fine_similarities, 
                                                           selector_scores, p, available_video_mask)
            results = dataset.evaluate(similarities, all_db, verbose=False)
            for k, v in results.items():
                if k not in mAPs: mAPs[k] = []
                mAPs[k].append(np.round(v, 4))
        print('perc.:\t{}'.format('\t'.join(['{}%'.format(int(p*100)) for p in percentages])))
        for k, v in mAPs.items():
            print('{}:\t{}'.format(k, '\t'.join(map(str, v))))
    else:
        time_dns = time_fs * float(args.percentage) + time_cs + np.sum(time_sn) / len(queries_ids)
        print('DnS: {} sec per query'.format(np.round(time_dns, 4)))

        print('\n---Retrieval results---')
        similarities = get_similarities_for_percentage(coarse_similarities, fine_similarities, 
                                                       selector_scores, float(args.percentage),
                                                       available_video_mask)
        dataset.evaluate(similarities, all_db)

    
if __name__ == '__main__':
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=80)
    parser = argparse.ArgumentParser(description='This is the code for the evaluation of the trained student on five datasets.', formatter_class=formatter)
    parser.add_argument('--dataset', type=str, required=True, choices=["FIVR-200K", "FIVR-5K", "CC_WEB_VIDEO", "SVD", "EVVE"],
                        help='Name of evaluation dataset.')
    parser.add_argument('--dataset_hdf5', type=str, required=True, 
                        help='Path to hdf5 file containing the features of the evaluation dataset')
    parser.add_argument('--percentage', type=str, default='all',
                        help='Dataset percentage sent to the Fine-grained Student for reranking. Providing \'all\' evaluates system\'s performance for all dataset percentages.')
    parser.add_argument('--attention', type=utils.bool_flag, default=False,
                        help='Boolean flag indicating whether a Fine-grained Attention Student will be used.')
    parser.add_argument('--binarization', type=utils.bool_flag, default=False, 
                        help='Boolean flag indicating whether a Fine-grained Binarization Student will be used.')
    parser.add_argument('--fine_student_path', type=str, default=None,
                        help='Path to a trained Fine-grained Student. If it is not provided, then the pretrained weights are used with the default architecture.')
    parser.add_argument('--coarse_student_path', type=str, default=None,
                        help='Path to a trained Coarse-grained Student. If it is not provided, then the pretrained weights are used with the default architecture.')
    parser.add_argument('--selector_network_path', type=str, default=None,
                        help='Path to a trained Selector Network. If it is not provided, then the pretrained weights are used with the default architecture.')
    parser.add_argument('--batch_sz', type=int, default=32,
                        help='Number of videos processed in each batch. Aplicable only with Coarse-greained Students.')
    parser.add_argument('--batch_sz_sim', type=int, default=2048,
                        help='Number of feature tensors in each batch during similarity calculation.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='ID of the GPU used for the student evaluation.')
    parser.add_argument('--load_queries', type=utils.bool_flag, default=True,
                        help='Boolean flag indicating whether the query fine-grained features will be loaded to the GPU memory. Aplicable only for Fine-grained Students.')
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
    
    print('\n> Loading Fine-grained Student')
    if args.fine_student_path is not None:
        d = torch.load(args.fine_student_path, map_location='cpu')
        fine_student = FineGrainedStudent(**vars(d['args']))
        fine_student.load_state_dict(d['model'])
    else:
        if not args.attention and not args.binarization:
            raise Exception('No pretrained network for the given inputs. Provide either `--attention` or `--binarization` arguments as true for the pretrained fine-grained students.')
        fine_student = FineGrainedStudent(attention=args.attention, 
                                          binarization=args.binarization, 
                                          pretrained=True)
    fine_student = fine_student.to(args.gpu_id)
    fine_student.eval()
    print('>> Network architecture')
    print(fine_student)
    
    print('\n> Loading Coarse-grained Student')
    if args.coarse_student_path is not None:
        d = torch.load(args.coarse_student_path, map_location='cpu')
        coarse_student = CoarseGrainedStudent(**vars(d['args']))
        coarse_student.load_state_dict(d['model'])
    else:
        coarse_student = CoarseGrainedStudent(pretrained=True)
    coarse_student = coarse_student.to(args.gpu_id)
    coarse_student.eval()
    print('>> Network architecture')
    print(coarse_student)
    
    print('\n> Loading Selector Network')
    if args.selector_network_path is not None:
        d = torch.load(args.selector_network_path, map_location='cpu')
        selector_network = SelectorNetwork(**vars(d['args']))
        selector_network.load_state_dict(d['model'])
    else:
        if not args.attention and not args.binarization:
            raise Exception('No pretrained network for the given inputs. Provide either `--attention` or `--binarization` arguments as true for the pretrained fine-grained students.')
        if args.fine_student_path is not None or args.coarse_student_path is not None:
            print('[WARNING] The pretrained Selector Network has been trained with the provided pretrained students. Using it with custom students may result in inaccurate predictions.')
        selector_network = SelectorNetwork(attention=args.attention, 
                                           binarization=args.binarization, 
                                           pretrained=True)
    selector_network = selector_network.to(args.gpu_id)
    selector_network.eval()
    print('>> Network architecture')
    print(selector_network)
    
    main(fine_student, coarse_student, selector_network, dataset, args)
