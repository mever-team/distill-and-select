# DnS: Distill-and-Select for Efficient and Accurate Video Indexing and Retrieval
This repository contains the PyTorch implementation of the paper [DnS: Distill-and-Select for Efficient and Accurate Video Indexing and Retrieval](https://arxiv.org/abs/2106.13266).

## Prerequisites
* Python 3
* PyTorch >= 1.1
* Torchvision >= 0.4

## Preparation

### Installation

* Clone this repo:
```bash
git clone https://github.com/mever-team/distill-and-select
cd distill-and-select
```
* You can install all the dependencies by
```bash
pip install -r requirements.txt
```
or
```bash
conda install --file requirements.txt
```

### Feature files
* We provide our extracted features for all datasets to facilitate reproducibility for future research.

* Download the feature files of the dataset you want:
    * [DnS-100K](https://mever.iti.gr/distill-and-select/features/dns_100k.hdf5) (219 GB)
    * [FIVR-200K](https://mever.iti.gr/distill-and-select/features/fivr_200k.hdf5) (406 GB), [FIVR-5K](https://mever.iti.gr/distill-and-select/features/fivr_5k.hdf5) (8.7 GB)
    * [CC_WEB_VIDEO](https://mever.iti.gr/distill-and-select/features/cc_web_video.hdf5) (31 GB)
    * [SVD](https://mever.iti.gr/distill-and-select/features/svd.hdf5) (150 GB)
    * [EVVE](https://mever.iti.gr/distill-and-select/features/evve.hdf5) (9 GB)
    * [VCDB](https://mever.iti.gr/distill-and-select/features/vcdb.hdf5) (118 GB)
    
* All feature files are in HDF5 format

## Distillation
We provide the code for training and evaluation of our student models.

### Student training

* To train a fine-grained student, run the `train_student.py` given `fine-grained` as value to the `--student_type` argument, as in the following command:
```bash
python train_student.py --student_type fine-grained --experiment_path experiments/DnS_students --trainset_hdf5 /path/to/dns_100k.hdf5
```

* You can train an attention or binarization fine-grained students by setting either the `--attention` or `--binarization` flags to `true`, respectively.

For fine-grained attention students:
```bash
python train_student.py --student_type fine-grained --binarization false --attention true --experiment_path /path/to/experiment/ --trainset_hdf5 /path/to/dns_100k.hdf5
```

For fine-grained binarization students:
```bash
python train_student.py --student_type fine-grained --binarization true --attention false --experiment_path /path/to/experiment/ --trainset_hdf5 /path/to/dns_100k.hdf5
```

* To train a coarse-grained students, provide `coarse-grained` to the `--student_type` argument:
```bash
python train_student.py --student_type coarse-grained --experiment_path /path/to/experiment/ --trainset_hdf5 /path/to/dns_100k.hdf5 --attention true --learning_rate 1e-5
```

* Provide one of the `teacher`, `fg_att_student_iter1`, `fg_att_student_iter2` to the `--teacher` argument in odrder to train a student with a different teacher:
```bash
python train_student.py --teacher fg_att_student_iter2 --experiment_path /path/to/experiment/ --trainset_hdf5 /path/to/dns_100k.hdf5
```

* You can optionally perform validation with FIVR-5K by providing its HDF5 file to the `--val_hdf5` and choosing one of the DSVR, CSVR, ISVR sets 
with the `--val_set` argument:
```bash
python train_student.py --student_type coarse-grained --val_hdf5 /path/to/fivr_5k.hdf5 --val_set ISVR --experiment_path /path/to/experiment/ --trainset_hdf5 /path/to/dns_100k.hdf5 --learning_rate 1e-5
```
### Student Evaluation
* Choose one of the `FIVR-5K`, `FIVR-200K`, `CC_WEB_VIDEO`, `SVD`, or `EVVE` datasets to evaluate your models.

* For the evaluation of the students, run the `evaluation_student.py` script by providing the path to the `.pth` model to the `--student_path` argument, as in the following command:
```bash
python evaluation_student.py --student_path experiments/DnS_students/model_fg_att_student.pth --dataset FIVR-5K --dataset_hdf5 /path/to/fivr_200k.hdf5
```

* If you don't pass any value to the `--student_path`, a pretrained model will be selected:
```bash
python evaluation_student.py --student_type fine-grained --attention true --dataset FIVR-5K --dataset_hdf5 /path/to/fivr_200k.hdf5
```


## Selection
We also provide the code for training of the selector network and the evaluation of our overall DnS framework.

### Selector training

* To train a selector network, run the `train_selector.py` as in the following command:
```bash
python train_selector.py --experiment_path experiments/DnS_students --trainset_hdf5 /path/to/dns_100k.hdf5
```

* Provide different values to `--threshold` argument to train the selector network with different label functions.

### DnS Evaluation

* For the evaluation of the DnS framework, run the `evaluation_dns.py` script by providing the path to the `.pth` model to the corresponding network arguments, as in the following command:
```bash
python evaluation_dns.py --selector_network_path experiments/DnS_students/model_selector_network.pth --dataset FIVR-5K --dataset_hdf5 /path/to/fivr_200k.hdf5
```

* If you don't pass any value to the network path argument, then the pretrained model will be selected. E.g. to evalute DnS with the Fine-grained Attention Student:
```bash
python evaluation_dns.py --attention true --dataset FIVR-5K --dataset_hdf5 /path/to/fivr_200k.hdf5
```

* Provide different values to `--percentage` argument to sent different number of video pairs for reranking to the Fine-grained student. 
Given the value `all`, it runs evaluation for all dataset percentages.


## Use our pretrained models
We also provide our pretrained models trained with the `fg_att_student_iter2` teacher.

* Load our pretrained models as follows:
```python
from model.feature_extractor import FeatureExtractor
from model.students import FineGrainedStudent, CoarseGrainedStudent
from model.selector import SelectorNetwork

# The feature extraction network used in out experiments
feature_extractor = FeatureExtractor(dims=512).eval()

# Our Fine-grained Students
fg_att_student = FineGrainedStudent(pretrained=True, attention=True).eval()
fg_bin_student = FineGrainedStudent(pretrained=True, binarization=True).eval()

# Our Coarse-grained Students
cg_student = CoarseGrainedStudent(pretrained=True).eval()

# Our Selector Networks
selector_att = SelectorNetwork(pretrained=True, attention=True).eval()
selector_bin = SelectorNetwork(pretrained=True, binarization=True).eval()
```

* First, extract video features by providing a video tensor to feature extractor (similar as [here](https://github.com/MKLab-ITI/visil/tree/pytorch#use-visil-in-your-python-code))
```python
video_features = feature_extractor(video_tensor)
```

* Use the `index_video()` function providing video features to extract video representations for the student and selector networks
```python
fg_features = fg_att_student.index_video(video_features)
cg_features = cg_student.index_video(video_features)
sn_features = selector_att.index_video(video_features)
```

* Use the `calculate_video_similarity()` function providing query and target features to calculate similarity based on the student networks.
```python
fine_similarity = fg_att_student.calculate_video_similarity(query_fg_features, target_fg_features)
coarse_similarity = cg_student.calculate_video_similarity(query_cg_features, target_cg_features)
```

* To calculate the selector's score for a video pair, call the selector network by providing the features extracted 
for each video and their coarse similarity
```python
selector_features = torch.cat([query_sn_features, target_sn_features, coarse_similarity], 1)
selector_scores = selector_att(selector_features)
```

## Citation
If you use this code for your research, please cite our paper.
```
@article{kordopatis2021dns,
  title={DnS: Distill-and-Select for Efficient and Accurate Video Indexing and Retrieval},
  author={Kordopatis-Zilos, Giorgos and Tzelepis, Christos and Papadopoulos, Symeon and Kompatsiaris, Ioannis and Patras, Ioannis},
  journal={arXiv preprint arXiv:2106.13266},
  year={2021}
}
```

## Related Projects
**[ViSiL](https://github.com/MKLab-ITI/visil)** **[FIVR-200K](https://github.com/MKLab-ITI/FIVR-200K)**

## Acknowledgements
This work has been supported by the projects WeVerify and MediaVerse, partially funded by the European Commission under contract number 825297 and 957252, respectively, and DECSTER funded by EPSRC under contract number EP/R025290/1.

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details

## Contact for further details about the project

Giorgos Kordopatis-Zilos (georgekordopatis@iti.gr)
