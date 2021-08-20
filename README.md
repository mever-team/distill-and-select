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
    * [FIVR-200K](https://mever.iti.gr/distill-and-select/features/fivr_200k.hdf5) (406 GB)
    * [CC_WEB_VIDEO](https://mever.iti.gr/distill-and-select/features/cc_web_video.hdf5) (31 GB)
    * [SVD](https://mever.iti.gr/distill-and-select/features/svd.hdf5) (150 GB)
    * [EVVE](https://mever.iti.gr/distill-and-select/features/evve.hdf5) (9 GB)
    * [VCDB](https://mever.iti.gr/distill-and-select/features/vcdb.hdf5) (118 GB)
    
* All feature files are in HDF5 format

## Distillation
We provide the code for training and evaluation of our student models 

### Student training

* To train a fine-grained student, run the `train_student.py` given `fine-grained` as value to the `--student_type` argument, as in the following command:
```bash
python train_student.py --student_type fine-grained --experiment_path experiments/DnS_students --trainset_hdf5 /path/to/dns_100k.hdf5
```

* You can train an attention or binarization fine-grained students by setting either the `--attention` or `--binarization` flags to `true`, respectively:
```bash
python train_student.py --student_type fine-grained --binarization true --experiment_path /path/to/experiment/ --trainset_hdf5 /path/to/dns_100k.hdf5
```

* To train a coarse-grained students, provide `coarse-grained` to the `--student_type` argument:
```bash
python train_student.py --student_type coarse-grained --experiment_path /path/to/experiment/ --trainset_hdf5 /path/to/dns_100k.hdf5
```

* Provide one of the `teacher`, `fg_att_student_iter1`, `fg_att_student_iter2` to the `--teacher` argument in odrder to train a student with a different teacher:
```bash
python train_student.py --teacher fg_att_student_iter2 --experiment_path /path/to/experiment/ --trainset_hdf5 /path/to/dns_100k.hdf5
```

### Evaluation
* Choose one of the `FIVR-5K`, `FIVR-200K`, `CC_WEB_VIDEO`, `SVD`, or `EVVE` datasets to evaluate your models.

* For the evaluation of the students, run the `evaluation.py` script by providing the path to the `.pth` model to the `--student_path` argument, as in the following command:
```bash
python evaluation_student.py --student_path experiments/DnS_students/model_fg_att_student.pth --dataset FIVR-5K --dataset_hdf5 /path/to/fivr_200k.hdf5 --load_queries true
```

* If you don't pass any value to the `--student_path`, a pretrained model will be selected:
```bash
python evaluation_student.py --student_type fine-grained --attention true --dataset FIVR-5K --dataset_hdf5 /path/to/fivr_200k.hdf5 --load_queries true
```


## Selection
*TODO*

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
