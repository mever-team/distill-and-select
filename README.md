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
    
* All feature files are in HDF5 format

## Distillation
*TODO*

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

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details

## Contact for further details about the project

Giorgos Kordopatis-Zilos (georgekordopatis@iti.gr)
