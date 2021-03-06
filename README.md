This is the code for the paper

# Geometric and Textural Augmentation for Domain Gap Reduction

### [Project Page](https://github.com/xch-liu/geom-tex-dg) | [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Geometric_and_Textural_Augmentation_for_Domain_Gap_Reduction_CVPR_2022_paper.pdf) | [Poster](https://github.com/xch-liu/geom-tex-dg) | [Video](https://github.com/xch-liu/geom-tex-dg)

## Preresquisites

### Testbed Install: 
We use [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch) as the testbed and the code is based on it.

```bash
# Create the conda environment (make sure conda is installed)
conda create -n dassl python=3.7
conda activate dassl

# Install dependencies
cd Dassl/
pip install -r requirements.txt

# Install torch (version >= 1.7.1) and torchvision based on your cuda version 
conda install pytorch torchvision cudatoolkit=your_cuda_version -c pytorch

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
```

### Datasets Install:
We use three commonly used multi-domain datasets (please download datasets into this [folder](https://github.com/xch-liu/geom-tex-dg/tree/main/Dassl/data)):

* PACS ([Li et al., 2017](https://arxiv.org/abs/1710.03077)) | Download Link: [google drive](https://drive.google.com/open?id=1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE).
* Office-Home-DG ([Venkateswara et al., 2017](https://arxiv.org/abs/1706.07522)) | Download Link: [google drive](https://drive.google.com/open?id=1gkbf_KaxoBws-GWT3XIPZ7BnkqbAxIFa).
* Digits-DG | Download Link: [google drive](https://drive.google.com/open?id=15V7EsHfCcfbKgsDmzQKj_DfXt_XYp_P7).

Download the style predictor [model](https://drive.google.com/file/d/1SR6nred3nUxTHxreuQ5jyzWqwshGmDYu/view?usp=sharing) into this [folder](https://github.com/xch-liu/geom-tex-dg/tree/main/Dassl/dassl/modeling/backbone/styleaugment/styleaug/checkpoints).

## Training and Testing

```bash
cd gta-dgr/scripts/

# Training on PACS
bash pacs.sh

# Training on Office-Home
bash officehome.sh

# Training on Digits-DG
bash digits.sh
```

If you find this code useful for your research, please cite
```
@InProceedings{Liu22GTDG, 
  author={Xiao-Chang Liu and Yong-Liang Yang and Peter Hall},
  title={Geometric and Textural Augmentation for Domain Gap Reduction},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```
