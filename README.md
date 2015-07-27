# Deep filter banks for texture recognition, description, and segmentation

The code provided runs the evaluation of RCNN and FV-CNN on various texture and material datasets (DTD, FMD, KTH-TIPS2b, ALOT), as well as for other domains: object (VOC07), scene (MIT Indoor), and fine-grained (CUB).
The results of these experiments are contained in Table 1 and 2 in the paper ** Deep Filter Banks for Texture Recognition and Segmentation, M. Cimpoi et al., CVPR 2015. ** and Tables 3, 4, 5, 6 in the paper **Deep filter banks for texture recognition, description, and segmentation, M. Cimpoi et al., http://arxiv.org/abs/1507.02620 

##   Getting starded

Once you have downloaded the code, make sure you installed the dependencies (see below).
Download the datasets you want to evaluate on, and link to them or copy them under data folder, in the location of your repository. Download the models (VGG-M, VGG-VD and AlexNet) in data/models. It is slightly faster to download them manually from here: http://www.vlfeat.org/matconvnet/pretrained/
Once done, simply run the `run_experiments.m` file.

In `texture_experiments.m` you could remove (or add) dataset names to the `datasetList` cell. Make sure you adjust the number of splits accordingly. The datasets are specified as {'dataset_name', <num_splits>} cells.

### Dependencies

The code relies on [vlfeat](http://www.vlfeat.org/), and [matconvnet], which should be downloaded and built before running the experiments.
Run git submodule update -i in the repository download folder.

To build vlfeat, go to <DEEP-FBANKS_DIR>/vlfeat and run make; ensure you have MATLAB executable and mex in the path.
To build matconvnet: in MATLAB, go to <DEEP-FBANKS_DIR>/matconvnet/matlab and run vl_compilenn; ensure you have CUDA installed, and nvcc in the path.

For LLC features (Table 3 in arxiv paper), please download the code from [http://www.robots.ox.ac.uk/~vgg/software/enceval_toolkit](http://www.robots.ox.ac.uk/~vgg/software/enceval_toolkit) and copy the following to the code folder (no subfolders!)

* `enceval/enceval-toolkit/+featpipem/+lib/LLCEncode.m`
* `enceval/enceval-toolkit/+featpipem/+lib/LLCEncodeHelper.cpp`
* `enceval/enceval-toolkit/+featpipem/+lib/annkmeans.m`

Create the corresponding dcnnllc encoder type (see the examples provided in run_experiments.m for BOVW, VLAD or FV).

### Paths and datasets

The `<DATASET_NAME>_get_database_.m` files generate the imdb file for each dataset. Make sure the datasets are copied or linked to manually in the data folder.

The datasets are stored in individual folders under data, in the current code folder, and experiment results are stored in data/exp01 folder, in the same location as the code. Alternatively, you could make data and experiments
symbolic links pointing to convenient locations.

Please be aware that the descriptors are stored on disk (in cache folder, under `data/exp01/<experiment-dir>`), and may require large amounts of free space (especially FV-CNN features).


### Dataset and evaluation

Describable Textures Dataset (DTD) is publicly available for download at:
[http://www.robots.ox.ac.uk/~vgg/data/dtd](http://www.robots.ox.ac.uk/~vgg/data/dtd). You can also download the  precomputed DeCAF features for DTD, the paper and evaluation results.

Our additional annotations for OpenSurfaces dataset are publicly available for download at:
http://www.robots.ox.ac.uk/~vgg/data/wildtex/

Code is available at:
TODO: GITHUB LINK

Code for CVPR14 paper (and Table 2 in arXiv paper):
http://www.robots.ox.ac.uk/~vgg/data/dtd/download/desctex.tar.gz

###   Citation

If you use the code and data please cite the following in your work:

FV-CNN and OpenSurfaces Additional Annotations:
@Article{Cimpoi15a,
  Author       = "Cimpoi, M. and Maji, S., Kokkinos, I. and Vedaldi, A.",
  Title        = "Deep Filter Banks for Texture Recognition, Description, and Segmentation"
  Journal      = "arXiv preprint arXiv:1507.02620",
  Year         = "2015",
}

@InProceedings{Cimpoi15,
  Author       = "Cimpoi, M. and Maji, S. and Vedaldi, A.",
  Title        = "Deep Filter Banks for Texture Recognition and Segmentation",
  Booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition",
  Year         = "2015",
}

DTD Dataset and IFV + DeCAF:
@inproceedings{cimpoi14describing,
  Author       = "M. Cimpoi and S. Maji and I. Kokkinos and S. Mohamed and A. Vedaldi",
  Title        = "Describing Textures in the Wild",
  Booktitle    = "Proceedings of the {IEEE} Conf. on Computer Vision and Pattern Recognition ({CVPR})",
  Year         = "2014",
}

