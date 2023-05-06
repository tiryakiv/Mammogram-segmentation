# Digital Database for Screening Mammography segmentation annotation data

The images files in this directory are 66 Digital Database for Screening Mammography (DDSM) mammograms and the corresponding manual annotations of mammograms which show
the fibroglandular, adipose, and pectoral muscle tissue regions. To our knowledge, the dataset is the first publicly available breast tissue segmentation masks for screen film mammography in the world. The permission for the use of DDSM data is explained in our paper.
The dataset includes manual annotations for 16 Type A, 20 Type B, 17 Type C, and 13 Type D mammograms. Manual annotation file names are the original mammogram file name
concatenated with “_LI”, which stands for “labelled image”. Mammograms and the manual annotations have
a resolution of 960x480. 64, 128, 192, and 255 intensity pixels in the manual annotations show background, adipose tissue, fibroglandular tissue, and pectoral muscle
tissue regions, respectively. The images are grayscale. Mammograms and manual annotations  are located under "fgt_seg" and "fgt_seg_labels" subdirectories of
"train_valid" and "test" directories. These are the training, validation, and test mammograms that were used for modelling the mammogram segmentation in our article. We
have given the names of the cross-validation file names in the supplementary materials document. You may find the methods about preprocessing of mammograms and manual
annotations in our journal article.

The Authors
