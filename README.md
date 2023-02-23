# Mammogram-segmentation

The codes in this repository belong to the methodology in the manuscript entitled 'Deep learning-based multi-label tissue segmentation and density assessment from mammograms'. We implemented various types of U-nets including training from scratch and transfer learning for mammogram segmentation [1,2]. We distribute our codes with a GNU GPLv3 license. 

train_valid*.ipynb files are the jupyter notebook files that shows the codes and results of the training and validation process. vXXX shows the version of each file. In each version, we validate performance of a specific model with different network depths, types of connections, loss functions, and transfer learning. We used five-fold cross-validation. The valid_performance_mammo_seg.xlsx file shows which file version is associated with the investigation situation. The first column in the excel file shows the model number which is indicated by the train&valid ipynb files’ last name. Performance values are provided for each validation fold separately in the table columns (val_dice1, val_dice2, …, val_dice5, val_acc1, val_acc2, … , val_acc5)  [3-13].

modelXLHzc* are the U-net model creation files that has 'X' network depth. modelUnetVGG16, modelUnetVGG19, and modelUnetResNet50 are the U-net model files that have coefficients from the VGG16, the VGG19, and the ResNet50 networks [14-17].

data*.py files are the data prepration files required for training, validating, and testing deep learning models. data.py is used when the method is transfer learning. data_1ch.py is used when the method is learning from scratch. 

Segmentation performance evaluation codes are provided in eval_performance.m and metrics.m files under the test_img folder. Open eval_performance with Octave or Matlab, run the codes and you will see the results of evaluation metrics.

The mammogram segmentation ground truth annotation images and mammograms that were used in our study are given in the "segmentation_data" directory. The label images are 960x480 pixels where 64, 128, 192, and 255, pixel intensities represent background, adipose, fibroglandular, and pectoral muscle tissues, respectively.

## Testing instructions:
Install Python 3.6, Tensorflow, Keras, numpy, skimage, and jupyter notebook to your computer. To test the pretrained model, you do not have to have a graphic processing unit. Here are the steps to test the pretrained model: Download all of the files and open test_ResNet50Unet_fgt_2dec21_v001.ipynb using Jupyter notebook. Download the ResNet50Unet from the following link:

https://drive.google.com/file/d/1aGidcMxjubGsYZlD8WA8HHZ-kNekcm8d/view?usp=sharing

Execute codes and you will see the predictions under the test folder. To evaluate the test performance, run the eval_performance.m file using Octave or Matlab. You will see performance evaluation results in terms of accuracy, Dice’s similarity coefficient(DSC), and intersection over union (IoU). If you want to test the segmentation model with your own images, make sure to rename them from 0 to n-1, where n is the number of images.

## Training instructions:
You should have a GPU on your computer to train a model. If you have a GPU on your computer, make sure that you install drivers correctly. This requires attention. We found the following youtube video by Dr. Jeff Heaton very useful for instructions about installing Tensorflow Keras with a GPU for Windows operating systems: https://www.youtube.com/watch?v=-Q6SM_usn84 He also has installation instruction videos for other OS, so check to see his channel. Install Python 3.6, Tensorflow, Keras, numpy, skimage, and jupyter notebook to your computer. You can also train your own model on Google Colab but there are some time restrictions. You can see the codes and segmentation results of five-layer Unet with Tversky loss function in our paper, in train_valid_unet5L_fgt_27nov21_v012.ipynb. You can see the training results of ResNet50-U-net in train_valid_uResNet50_tve_fgt_3dec21_v020.ipynb. If you want to train a model yourself, download mammogram dataset, apply pre-processing described in the paper, move the mammograms and labels to the folders, open one of the train_valid_*.ipynb with jupyter notebook, and run the codes. Lower the batch size if you get memory errors.

For more information, please refer to our manuscript or contact us by sending an e-mail to corresponding author (tiryakiv@siirt.edu.tr). Please consider citing our following article if you find the content useful for your research. 

Volkan Müjdat Tiryaki, Veysel Kaplanoğlu, Deep learning-based multi-label tissue segmentation and density assessment from mammograms, IRBM, 2022, https://doi.org/10.1016/j.irbm.2022.05.004. ISSN 1959-0318 (https://www.sciencedirect.com/science/article/pii/S1959031822000562) 

Volkan Müjdat TİRYAKİ, Veysel KAPLANOĞLU.

# REFERENCES

Ronneberger O, Fischer P, Brox T. U-net: Convolutional networks for biomedical image segmentation. Lect. Notes Comput. Sci. (including Subser. Lect. Notes Artif. Intell. Lect. Notes Bioinformatics) 2015;9351:234–241.

Falk T, Mai D, Bensch R, Çiçek Ö, Abdulkadir A, Marrakchi Y, Böhm A, Deubner J, Jäckel Z, Seiwald K, Dovzhenko A, Tietz O, Bosco CD, Walsh S, Saltukoglu D, Tay TL, Prinz M, Palme K, Simons M, Diester I, Brox T, Ronneberger O. U-Net: deep learning for cell counting, detection, and morphometry. Nat. Methods 2018;16:67–70.

Kingma DP, Ba JL. Adam: A method for stochastic optimization. 3rd Int. Conf. Learn. Represent. ICLR 2015 - Conf. Track Proc. 2015:1–15.

Zhixuhao. Implementation of deep learning framework -- Unet, using Keras. 2019.

Ioffe S, Szegedy C. Batch normalization: Accelerating deep network training by reducing internal covariate shift. 32nd Int. Conf. Mach. Learn. ICML 2015 2015;1:448–456.

Maas AL, Hannun AY, Ng AY. Rectifier nonlinearities improve neural network acoustic models. Proc. 30th Int. Conf. Mach. Learn. 2013;30.

Srivastava N, Hinton GE, Krizhevsky A, Salakhutdinov I, Salakhutdinov R. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. J. Mach. Learn. Res. 2014;15:1929–1958.

HZCTony. Unet : multiple classification using Keras. github 2019.

Yi-de M, Qing L, Zhi-bai Q. Automated Image Segmentation Using Improved PCNN Model Based on Cross-Entropy. In: Proc. 2004 Int. Symp. Intell. Multimedia, Video Speech Process. Hong Kong: IEEE; 2004. p 743–746.

Sudre CH, Li W, Vercauteren T, Ourselin S, Jorge Cardoso M. Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations. Lect. Notes Comput. Sci. 2017:240–248.

Salehi SSM, Erdogmus D, Gholipour A. Tversky loss function for image segmentation using 3D fully convolutional deep networks. Lect. Notes Comput. Sci. 2017:379–387.

Abraham N, Khan NM. A novel focal tversky loss function with improved attention u-net for lesion segmentation. Proc. - Int. Symp. Biomed. Imaging 2019;2019-April:683–687.

Jadon S. A survey of loss functions for semantic segmentation. In: 2020 IEEE Conf. Comput. Intell. Bioinforma. Comput. Biol. CIBCB 2020. Via del Mar, Chile; 2020. p 1–7.

Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition. In: 3rd Int. Conf. Learn. Represent. ICLR 2015 - Conf. Track Proc. 2015. p 1–14.

Tomar N. Semantic-Segmentation-Architecture. github 2021.


