# Last revised on 12 March 2021

from __future__ import print_function
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans

pectoral = [255,0,0] #red
fibrogland = [255,255,0] #yellow
adipose = [0,255,0] # green
background = [0,0,255] # blue

COLOR_DICT = np.array([ pectoral, fibrogland, adipose, background ])


def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        #img = np.concatenate((img,)*3, axis=-1) #convert grayscale image to 3-channel
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        mask = np.array(mask)
        new_mask = np.zeros(mask.shape + (num_class,))
        ########################################################################
        #You should define the value of your labelled gray imgs
        #For example,the imgs in /data/catndog/train/label/cat is labelled white
        #you got to define new_mask[mask == 255, 0] = 1
        #it equals to the one-hot array [1,0,0].
        ########################################################################
        new_mask[mask == 64.,   3] = 1
        new_mask[mask == 128.,   2] = 1
        new_mask[mask == 192.,   1] = 1
        new_mask[mask == 255.,   0] = 1
        mask = new_mask

    elif(np.max(img) > 1):
        img = np.concatenate((img,)*3, axis=-1) #convert grayscale image to 3-channel
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)



def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = True,num_class = 4,save_to_dir = None,target_size = (960,480),seed = 42):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)

def testGenerator(test_path,num_image = 64,target_size = (960,480,1),flag_multi_class = True,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        #img = np.concatenate((img,)*3, axis=-1) #convert grayscale image to 3-channel
        img = img / 255
        img = trans.resize(img,target_size)
        #img = np.reshape(img,img.shape+(1,)) if (flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img


def labelVisualize(num_class,  color_dict, img):
    img_out = np.zeros(img[:,:,0].shape + (3,))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            index_of_class = np.argmax(img[i,j])
            img_out[i,j] = color_dict[index_of_class]
    return img_out



def saveResult(save_path,npyfile,vid,flag_multi_class = True,num_class = 4):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        #img = (img-img.min())*(1/(img.max()-img.min()))
        #img = img*255
        img = img.astype(np.uint8)
        io.imsave(os.path.join(save_path,"%d_predict_%s.png"%(i,vid)),img)
        
        
