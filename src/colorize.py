import tensorflow as tf
from keras import applications
from keras.models import load_model
import os
import numpy as np
import cv2
import glob
import math
from src import config
from shutil import copyfile, rmtree

class DATA():
    def __init__(self, dirname):
        self.dir_path =dirname
        self.filelist = os.listdir(self.dir_path )
        self.batch_size = 1
        self.size = len(self.filelist)
        self.data_index = 0

    def read_img(self, filename):
        IMAGE_SIZE = 224
        MAX_SIDE = 1500
        img = cv2.imread(filename, 3)
        if img is None:
          print('Unable to read image: ' + filename)
          return False, False, False, False, False
        height, width, channels = img.shape
        if height > MAX_SIDE or width > MAX_SIDE:
          print("Image " + filename + " is of size (" + str(height) + "," + str(width) +  ").")
          print("The maximum image size allowed is (" + str(MAX_SIDE) + "," + str(MAX_SIDE) +  ").")
          r = min(MAX_SIDE/height,MAX_SIDE/width)
          height = math.floor(r*height)
          width = math.floor(r*width)
          img = cv2.resize(img,(width,height))
          print("It has been resized to (" + str(height) + "," + str(width) + ")")
        labimg = cv2.cvtColor(cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_BGR2Lab)
        labimg_ori = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        return True, np.reshape(labimg[:,:,0], (IMAGE_SIZE, IMAGE_SIZE, 1)), labimg[:, :, 1:], img, np.reshape(labimg_ori[:,:,0], (height, width, 1))

    def generate_batch(self):
        batch = []
        labels = []
        filelist = []
        labimg_oritList = []
        originalList = []
        for i in range(self.batch_size):
            filename = os.path.join(self.dir_path, self.filelist[self.data_index])
            ok, greyimg, colorimg, original, labimg_ori = self.read_img(filename)
            if ok:
              filelist.append(self.filelist[self.data_index])
              batch.append(greyimg)
              labels.append(colorimg)
              originalList.append(original)
              labimg_oritList.append(labimg_ori)
              self.data_index = (self.data_index + 1) % self.size
        batch = np.asarray(batch)/255 # values between 0 and 1
        labels = np.asarray(labels)/255 # values between 0 and 1
        originalList = np.asarray(originalList)
        labimg_oritList = np.asarray(labimg_oritList)/255
        return batch, labels, filelist, originalList, labimg_oritList


def deprocess(imgs):
    imgs = imgs * 255
    imgs[imgs > 255] = 255
    imgs[imgs < 0] = 0
    return imgs.astype(np.uint8)

def reconstruct(batchX, predictedY):
    result = np.concatenate((batchX, predictedY), axis=2)
    result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)

    return result

def move_data(path):
    latest_edited_file = max([f for f in os.scandir(path)], key=lambda x: x.stat().st_mtime).name
    os.mkdir(path + '/new')
    copyfile(path + '/' + latest_edited_file, path + '/new/' + latest_edited_file)
    copyfile(path + '/new/' + latest_edited_file, config.OUT_DIR + '/' + latest_edited_file)
    return path + '/new/'

def fix_path(path):
    rmtree(path)

def colorize(data_path):
    VGG_modelF = applications.vgg16.VGG16(weights='imagenet', include_top=True)
    save_path = os.path.join(config.MODEL_DIR, config.PRETRAINED_MODEL)
    colorizationModel = load_model(save_path)
    absolute_path = move_data(data_path)
    test_data = DATA(absolute_path)
    assert test_data.size >= 0, 'Your list of images to colorize is empty. Please load images.'

    if not os.path.exists(config.OUT_DIR):
      print('created save result path')
      os.makedirs(config.OUT_DIR)
    batchX, batchY, filelist, original, labimg_oritList = test_data.generate_batch()
    if batchX.any():
      predY, _ = colorizationModel.predict(np.tile(batchX,[1,1,1,3]))
      predictVGG = VGG_modelF.predict(np.tile(batchX,[1,1,1,3]))
      loss = colorizationModel.evaluate(np.tile(batchX,[1,1,1,3]), [batchY, predictVGG], verbose=0)
      originalResult = original[0]
      height, width, channels = originalResult.shape
      predY_2 = deprocess(predY[0])
      predY_2 = cv2.resize(predY_2, (width,height))
      labimg_oritList_2 = labimg_oritList[0]
      predResult_2= reconstruct(deprocess(labimg_oritList_2), predY_2)
      psnr = tf.keras.backend.eval( tf.image.psnr(tf.convert_to_tensor(originalResult, dtype=tf.float32), tf.convert_to_tensor(predResult_2, dtype=tf.float32), max_val=255))
      save_path = os.path.join(config.OUT_DIR, "{:.8f}_".format(psnr) + filelist[0][:-4] + '_reconstructed.jpg')
      cv2.imwrite(save_path, predResult_2)
      fix_path(absolute_path)
      return '{:.8f}_'.format(psnr) + filelist[0][:-4] + '_reconstructed.jpg'
