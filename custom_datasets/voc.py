import tensorflow as tf
from lxml import etree
import os
import tqdm
from PIL import Image
from augmentation import apply as apply_aug

image_size = (300, 300)

'''
labels_dict = {"dock":1,"boat":2,"lift":3,"jetski":4,"car":5}
labels      = ['dock','boat','lift','jetski','car']
'''

labels_dict = {"fish":1,"jellyfish":2,"penguin":3,"shark":4,"puffin":5,"stingray":6,"starfish":7}
labels      = ['fish', 'jellyfish', 'penguin', 'shark', 'puffin', 'stingray','starfish']
class Voc:
    train = val = test = None

    def __init__(self, labels):
        self.labels = labels

    def get_list_of_sample_names(path):
        #Extract Files names:
        import os
        
        dirs = os.listdir(path)

        files = []
        for filename in tqdm.tqdm(dirs, desc='dirs') :
            name = os.path.basename(filename)
            name = filename.split('.')
            name.pop()
            name = '.'.join(name)
            files.append(name)

        samples   = set(files)
        samples   = list(samples)
        return samples

    def load_sample_by_name(path,sample_name,augmentation_func=None,input_shape=image_size,img_extension='.jpg'):
        img_name= os.path.join(path, sample_name + img_extension)

        image = Image.open(img_name)
        height, width = input_shape

        #Resize Image
        img = tf.image.convert_image_dtype(image, tf.float32)
        img = tf.image.resize(img, (height, width))
        
        #Extract Bounding boxes
        bbox    = []
        labels  = []
        
        path = os.path.join(path, sample_name + '.xml')
        with tf.compat.v1.gfile.GFile(path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = recursive_parse_xml_to_dict(xml)['annotation']

        iw = int(data['size']['width'])
        ih = int(data['size']['height'])

        if 'object' in data:
            for obj in data['object']:
                #BOX INFORMATION
                xmin = (float(obj['bndbox']['xmin']) / iw)
                ymin = (float(obj['bndbox']['ymin']) / ih)
                xmax = (float(obj['bndbox']['xmax']) / iw)
                ymax = (float(obj['bndbox']['ymax']) / ih)
                
                
                bbox.append([xmin,ymin,xmax,ymax])
                #Labels INFORMATION
                label = obj['name']
                label = labels_dict[label]
                labels.append(tf.cast(label, tf.int32))

        bbox = tf.clip_by_value(tf.stack(bbox), clip_value_min=0, clip_value_max=1)
        if augmentation_func:
            img, bbox = augmentation_func(tf.constant(img,dtype=tf.float32),bbox)
        return img, bbox, labels

    def pad_custom_data(batch_size,boxes,labels):
        max_len = 0

        for idx in range(batch_size):
            if len(boxes[idx]) > max_len : max_len = len(boxes[idx])
        
        for idx in range(batch_size):
            pad_size  = max_len - len(boxes[idx])
            if pad_size > 0:
                pad_label = tf.fill((pad_size),-1,tf.int32)
                pad_box   = tf.fill((pad_size,4),0.0,tf.float32)
                boxes[idx] = tf.cast(boxes[idx],tf.float32)
                #for _ in range(len(boxes[idx]),max_len):
                try:
                    labels[idx] = tf.concat([labels[idx],pad_label],axis=0)
                    boxes[idx]  = tf.concat([boxes[idx],pad_box],axis=0)
                except Exception as e:
                    print(e)
                    print(pad_box)
                    print(boxes[idx])
            
        #boxes  = tf.stack(boxes)
        #labels = tf.stack(labels)

        return boxes,labels
        
    def get_custom_data_generator(path,batch_size=32,augmentation_func=None):
        #Load samples names
        sample_names = Voc.get_list_of_sample_names(path)
        size         = int(len(sample_names)/batch_size)
        
        return Voc.custom_data_generator(sample_names,path,batch_size,augmentation_func),size
    
    def custom_data_generator(sample_names,path,batch_size=32,augmentation_func=None):
        import random

        #Shuffle Data
        size = len(sample_names)
        random.shuffle(sample_names)

        #Create Generator Structure
        idx = 0
        while True:
            #Extract sample data

            samples_dataset = []
            boxes_dataset   = []
            labels_dataset  = []
            #Create a batch of data
            for b in range(batch_size):
                #Extract sample data
                sample,boxes,labels = Voc.load_sample_by_name(path,sample_names[idx],augmentation_func)

                samples_dataset.append(sample)
                boxes_dataset  .append(boxes)
                labels_dataset .append(labels)
                idx = (idx+1) % size

                #Shuffle Data On Epoch End
                if idx == 0:
                    random.shuffle(sample_names)
            
            boxes_dataset,labels_dataset = Voc.pad_custom_data(batch_size,boxes_dataset,labels_dataset)

            yield tf.stack(samples_dataset),boxes_dataset, labels_dataset

def recursive_parse_xml_to_dict(xml):
  """Recursively parses XML contents to python dict.

  We assume that `object` tags are the only ones that can appear
  multiple times at the same level of a tree.

  Args:
    xml: xml tree obtained by parsing XML file contents using lxml.etree

  Returns:
    Python dictionary holding XML contents.
  """
  if not xml:
    return {xml.tag: xml.text}
  result = {}
  for child in xml:
    child_result = recursive_parse_xml_to_dict(child)
    if child.tag != 'object':
      result[child.tag] = child_result[child.tag]
    else:
      if child.tag not in result:
        result[child.tag] = []
      result[child.tag].append(child_result[child.tag])
  return {xml.tag: result}