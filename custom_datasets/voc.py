from .lib import Split, resize_image, load_dataset, dataset_size
import tensorflow as tf
from lxml import etree
import os
import tqdm
from PIL import Image

image_size = (300, 300)


# def load_datasets(labels):


# print(label_ids)



# train_data = train_data.map(filter_features).filter(has_labels)
# val_data = val_data.map(filter_features).filter(has_labels)

labels_dict = {"dock":1,"boat":2,"lift":3,"jetski":4,"car":5}

class Voc:
    train = val = test = None

    def __init__(self, labels):
        self.labels = labels

    @staticmethod
    def download_data():
        load_dataset("voc/2007")
        #load_dataset("voc/2012")

    def load_data(self):
        train_data, all_labels = load_dataset("voc/2007", "train+validation")
        label_ids = tf.constant(list(map(lambda label: all_labels.index(label), self.labels)), dtype=tf.int64)
        #train_2012, _ = load_dataset("voc/2012", "train+validation")
        #train_data = train_data.concatenate(train_2012)

        val_data, _ = load_dataset("voc/2007", "test")

        test_data, _ = load_dataset("voc/2007", "test")

        def has_labels(element):
            orig_labels = element['objects']['label']
            a0 = tf.expand_dims(orig_labels, 1)
            b0 = tf.expand_dims(label_ids, 0)
            return tf.reduce_any(a0 == b0)

        train_data = train_data.filter(has_labels)
        val_data = val_data.filter(has_labels)
        test_data = test_data.filter(has_labels)

        train_size = dataset_size(train_data)
        val_size = dataset_size(val_data)
        test_size = dataset_size(test_data)
        label_ids = tf.cast(label_ids, dtype=tf.int32)

        def remove_features(element):
            orig_bboxes = element['objects']['bbox']
            orig_labels = tf.cast(element['objects']['label'], dtype=tf.int32)
            a0 = tf.expand_dims(orig_labels, 1)
            b0 = tf.expand_dims(label_ids, 0)
            indices = tf.where(tf.reduce_any(a0 == b0, 1))
            # tf.print(tf.gather_nd(orig_labels, indices))
            return {
                'objects': {
                    'bbox': tf.gather_nd(orig_bboxes, indices),
                    'label': tf.gather_nd(orig_labels, indices)
                }
            }
        
        train_data = train_data.map(remove_features)
        val_data = val_data.map(remove_features)
        test_data = test_data.map(remove_features)

        self.train = Split(train_data, train_size)
        self.val = Split(val_data, val_size)
        self.test = Split(test_data, test_size)


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

    def load_sample_by_name(path,sample_name,input_shape=image_size,img_extension='.jpg'):
        img_name= os.path.join(path, sample_name + img_extension)

        image = Image.open(img_name)
        iw, ih = image.size
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
                labels.append(label)

                #TODO LABEL MAT DICT
                #classes.append(label_map_dict[obj['name']])
        
        for _ in range(len(bbox),20):
           bbox.append([0,0,0,0])
           labels.append(0)
        
        return img, bbox, labels

    def get_custom_data_generator(path,batch_size=32):
        #Load samples names
        sample_names = Voc.get_list_of_sample_names(path)
        size         = len(sample_names)
        
        return Voc.custom_data_generator(sample_names,path,batch_size),size
    
    def custom_data_generator(sample_names,path,batch_size=32):
        import random

        #Shuffle Data
        size = len(sample_names)
        random.shuffle(sample_names)

        #Create Generator Structure
        idx = 0
        while True:
            #Extract sample data

            '''
            sample,boxes,labels = Voc.load_sample_by_name(path,sample_names[idx])
            idx = (idx+1) % size
            yield sample,boxes,labels
            '''
            samples_dataset = []
            boxes_dataset   = []
            labels_dataset  = []
            #Create a batch of data
            for b in range(batch_size):
                #Extract sample data
                sample,boxes,labels = Voc.load_sample_by_name(path,sample_names[idx])

                samples_dataset.append(sample)
                boxes_dataset  .append(boxes)
                labels_dataset .append(labels)
                idx = (idx+1) % size

                #Shuffle Data On Epoch End
                if idx == 0:
                    random.shuffle(sample_names)
            yield tf.stack(samples_dataset),boxes_dataset,labels_dataset
            
            

    def cache_splits(self, cache_path):
        self.train.cache(cache_path + 'train')
        self.val.cache(cache_path + 'val')
        self.test.cache(cache_path + 'test')

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