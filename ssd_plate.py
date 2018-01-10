
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')

dataset_path = "/home/kun/dataset/my_plate_dataset/daylight"
recog_log_file = './recog_log.txt'
ssd_plate_recog_path = '/home/kun/ssd-plate_detection'
# Make sure that caffe is on the python path:
caffe_root = '/home/kun/caffe-weiliu89-ssd'


plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, './python')
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

os.chdir(ssd_plate_recog_path)

# In[2]:


from google.protobuf import text_format
from caffe.proto import caffe_pb2

# load plate  labels
labelmap_file = '/home/kun/ssd-plate_detection/labelmap_plate.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames


# In[3]:


model_def = '/home/kun/ssd-plate_detection/deploy.prototxt'
model_weights = '/home/kun/ssd-plate_detection/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB


# In[4]:


# set net to batch size of 1
image_resize = 300
net.blobs['data'].reshape(1,3,image_resize,image_resize)
# Load a image from the original repo.
# image = caffe.io.load_image('/home/kun/ssd-plate_detection/1.jpg')

def get_file_paths_with_extension(path,ext):
    files = os.listdir(path)
    file_path_list = []
    for f in files:
        if os.path.isfile(path+'/'+f) and f.endswith(ext):
            file_path_list.append(path+'/'+f)
    return file_path_list

dataset_path = "/home/kun/dataset/my_plate_dataset/daylight"
files = get_file_paths_with_extension(dataset_path,".jpg")
recog_log = open(recog_log_file,'w')
for img_no,img_file in enumerate(files):
    print('===== Image No %d: %s =====' %  (img_no, img_file))
    recog_log.write('===== Image No %d: %s =====\n' %  (img_no, img_file))
    # Load a image from Internet.
    # image = caffe.io.load_image('/home/kun/Downloads/1.jpg')
    # Use my own plate dataset.
    image = caffe.io.load_image(img_file)
    plt.imshow(image)
    
    
    # In[5]:
    
    
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    import time
    # Forward pass.
    start = time.clock()
    detections = net.forward()['detection_out']
    end = time.clock()
    print "read: %f s" % (end - start)
    # Parse the outputs.
    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]
    
    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]
    
    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]
    
    
    # In[6]:
    
    
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    #plt.imshow(image)
    currentAxis = plt.gca()
    print("    Top confidence > 0.6 regions：%d" % top_conf.shape[0])
    recog_log.write("    Top confidence > 0.6 regions：%d\n" % top_conf.shape[0])
    for i in xrange(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * image.shape[1]))
        ymin = int(round(top_ymin[i] * image.shape[0]))
        xmax = int(round(top_xmax[i] * image.shape[1]))
        ymax = int(round(top_ymax[i] * image.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
        label_name = top_labels[i]
        display_txt = '%s: %.2f'%(label_name, score)
        print score
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[label]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=8))
        #currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})     
    plt.savefig('./result_img/' + str(img_no) + '.jpg')
recog_log.close()


