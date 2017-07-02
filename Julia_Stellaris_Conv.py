
# coding: utf-8

# In[26]:

#0. Load train images
#1. Genetic Algorithm work
#2. Make convolutional network for images
#3. Evaluation of the generated images and sort
#4. Make Stellaris file from all images
#5. ...
#6. PROFIT


# In[39]:

# import libs
# not worked without it
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# python helpers
from __future__ import division, print_function, absolute_import
import pickle
import glob
import numpy as np
import random
from PIL import Image
from time import time
# genetic lib
from pygene3.gene import IntGeneRandom, DiscreteGene
from pygene3.organism import Organism
from pygene3.population import Population
# nero lib
import tensorflow as tf
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

# dirs
desktop = os.path.join(os.environ['USERPROFILE'], 'Desktop')
work_dir = os.path.join(desktop, 'Stellaris_Julia_Nero_test')

good_pics_dir = os.path.join(work_dir, 'good_pics')
bad_pics_dir = os.path.join(work_dir, 'bad_pics')
good_pics_sorted_dir = os.path.join(work_dir, 'good_pics_sorted')
bad_pics_sorted_dir = os.path.join(work_dir, 'bad_pics_sorted')
unsorted_pics_dir = os.path.join(work_dir, 'unsorted_pics')
stellaris_maps_dir = os.path.join(work_dir, 'map\\setup_scenarios')


# In[45]:

#0. Load train images

good_pics = np.array(glob.glob(good_pics_dir + '\\*.jpg'))
x = np.array([np.array(Image.open(fname)) for fname in good_pics])
x = x[:,:,:,[3]]
y = np.ones(x.shape[0])

bad_pics = np.array(glob.glob(bad_pics_dir + '\\*.jpg'))
xtest = np.array([np.array(Image.open(fname)) for fname in bad_pics])

xtest = xtest[:,:,:,[3]]
ytest = np.zeros(xtest.shape[0])

xtest, ytest = shuffle(xtest, ytest)

# take part of bad for train
x1 = xtest[0:x.shape[0]]
y1 = ytest[0:x.shape[0]]

y = np.concatenate((y, y1))
x = np.concatenate((x, x1))
ytest = np.concatenate((y, ytest))
xtest = np.concatenate((x, xtest))

xtest, ytest = shuffle(xtest, ytest)
x, y = shuffle(x, y)

y = to_categorical(y, 2)
ytest = to_categorical(ytest, 2)

# save by pikle for fater load in future
with open(os.path.join(work_dir, 'julia.pickle'), 'wb') as handle:
    pickle.dump([x, y, xtest, ytest], handle, protocol=pickle.HIGHEST_PROTOCOL)
print('done')


# In[3]:

#1. Genetic Algorithm work
# GA classes
class ConvGene(IntGeneRandom):
    mutProb = 0.3
    mutAmt = 1
    randMin = 1
    randMax = 5

    def __repr__(self):
        return str(self.value)
    
class FilterGene(IntGeneRandom):
    mutProb = 0.3
    mutAmt = 1
    randMin = 2
    randMax = 16

    def __repr__(self):
        return str(self.value)
    
class NeuronGene(IntGeneRandom):
    mutProb = 0.3
    mutAmt = 20
    randMin = 300
    randMax = 500

    def __repr__(self):
        return str(self.value)

class Julia_Network:
    def __init__(self, conv1, conv2, conv3, filter1, filter2, filter3, neronum):
        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3
        self.filter1 = filter1
        self.filter2 = filter2
        self.filter3 = filter3
        self.neronum = neronum
        
        network = input_data(shape=[None, 100, 100, 1])
        if self.conv1 > 1:
            network = conv_2d(network, self.filter1, self.conv1, activation='relu')
            network = max_pool_2d(network, 2)
        if self.conv2 > 1:
            network = conv_2d(network, self.filter2, self.conv2, activation='relu')
            network = max_pool_2d(network, 2)
        if self.conv3 > 1:
            network = conv_2d(network, self.filter3, self.conv3, activation='relu')
            network = max_pool_2d(network, 2)
        network = fully_connected(network, self.neronum, activation='relu')
        network = dropout(network, 0.8)
        network = fully_connected(network, y.shape[1], activation='softmax')
        network = regression(network, optimizer='adam',
                             loss='categorical_crossentropy',
                             learning_rate=0.001,
                            )
        self.model = tflearn.DNN(network, tensorboard_verbose=0,
                                #checkpoint_path='julia_classifier.tfl.ckpt',
                                max_checkpoints = 0,
                               )

class Julia_Solver(Organism):
    
    genome = {
        'conv1' : ConvGene,
        'conv2' : ConvGene,
        'conv3' : ConvGene,
        'filter1' : FilterGene,
        'filter2' : FilterGene,
        'filter3' : FilterGene,
        'neronum' : NeuronGene,
    }
    
    def fitness(self):
        # not worked without it
        tf.reset_default_graph()
        currentnetwork = Julia_Network(self['conv1'], self['conv2'], self['conv3'], self['filter1'], self['filter2'], self['filter3'], self['neronum'])
        currentnetwork.model.fit(x, y, n_epoch=30,#self['n_epoch'],
                                shuffle=True,
                                show_metric=False,
                                snapshot_epoch=False,
                                #batch_size=98,
                                #validation_set=0.5
                            )
        
        result = currentnetwork.model.evaluate(xtest, ytest)
        #del currentnetwork
        return (-1*result[0])

    def __repr__(self):
        return '<fitness=%f conv1=%s conv2=%s conv3=%s filter1=%s filter2=%s filter3=%s neronum=%s>' % (
            self.fitness(), self['conv1'], self['conv2'], self['conv3'], self['filter1'], self['filter2'], self['filter3'], self['neronum'])
    
class Julia_Population(Population):
    species = Julia_Solver
    initPopulation = 30
    # max pops
    childCull = 15
    # reproduse pops
    childCount = 30
    mutants = 1#0.8
    mutateAfterMating = True
    numNewOrganisms = 0
    # we need best net
    incest = 1
    
# GA work
with open(os.path.join(work_dir, 'julia.pickle'), 'rb') as handle:
    x, y, xtest, ytest = pickle.load(handle)
    
# create population
ph = Julia_Population()

timer = time()

print('gogo')
# 20 generations
i = 0
while True:
    b = ph.best()
    # statistics
    current_time = (time() - timer)/60
    stat = 'generation %02d: %s average=%s time=%.1f min)' % (
        i, repr(b), ph.fitness(), current_time)
    
    print(stat)
    # stats to file
    with open(os.path.join(work_dir, 'log.txt'), 'a') as handle:
        handle.write(stat)
        handle.write('\n')

    if b.get_fitness() < -0.99:
        break
        
    if i > 20:
        break

    #sys.stdout.flush()
    i += 1
    ph.gen()
print(repr(b))
# sleep mod
os.system(r'rundll32.exe powrprof.dll,SetSuspendState Hibernate')


# In[46]:

#2. Make convolutional network for images
with open(os.path.join(work_dir, 'julia.pickle'), 'rb') as handle:
    x, y, xtest, ytest = pickle.load(handle)

timer = time()
# not worked without this
tf.reset_default_graph()

# network 
network = input_data(shape=[None, 100, 100, 1])
network = conv_2d(network, 3, 11, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 3, 10, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 2, 11, activation='relu')
network = max_pool_2d(network, 2)

network = fully_connected(network, 309, activation='relu')
network = dropout(network, 0.8)

network = fully_connected(network, y.shape[1], activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001,
                    )

model = tflearn.DNN(network, tensorboard_verbose=0,
                    checkpoint_path='julia_classifier.tfl.ckpt',
                    #best_checkpoint_path='superclassifier_best.tfl.ckpt',   
                    max_checkpoints = 0,
                   )

model.fit(x, y, n_epoch=30, shuffle=True,
          show_metric=True,
          #batch_size=30,
          snapshot_epoch=False,
          #snapshot_step = 30,
          #validation_set=0.3,
          run_id='julia_classifier'
         )
model.save(os.path.join(work_dir, 'julia_classifier.tfl'))
# can load model if needed
#model.load(os.path.join(work_dir, 'julia_classifier.tfl'))
print('Network trained and saved as julia_classifier.tfl')
print(model.evaluate(xtest, ytest))
print((time() - timer)/60)


# In[48]:

#3. Evaluation of the generated images and sort
unsorted_pics = np.array(glob.glob(unsorted_pics_dir + '\\*.jpg'))
xsort = np.array([np.array(Image.open(fname)) for fname in unsorted_pics])
xsort = xsort[:,:,:,[3]]

for i, pic in enumerate(xsort):
    result = model.predict_label(xsort[i:i+1])
    name = str(i) + '.jpg'
    #name = os.path.split(unsorted_pics[i])[1]
    if result[0][0] == 1:
        # move to good
        os.rename(unsorted_pics[i], os.path.join(good_pics_sorted_dir, name))
    else:
        os.rename(unsorted_pics[i], os.path.join(bad_pics_sorted_dir, name))
print('done')


# In[ ]:

#4. Make Stellaris file from all images
def gen_galaxy(current_pic):
    pic = np.array(Image.open(current_pic))
    s = 0
    i = 0
    coordinates = list()
    while i < 100:
        j = 0
        while j < 100:
            if pic[i][j][3] == 255:
                s += 1
                coordinates.append((i, j))
                #print(i,j)
            j += 1
        i += 1
    coordinates = shuffle(coordinates)
    fname = os.path.splitext(os.path.basename(current_pic))[0]
    #print(fname)
    with open(os.path.join(stellaris_maps_dir, fname + '.txt'), 'w') as handle:
        handle.write('#stars ' + str(s))
        handle.write('\nstatic_galaxy_scenario = {\n\tname = \"' + fname + '\"\n\tpriority = 0\n\tdefault = no\n\tcolonizable_planet_odds = 1.0\n\tnum_empires = { min = 0 max = 60 }\n\tnum_empire_default = 21\n\tfallen_empire_default = 4\n\tfallen_empire_max = 4\n\tadvanced_empire_default = 7\n\tcore_radius = 0\n\trandom_hyperlanes = yes\n\n')
        for i, coordinate in enumerate(coordinates[0]):
            y = (10 * (coordinate[0] - 50) + random.randint(-3, 4))
            x = (10 * (coordinate[1] - 50) + random.randint(-4, 5))
            # move away from border
            if x > 500:
                x -= 8
            elif x < -500:
                x += 8
            if y > 500:
                y -= 8
            elif y < -500:
                y += 8;
            # write to file
            handle.write('\tsystem = {\n\t\tid = ' + str(i) + '\n\t\tposition = {\n\t\t\tx = ' + str(x) + '\n\t\t\ty = ' + str(y) + '\n\t\t}\n\t}\r')
            # place some nebulas
            if random.randint(0, 250) == 1:
                handle.write('\tnebula = {\n\t\tposition = {\n\t\t\tx = ' + str(x) + '\n\t\t\ty = ' + str(y) + '\n\t\t}\n\t\tradius = ' + str(random.randint(40, 100)) + '\n\t}\r')
        handle.write('}')
    return

# make files for stellaris
good_pics = np.array(glob.glob(good_pics_dir + '\\*.jpg'))
for good_pic in good_pics:
    gen_galaxy(good_pic)
print('done')


# In[47]:

#5. ...


# In[22]:

#6. PROFIT


# In[ ]:




# In[ ]:




# In[44]:

# rename images
good_pics = np.array(glob.glob(good_pics_dir + '\\*.jpg'))
for i, pic in enumerate(good_pics):
    #Insane Julia Set Nero Classification
    name = 'IJSNC ' + str(i) + '.jpg'
    os.rename(good_pics[i], os.path.join(good_pics_dir, name))
bad_pics = np.array(glob.glob(bad_pics_dir + '\\*.jpg'))
for i, pic in enumerate(bad_pics):
    #Insane Julia Set Nero Classification
    name = 'IJSNC_bad ' + str(i) + '.jpg'
    os.rename(bad_pics[i], os.path.join(bad_pics_dir, name))
print('done')


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



