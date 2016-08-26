from multiprocessing import Pool, Process, Queue, Value
import sys, datetime
import os
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import skimage.transform
import skimage.io
from sets import Set
import itertools

class Dataset:
    class tree :
        def __init__(self,wnid) :
            self.wnid = wnid
            self.parents = []
            self.children = []
        def add_parent(self,parent) :
            self.parents.append(parent)
        def add_child(self,child) :
            self.children.append(child)
        def __str__(self):
            return self.wnid

    def __init__(self, base, training_list, test_list, is_a_file, synsets_file):
        # training list is the list of images
        self.base = base

        self.trainings = [ (l.split(' ')[0], int(l.split(' ')[1].strip())) for l in training_list ]

        self.tests = [ (l.split(' ')[0], int(l.split(' ')[1].strip())) for l in test_list ]
        self.last_test = 0

        #For tree 
        nodes_dict = {}
        for line in open(is_a_file).readlines() :
            parent_id, child_id = line.strip().split(' ')
            if parent_id in nodes_dict :
                parent = nodes_dict[parent_id]
            else :
                parent = self.tree(parent_id)
                nodes_dict[parent_id] = parent
            
            if child_id in nodes_dict :
                child = nodes_dict[child_id]
            else :
                child = self.tree(child_id)
                nodes_dict[child_id] = child
            
            child.add_parent(parent)
            parent.add_child(child)

        synset_list = open(synsets_file).readlines()
        synset_list = [wnid.strip() for wnid in synset_list]

        label_to_synset = {}
        for i,wnid in enumerate(synset_list) :
            label_to_synset[wnid] = i

        self.nodes_dict = nodes_dict 
        self.synset_list = synset_list
        self.label_to_synset = label_to_synset

    def _get_parents(self,node,parents) :
        for parent in node.parents :
            if parent not in parents :
                self._get_parents(parent,parents)
        parents.add(node)

    def get_target_prob(self,leaf_label) :
        """
        Input - Leaf Label[int; of synsets.txt]
        Return- Return vectors of one 
        """
        ret = np.zeros((len(self.synset_list),),np.float32)
        
        parents = Set()
        self._get_parents(self.nodes_dict[self.synset_list[leaf_label]],parents)
        
        for node in parents :
            ret[ self.label_to_synset[node.wnid] ] = 1
        return ret

    def fetch_train(self, num) :
        #return numpy array of images from list and 
        #per class label
        ims = []
        labels = []

        for _ in np.arange(num) :
            ri = random.randint(0,len(self.trainings)-1)
            loc, label = self.trainings[ri]
            ims.append( skimage.io.imread( os.path.join(self.base+'/train_resized/'+loc) ) )
            labels.append( self.get_target_prob(label) )
            #labels.append( np.append(self.one_hot_pos(label),self.zero_hot_neg(label)) )
        return (ims,labels)

    def fetch_test_from_beg(self) :
        self.last_test = 0
        return

    def fetch_test_random(self, num) :
        ims = []
        labels = []

        for _ in np.arange(num) :
            ri = random.randint(0,len(self.tests)-1)
            loc, label = self.tests[ri]
            ims.append( skimage.io.imread( os.path.join(self.base+'/val_resized/'+loc) ) )
            labels.append( self.get_target_prob(label) )
            #labels.append( np.append(self.one_hot_pos(label),self.zero_hot_neg(label)) )
        return (ims,labels)

    def fetch_test(self, num, continue_when_end = True) :
        #return numpy array of images from list and 
        #per class label
        ims = []
        labels = []

        idx = self.last_test
        while(num > 0) :
            loc, label = self.tests[idx]
            ims.append( skimage.io.imread( os.path.join(self.base+'/val_resized/'+loc) ) )
            labels.append( self.get_target_prob(label) )
            #labels.append( np.append(self.one_hot_pos(label),self.zero_hot_neg(label)) )

            idx = (idx+1)%len(self.tests)
            if( idx == 0 and not(continue_when_end) ):
                break
            num = num-1
        self.last_test = idx

        return ( (ims,labels), self.last_test == 0 and not(continue_when_end) )

class ImageAugmentation:
    def __init__(self,crop_type,crop_size,scale=None,mean=None) :
        self.crop_size = crop_size
      	self.mean = mean
        self.scale = scale
        if crop_type == 'center' :
            self.crop_f = self._center_crop
        elif crop_type == 'random' :
            self.crop_f = self._random_crop
        else :
            print 'not implented'
            raise 
    def crop(self,im,crop_region) :
        return im[crop_region[0]:crop_region[1],crop_region[2]:crop_region[3],:]

    def _center_crop(self,shape) :
        h, w = shape
        c_h, c_w = (self.crop_size[0],self.crop_size[1])
        return (h//2-c_h//2,h//2+c_h//2,w//2-c_w//2,w//2+c_w//2)
    def _random_crop(self,shape) :
        h, w = shape
        c_h, c_w = (self.crop_size[0],self.crop_size[1])

        margin_h = h - c_h
        margin_w = h - c_w
        if( margin_h < 0 or margin_w < 0) :
            raise
        rand_h = random.randint(0,margin_h)
        rand_w = random.randint(0,margin_w)

        return(rand_h,rand_h+c_h,rand_w,rand_w+c_w)

    def crop_region(self,shape) :
        return self.crop_f(shape)
    
    def augment(self,im,crop_region) :
        #crop
        im = self.crop(im,crop_region)
        # Shuffle Axes to c*w*h
        im = np.swapaxes(np.swapaxes(im,1,2),0,1)
        # convert to BGR order
        im = im[::-1,:,:]
        if(self.scale is not None) :
            im = np.float32(im) * self.scale
        if(self.mean is not None) :
            im = np.float32(im) - self.mean
        return im.astype(np.float32)

class DatasetFetcher :
    def __init__(self, dataset, aug, im_size) :
        self.dataset = dataset 
        self.aug = aug
        self.im_size = im_size
    
    def fetch_random(self,fetch_size) :
        (ims, target) = self.dataset.fetch_test_random(fetch_size)
        crop_regions = [ self.aug.crop_region(shape=(self.im_size,self.im_size)) for _ in np.arange(len(ims)) ]
        ims = [ self.aug.augment(im,cr) for im,cr in zip(ims,crop_regions) ]
    
        return ((ims,target), False)

    def fetch_test(self,fetch_size,from_beg = False, continue_when_end = True) :
        if( from_beg ) :
            self.dataset.fetch_test_from_beg()

        (ims, target), ended = self.dataset.fetch_test(fetch_size, continue_when_end )
        crop_regions = [ self.aug.crop_region(shape=(self.im_size,self.im_size)) for _ in np.arange(len(ims)) ]
        ims = [ self.aug.augment(im,cr) for im,cr in zip(ims,crop_regions) ]

        return ( (ims,target), ended )

class DatasetPreFetcher:
    def __init__(self, dataset, aug, im_size) :
        self.dataset = dataset 
        self.aug = aug
        self.p_list = []
        self.im_size = im_size

    def start(self,max_prefetch,fetch_size) :
        self.q = Queue()

        def fetch(q,dataset) :
            while(1) :
                if( q.qsize() < max_prefetch ) :
                    ims, target = dataset.fetch_train(fetch_size)
                    crop_regions = [ self.aug.crop_region(shape=(self.im_size,self.im_size)) for _ in np.arange(len(ims)) ]
                    ims = [ self.aug.augment(im,cr) for im,cr in zip(ims,crop_regions) ]

                    q.put((ims,target))

        for i in range(4) :
            p = Process(target = fetch,args=(self.q,self.dataset))
            p.start()

            self.p_list.append(p)

        return self.q;

    def end(self):
        for p in self.p_list :
            p.terminate()
            p.join()
        self.p_list = []
        if( self.q ) :
            self.q = None
