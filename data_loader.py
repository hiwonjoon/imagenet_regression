# Preparing Dataset
import os
from os import path

class DataLoader :
    def __init__(self,classes,traindir,testdir):
        self.train_ims = []
        self.last_idx = 0

        for cla_idx, cla in enumerate(classes) :
            self.train_ims.extend([ (cla_idx,path.join(traindir+'/'+cla,f)) for f in os.listdir(traindir+'/'+cla) if path.isfile(path.join(traindir+'/'+cla,f)) ])

        self.val_ims = [ (-1,path.join(testdir+'/',f)) for f in os.listdir(testdir) if path.isfile(path.join(testdir+'/',f)) ]

    def clear_ptr(self) :
        self.last_ptr = 0

    def fetch_val(self, size) :
        last_idx  = min(self.last_idx+size,len(self.val_ims))
        ret = self.val_ims[self.last_idx:last_idx]
        self.last_idx = last_idx
        return ret
    
    def fetch_train(self,size) :
        import random

        ret = [];
        for _ in range(size) :
            ri = random.randint(0,len(self.train_ims)-1)
            ret.append(self.train_ims[ri])
        return ret

if __name__ == "__main__" :
    classes = open('synsets.txt').readlines()
    classes = [wnid.strip() for wnid in classes]

    train_set = DataLoader(classes,'../ilsvrc2012/train_resized')
    print train_set.fetch_serial(64)
