# Preparing Dataset
import os
from os import path

class DataLoader :
    def __init__(self,dataset_loc,train_txt,val_txt):
        self.train_ims = []
        self.val_ims = []
        self.last_idx = 0

        with open(train_txt,"r") as file :
            for line in file:
                loc, label = line.strip().split(' ')
                self.train_ims.append( (int(label),path.join(dataset_loc+'/train_resized/'+loc)) )
        with open(val_txt,"r") as file :
            for line in file:
                loc, label = line.strip().split(' ')
                self.val_ims.append( (int(label),path.join(dataset_loc+'/val_resized/'+loc)) )

        #for cla_idx, cla in enumerate(classes) :
        #    self.train_ims.extend([ (cla_idx,path.join(traindir+'/'+cla,f)) for f in os.listdir(traindir+'/'+cla) if path.isfile(path.join(traindir+'/'+cla,f)) ])

        #self.val_ims = [ (-1,path.join(testdir+'/',f)) for f in os.listdir(testdir) if path.isfile(path.join(testdir+'/',f)) ]

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
