from sets import Set
import itertools
import numpy as np

class Distance :
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

    def __init__(self, is_a_file, synsets_file) :
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
        ret = np.zeros((len(self.synset_list)*2,),np.float32)
        ret[0:len(self.synset_list)] = 1
        
        parents = Set()
        self._get_parents(self.nodes_dict[self.synset_list[leaf_label]],parents)
        
        for node in parents :
            ret[ self.label_to_synset[node.wnid] ] = 0
            ret[ len(self.synset_list)+self.label_to_synset[node.wnid] ] = 1
        return ret
