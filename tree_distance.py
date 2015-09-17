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

        def get_parents(node,parents) :
            for parent in node.parents :
                if parent not in parents :
                    get_parents(parent,parents)
            parents.add(node)
            
        synset_list_2012 = open(synsets_file).readlines()
        synset_list_2012 = [wnid.strip() for wnid in synset_list_2012]
        num_class = len(synset_list_2012)

        distance_matrix = np.zeros((num_class,num_class),np.uint32)
        for i in range(num_class) :
            for j in range(i,num_class) :
                p1 = Set(); p2 = Set();
                get_parents(nodes_dict[synset_list_2012[i]],p1)
                get_parents(nodes_dict[synset_list_2012[j]],p2)

                distance_matrix[i,j] = len( p1.symmetric_difference(p2) )
                distance_matrix[j,i] = distance_matrix[i,j]

        self.distance_matrix = distance_matrix

    def dist(self,ims1,ims2) :
        if( len(ims1) != len(ims2) ) :
            print 'length unequal'
            return

        ret = np.zeros((len(ims1),1),np.int32)
        for i in range(len(ims1)) :
            ret[i,0] = self.distance_matrix[ims1[i][0],ims2[i][0]]
        return ret

