# -*- coding: utf-8 -*-
import random

class kFoldDatasetGenerator():
    def __init__(self,k):
        self.k = k
        
    def get_k_fold_dataset(self,dataset_size):
        """ Returns a list having K tuples. In each tuple,
            there are two lists.The first list is test set
            and the second list is training set"""    
        dataset_indexes = []
        #Step 1: [1 2 3 4 5 6 7 8 9]
        indexes= [idx for idx in range(0,dataset_size)]
        #Step 2: Randomize: [9 2 1 4 3 6 8 7 5]
        random.shuffle(indexes)                          
        step = (dataset_size/self.k)
        test_strt_idx= 0
        for cnt in range(0,self.k):
            test_end_idx= test_strt_idx+step
            test_data = []
            train_data = []
            for idx in range(0,dataset_size):
                if idx >= test_strt_idx and idx < test_end_idx:
                    test_data.append(indexes[idx])
                else:
                   train_data.append(indexes[idx]) 
            dataset_indexes.append((test_data,train_data))
            test_strt_idx = test_end_idx
        return dataset_indexes

# Sample Test Function
if __name__=="__main__":
    kFoldGenerator = kFoldDatasetGenerator(10)
    dataset_list =  kFoldGenerator.get_k_fold_dataset(10)
    for item in dataset_list:
        print item[0], item[1]