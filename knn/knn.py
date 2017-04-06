# -*- coding: utf-8 -*-
import math
from CV_set_generator import kFoldDatasetGenerator

class KNN():
     def __init__(self, knn_k, persist_results):
         self.knn_k = knn_k 
         self.evaluation = {} 
         
         self._dist_measure_types = ["euclid","cosine"]
         self.dist_type = self._dist_measure_types[0]
        
         """
         Persist results across multiplt run_classifier calls. Used in case
         of k-fold validation to aggregate results across folds
         """
         self.persist_results = persist_results
         

     def run_classifier(self,dataset, train_idxs, test_idxs):
        """
        Classifies the data in dataset at test_idxs using,
        data at train_idxs as training dataset. Prints the 
        evaluation(F-measure, accuracy etc.) 
        """
        self.dataset = dataset
        self.test_idxs = test_idxs
        self.train_idxs = train_idxs
        self.evaluation = self.evaluation if  self.persist_results else {}
        for a_test_data_idx in self.test_idxs:
            test_data_str = self.dataset[a_test_data_idx]
            nn = self._get_nearest_neighbours(test_data_str)
            to_discard, actual_class = self._get_vector_category_pair(test_data_str)            
            predicted_class = self._get_decison(nn)
            if self.evaluation.__contains__(actual_class):
                if self.evaluation[actual_class].__contains__(predicted_class):
                    self.evaluation[actual_class][predicted_class] = \
                            self.evaluation[actual_class][predicted_class]+1
                else:
                    self.evaluation[actual_class][predicted_class] = 1
            else:
                    self.evaluation[actual_class] = {}
                    self.evaluation[actual_class][predicted_class] = 1

            #print actual_class+"===>"+predicted_class
        
     def get_evaluation(self):
         return self.evaluation
            
     def _get_decison(self, nn_tuple):
         # Majority voting
         #print nn_tuple
         cat_to_vote = {}
         for dist_cat in nn_tuple:
             cat = dist_cat[1]
             #dist = dist_cat[0]
             if not cat_to_vote.__contains__(cat):
                 cat_to_vote[cat] = 1
             else:
                 cat_to_vote[cat] = cat_to_vote[cat] + 1
         sorted_tuple_list = sorted(cat_to_vote.items(),key=lambda x:x[1], reverse=True)
         return sorted_tuple_list[0][0]

     def _get_distance(self,vector_a_str, vector_b_str):
        dist = 0
        vector_a , cat_a = self._get_vector_category_pair(vector_a_str)
        vector_b , cat_b = self._get_vector_category_pair(vector_b_str)
        if self.dist_type == "euclid":
            sq_sum = 0
            for idx in range(0,len(vector_a)):
                va = vector_a[idx]
                vb = vector_b[idx]
                #print va , vb
                sq_sum = sq_sum +  float(math.pow( (va-vb),2))
            val = math.sqrt(sq_sum)
            return val 
        elif self.dist_type == "cosine":
            #TODO: Implement it.
            dist =0
        return dist
        
     def _get_nearest_neighbours(self,test_data_str):  
         """ Returns a list of (category, distance) tuples, 
         which are nearest to the test vector"""
         nn = []
         for a_train_data_idx in self.train_idxs:
             
             train_data_str = self.dataset[a_train_data_idx]
             
             if len(train_data_str) < 400:
                 print "Problem at idx:"+str(a_train_data_idx)
             new_dist = self._get_distance(test_data_str,train_data_str)
             to_discard, category = self._get_vector_category_pair(train_data_str)            
             if len(nn) < self.knn_k:
                 nn.append((new_dist,category)) #[(5,classA),(9,classB)...]
             elif self.dist_type == "euclid":
                 #Find the idx in nn which has largest distance
                 largest_dist = -1
                 largest_dist_idx= -1
                 for idx in range(0,len(nn)):
                     if nn[idx][0] > largest_dist:
                         largest_dist = nn[idx][0]
                         largest_dist_idx = idx
                 nn[largest_dist_idx] = (new_dist,category)
             elif self.dist_type == "cosine":
                 #Find the idx in nn which has least similarity
                 least_sim = -100000
                 least_sim_idx= -1
                 for idx in range(0,len(nn)):
                     if nn[idx][0] < least_sim:
                         least_sim = nn[idx][0]
                         least_sim_idx = idx
                 nn[least_sim_idx] = (new_dist,category)
         return nn
     
     """
     NOTE: It is assumed that the last columns has the category
     """
     def _get_vector_category_pair(self,data_row): 
        data = data_row.split(",")
        category = data[-1:]
        vector_str_list = data[:-1]
        vector = [float(score) for score in vector_str_list]
        return vector, category [0]        

def get_dataset_As_list(data_set_path):
    """ Reads a file, splits at line boundaries,
        and returns the lines as list."""
    with open(data_set_path,'r') as fh:
        data= fh.read()
    return data.split("\n")
    
def evaluate_results(evaluation):
    categories = [ item[0] for item in evaluation.items()]
    
    tp = {}
    fp = {}
    fn = {}
    tot_instances = {}
    f_score = {}
    for a_cat in categories:
        tp[a_cat] = 0
        fp[a_cat] = 0
        fn[a_cat] = 0

    for a_cat in categories:
        results = evaluation[a_cat]
        tot_instances[a_cat] = 0
        for res in results.items():
            cat = res[0]
            cnt = res[1]
            tot_instances[a_cat] = tot_instances[a_cat] + cnt
            if cat == a_cat:
                tp[cat] = cnt
            else:
                fn[a_cat] = fn[a_cat]+cnt
                fp[cat] = fn[cat]+cnt
    print "\nTP:"+str(tp)
    print "FP:"+str(fp)
    print "FN:"+str(fn)
    for cat in categories:
        try:
            precision = (tp[cat])/float(tp[cat]+fp[cat])
            recall = (tp[cat])/float(tp[cat]+fn[cat])
            f_scored = round(2 * precision * recall / float(precision + recall),3)
        except:
            f_scored =0
        f_score[cat] = f_scored
        print cat+"\tPrecision:"+str(precision)+"\tRecall:"+str(recall)+"\tf_score:"+str(f_scored)
        
    wt_avg_fscore = 0
    tot = 0
    for inst in tot_instances.items():
        cat = inst[0]
        cnt = inst[1]
        wt_avg_fscore = wt_avg_fscore+float(f_score[cat]*cnt)
        tot = tot+cnt
    wt_avg_fscore = round(wt_avg_fscore/float(tot),3)
    print "Total Instances:"+str(tot)+" Weighted F-Measure: "+str(wt_avg_fscore)
              
if __name__=="__main__":
    
    has_header = True
    data_set_path = "./../../data/dal_knn_dataset_500.csv"
    
    dataset_as_list = get_dataset_As_list(data_set_path)
    dataset = dataset_as_list[1:] if has_header else dataset_as_list
    
    k_fold = 10
    kFolddatasetGenerator = kFoldDatasetGenerator(k_fold)
    test_train_tup_list =  kFolddatasetGenerator.get_k_fold_dataset(len(dataset)-1)
    
    knn_k = 15
    knn_classifier = KNN(knn_k,True)
    for test_train_tup in test_train_tup_list:
        test_idx_list = test_train_tup[0]
        train_idx_list = test_train_tup[1]
        knn_classifier.run_classifier(dataset,train_idx_list,test_idx_list)
        evaluation = knn_classifier.get_evaluation()
        for item in evaluation.items():
            print item[0],item[1]
        print "Completed a fold...\n\n\n\n"
    evaluate_results(evaluation)        
        