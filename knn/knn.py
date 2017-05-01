# -*- coding: utf-8 -*-
import math
from CV_set_generator import kFoldDatasetGenerator

class KNN():
     def __init__(self, knn_k, persist_results):
         self.knn_k = knn_k 
         self.evaluation = {} 
         
         self.dist_measure_types = ["euclid","cosine"]
         self.dist_type = self.dist_measure_types[1]
        
         self.decision_function_types = [ "majority_vote",
                                          "sim_wtd_majority",
                                          "density_wtd_majority"]
         self.decision_type = self.decision_function_types[2]

         """
         Persist results across multiplt run_classifier calls. Used in case
         of k-fold validation to aggregate results across folds
         """
         self.persist_results = persist_results
         self.k_filed_density_dict = {}
     
         
     def _build_k_filed_density_dict(self):    
         for a_train_data_idx in self.train_idxs:
             train_data_str = self.dataset[a_train_data_idx]
             train_data_vector, category = self._get_vector_category_pair(train_data_str)            
             nn = self._get_nearest_neighbours(train_data_str,self.dist_type,category)
             #nn = > [(0.46, 'cat1',idx_in_tr_set), (0.49, 'cat1',idx_in_tr_set),..]
             density_sum = 0
             for item in nn:
                 density_sum += item[0]
             density = density_sum / float(self.knn_k)
             self.k_filed_density_dict[a_train_data_idx] = density             

         # Get k filed density for test vectors
         for a_test_data_idx in self.test_idxs:
             test_data_str = self.dataset[a_test_data_idx]
             test_data_vector, category = self._get_vector_category_pair(test_data_str)            
             nn = self._get_nearest_neighbours(test_data_str,self.dist_type,"any")
             density_sum = 0
             for item in nn:
                 density_sum += item[0]
             density = density_sum / float(self.knn_k)
             self.k_filed_density_dict[a_test_data_idx] = density             
         
             
     def get_k_filed_density(self,vector):
         return self.k_filed_density_dict
         
     def run_classifier(self,dataset, train_idxs, test_idxs):
        """
        Classifies the data in dataset at test_idxs using,
        data at train_idxs as training dataset. Prints the 
        evaluation(F-measure, accuracy etc.) 
        """
        self.dataset = dataset
        self.test_idxs = test_idxs
        self.train_idxs = train_idxs
        
        if (self.dist_type == self.dist_measure_types[1]) and \
                (self.decision_type == self.decision_function_types[2]):
            print "Building K-Filed density dictionary.."
            self._build_k_filed_density_dict()
        
        self.evaluation = self.evaluation if  self.persist_results else {}
        for a_test_data_idx in self.test_idxs:
            test_data_str = self.dataset[a_test_data_idx]
            nn = self._get_nearest_neighbours(test_data_str,self.dist_type,"any")
            to_discard, actual_class = self._get_vector_category_pair(test_data_str)            
            predicted_class = self._get_decison(nn,a_test_data_idx)
            self._update_evaluation(actual_class, predicted_class)
     
      
     
     def get_evaluation(self):
         return self.evaluation
            
     def _update_evaluation(self,actual_class, predicted_class):
         if self.evaluation.__contains__(actual_class):
                if self.evaluation[actual_class].__contains__(predicted_class):
                    self.evaluation[actual_class][predicted_class] = \
                            self.evaluation[actual_class][predicted_class]+1
                else:
                    self.evaluation[actual_class][predicted_class] = 1
         else:
                self.evaluation[actual_class] = {}
                self.evaluation[actual_class][predicted_class] = 1   

     def _get_decison(self, nn_tuple,a_test_data_idx):
         """nn tuple => (category,distance)"""
         
         if self.decision_type == self.decision_function_types[0]:
             # Majority voting
             return self._get_majority_voting_descison(nn_tuple)
         elif self.decision_type == self.decision_function_types[1]:
             # Similarity Weighted Majority voting
             return self._get_sim_wtd_majority_voting_descison(nn_tuple)
         elif self.decision_type == self.decision_function_types[2]:
             # Density , similarity weighted Majority voting
             return self._get_sim_and_density_wtd_majority_voting_descison(nn_tuple,a_test_data_idx)
             
     def _get_majority_voting_descison(self,nn_tuple):
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

     def _get_sim_wtd_majority_voting_descison(self,nn_tuple):
         cat_to_vote = {}
         for dist_cat in nn_tuple:
             cat = dist_cat[1]
             sim = dist_cat[0]
             if not cat_to_vote.__contains__(cat):
                 cat_to_vote[cat] = 1*sim
             else:
                 cat_to_vote[cat] = cat_to_vote[cat] + 1*sim
         sorted_tuple_list = sorted(cat_to_vote.items(),key=lambda x:x[1], reverse=True)
         return sorted_tuple_list[0][0]

     def _get_density_based_normalization_factor(self,idx_train,idx_test):
         factor= 1
         factor = math.exp(self.k_filed_density_dict[idx_test]) / \
                        float(math.exp(self.k_filed_density_dict[idx_train]))
         #print factor
         return factor
         
     def _get_sim_and_density_wtd_majority_voting_descison(self,nn_tuple,a_test_data_idx):
         #TODO
         cat_to_vote = {}
         for dist_cat_idx in nn_tuple:
             sim = dist_cat_idx[0]
             cat = dist_cat_idx[1]
             idx_in_train_data = dist_cat_idx[2]
             if not cat_to_vote.__contains__(cat):
                 cat_to_vote[cat] = 1*sim*\
                    self._get_density_based_normalization_factor(idx_in_train_data,a_test_data_idx)
             else:
                 cat_to_vote[cat] += 1*sim*\
                    self._get_density_based_normalization_factor(idx_in_train_data,a_test_data_idx)
         sorted_tuple_list = sorted(cat_to_vote.items(),key=lambda x:x[1], reverse=True)
         return sorted_tuple_list[0][0]

     def _get_distance(self,vector_a_str, vector_b_str):
        dist = 0
        vector_a , cat_a = self._get_vector_category_pair(vector_a_str)
        vector_b , cat_b = self._get_vector_category_pair(vector_b_str)
        if not len(vector_a) == len(vector_b):
            print "\nError:_get_distance:Vector dimensions do not match!\n"
            return -1
        if self.dist_type == self.dist_measure_types[0]:
            sq_sum = 0
            for idx in range(0,len(vector_a)):
                va = vector_a[idx]
                vb = vector_b[idx]
                sq_sum = sq_sum +  float(math.pow( (va-vb),2))
            dist = math.sqrt(sq_sum)
        elif self.dist_type == self.dist_measure_types[1]:
            dist = self._get_cosine_similarity(vector_a,vector_b)
        return dist
     
    
     def _get_cosine_similarity(self,v1,v2):
         sumxx, sumxy, sumyy = 0, 0, 0
         for i in range(len(v1)):
            x = v1[i]; y = v2[i]
            sumxx += x*x
            sumyy += y*y
            sumxy += x*y
         #if not  sumxy:
         #   print "\nERROR: Should not be zero...."
         ret = sumxy/math.sqrt(sumxx*sumyy)
         return ret
    
     def _get_nearest_neighbours(self,test_data_str,dist_type, nn_of_class):  
         """ Returns a list of (category, distance) tuples, 
         which are nearest to the test vector"""
         nn = []
         for a_train_data_idx in self.train_idxs:
             train_data_str = self.dataset[a_train_data_idx]
             new_dist = self._get_distance(test_data_str,train_data_str)
             train_data_vector, category = self._get_vector_category_pair(train_data_str)            
             
             # If Nearest neighbours of same class are requested,
             # continue if the current category is different
             # NN of same class is required for calculating class density.
             if not nn_of_class == "any":
                 if not category == nn_of_class:
                     continue
                 
             if len(nn) < self.knn_k:
                 nn.append((new_dist,category,a_train_data_idx)) #[(5,classA,idx_in_tr_set),(9,classB,idx_in_tr_set)...]
             elif dist_type == self.dist_measure_types[0]:
                 #Find the idx in nn which has largest distance
                 largest_dist = -1
                 largest_dist_idx= -1
                 for idx in range(0,len(nn)):
                     if nn[idx][0] > largest_dist:
                         largest_dist = nn[idx][0]
                         largest_dist_idx = idx
                 nn[largest_dist_idx] = (new_dist,category,a_train_data_idx)
             elif dist_type == self.dist_measure_types[1]:
                 #Find the idx in nn which has least similarity
                 least_sim = nn[0][0]
                 least_sim_idx= 0
                 for idx in range(0,len(nn)):
                     if nn[idx][0] < least_sim:
                         least_sim = nn[idx][0]
                         least_sim_idx = idx
                 nn[least_sim_idx] = (new_dist,category,a_train_data_idx)
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
    
    tp= {};fp= {};fn= {};tot_instances= {};f_score= {}
    for a_cat in categories:
        tp[a_cat]= 0 ;fp[a_cat]= 0;fn[a_cat]= 0

    for a_cat in categories:
        results = evaluation[a_cat]
        tot_instances[a_cat] = 0
        for res in results.items():
            cat = res[0];cnt = res[1]
            tot_instances[a_cat] = tot_instances[a_cat] + cnt
            if cat == a_cat:
                tp[cat] = cnt
            else:
                fn[a_cat] = fn[a_cat]+cnt
                fp[cat] = fn[cat]+cnt
    print "\nTP:"+str(tp)+"\nFP:"+str(fp)+"\nFN:"+str(fn)
    precision= 0; recall= 0
    for cat in categories:
        precision = (tp[cat])/float(tp[cat]+fp[cat])
        recall = (tp[cat])/float(tp[cat]+fn[cat])
        f_scored = round(2 * precision * recall / float(precision + recall),3)        
        f_score[cat] = f_scored
        print cat+"\tPrecision:"+str(round(precision,2))+\
                  "\tRecall:"+str(round(recall,2))+\
                  "\tf_score:"+str(round(f_scored,2))
        
    wt_avg_fscore= 0;tot = 0
    for inst in tot_instances.items():
        cat = inst[0]
        cnt = inst[1]
        wt_avg_fscore = wt_avg_fscore+float(f_score[cat]*cnt)
        tot = tot+cnt
    wt_avg_fscore = round(wt_avg_fscore/float(tot),3)
    print "\n\nTotal Instances:"+str(round(tot,2))+\
          " Weighted F-Measure: "+str(round(wt_avg_fscore,2))
    return wt_avg_fscore
              
if __name__=="__main__":

    data_set_ranges = [400]#,200,300,400,500,600,700,800,900,1000,1250,1500]
    knn_k_ranges = [10]
    final_eval = {}
    for dsr in data_set_ranges:
        final_eval[dsr]={}
        for knr in knn_k_ranges:
            #try:
            has_header = True
            data_set_path = "./../data/dal_knn_dataset_"+str(dsr)+".csv"
            
            dataset_as_list = get_dataset_As_list(data_set_path)
            dataset = dataset_as_list[1:] if has_header else dataset_as_list
            
            k_fold = 10
            kFolddatasetGenerator = kFoldDatasetGenerator(k_fold)
            test_train_tup_list =  kFolddatasetGenerator.get_k_fold_dataset(len(dataset)-1)
            
            knn_k = knr
            knn_classifier = KNN(knn_k,True)
            for test_train_tup in test_train_tup_list:
                test_idx_list = test_train_tup[0]
                train_idx_list = test_train_tup[1]
                knn_classifier.run_classifier(dataset,train_idx_list,test_idx_list)
                evaluation = knn_classifier.get_evaluation()
                for item in evaluation.items():
                    print item[0],item[1]
                print "Completed a fold for dataset "+ str(dsr)+" :knr: "+str(knr)+"...\n"
            wt_fscore = evaluate_results(evaluation)
            final_eval[dsr][knr] = wt_fscore
            #except:
            #    print "....."
    print "\n\nFinal Evaluation:"+str(final_eval)
        