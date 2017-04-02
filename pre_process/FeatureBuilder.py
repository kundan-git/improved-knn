# -*- coding: utf-8 -*-
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

import math
from os import listdir
from os.path import isfile, join

from nltk_corpus_stopwords import get_augmented_stopwords
from html_to_text.config_paths import OUT_DIR
from html_to_text.DataReader import DataReaderAndHtmlParser

stop_words = get_augmented_stopwords()

class FeatureBuilder():
     def __init__(self,categories):
         self.max_features = 0
         self.categories = categories
         self.d_val_file_to_word_to_count = {}
         self.tf_file_to_word_to_count = {}
         self.df_word_to_count={}

         # D-Value = (n-m),
         # n: # docs having term, m: # max docs in a category having the term
         self.d_value_word_to_count={} 
             
     def get_fetaures(self,max_features):
        self.max_features = max_features        
        self._getnerate_tf_dict()
        self._generate_df_dict()
        #self._inspect_df(500)
        self._generate_d_val_dict()
        #self._inspect_d_val(500)
        
        # Calculate aggregate TF
        tf_aggregate = {}
        for fname_to_word_to_cnt in self.tf_file_to_word_to_count.items():
            word_to_count = fname_to_word_to_cnt[1]
            for pair in word_to_count.items():
                word = pair[0]
                count = pair[1]
                if tf_aggregate.__contains__(word):
                    tf_aggregate[word] = tf_aggregate[word] +count
                else:
                    tf_aggregate[word] = count

        
        # Calculate TF*IDF*Ci
        word_feature_score = {}
        corpus_size = len(self.tf_file_to_word_to_count.items())
        for word_agtf in tf_aggregate.items():            
            word = word_agtf[0]
            tf = word_agtf[1]
            idf = math.log(corpus_size/float(self.df_word_to_count[word]))
            d_val = 1/float(self.d_value_word_to_count[word]+1)
            score = tf*idf*d_val
            word_feature_score[word] = score
            print word,tf, idf,d_val,score
            
        sorted_score_tuple_list = sorted(word_feature_score.items(),\
                                         key=lambda x:x[1], reverse=True)
        print "\nTOP N Features-----"
        feat_cnt= 0
        fetaures = []
        for word_score in  sorted_score_tuple_list:
            fetaures.append(word_score[0])
            print word_score[0],word_score[1]
            feat_cnt = feat_cnt+1
            if max_features == feat_cnt:
                break
        return fetaures
       
     def _generate_d_val_dict(self):
         for word_doc in self.df_word_to_count.items():
             word = word_doc[0]
             # Initialize
             cat_to_cnt = {}
             for cat in self.categories:
                 cat_to_cnt[cat] = 0
             self.d_val_file_to_word_to_count[word] = cat_to_cnt

             for fname_to_word_to_cnt in self.tf_file_to_word_to_count.items():
                 fname = fname_to_word_to_cnt[0]
                 word_to_count = fname_to_word_to_cnt[1]
                 category = self._get_category_from_file_name(fname)
                 if word_to_count.__contains__(word):
                     self.d_val_file_to_word_to_count[word][category] = \
                        self.d_val_file_to_word_to_count[word][category] +1

             # Get max occurence in category
             max_occurence = 0
             total_occurences = 0
             for item in self.d_val_file_to_word_to_count[word].items():
                 total_occurences = total_occurences+item[1]
                 if item[1] > max_occurence:
                     max_occurence = item[1]
             self.d_value_word_to_count[word]= (total_occurences-max_occurence)

     def _get_category_from_file_name(self, fname):
         for cat in self.categories:
             if fname.__contains__(cat):
                 return cat
             
         
     def _generate_df_dict(self):         
         for file_to_word_to_cnt in self.tf_file_to_word_to_count.items():
             for word_count in file_to_word_to_cnt[1].items():
                 word = word_count[0]
                 if not self.df_word_to_count.__contains__(word):
                    self.df_word_to_count[word] = 0                
                 self.df_word_to_count[word] = self.df_word_to_count[word] +1
    
     def _getnerate_tf_dict(self):
        all_files =  self._get_files_in_dir(OUT_DIR)
        for afile in all_files:
            words_sans_stopwords = []
            with open(afile,'r') as fh:
                words_sans_stopwords = self._remove_stop_words(fh.read())
            word_count_dict = {}
            for word in words_sans_stopwords:
                if not word_count_dict.__contains__(word):   
                    word_count_dict[word] = 1
                else:
                    word_count_dict[word] = word_count_dict[word]+1
            self.tf_file_to_word_to_count[afile] = word_count_dict
            
            
     def _get_files_in_dir(self,path):
        return [path+f for f in listdir(path) if isfile(join(path, f))]
        
     def _remove_stop_words(self,text):
        words = []
        text = text.decode("ascii","ignore").encode("ascii")
        for word in word_tokenize(text):
            word = word.lower()
            if word not in stop_words:
                try:
                    word = porter.stem(word)
                except:
                    pass
                words.append(word)
        return words
    
     def _inspect_df(self, cnt):
        sorted_df_tuple_list = sorted(self.df_word_to_count.items(),\
                                         key=lambda x:x[1], reverse=True)        
        for item in sorted_df_tuple_list:
            print item[0], item[1]
            if cnt ==0:
                break
            cnt = cnt -1
        print len(self.df_word_to_count.items())
        
     def _inspect_d_val(self,idx):
        for it in self.d_val_file_to_word_to_count.items():
            print it[0], it[1]
            print self.d_value_word_to_count[it[0]]
            idx= idx-1
            if idx==0:
                break
            print "\n"
        
if __name__=="__main__":
    
    reader = DataReaderAndHtmlParser()
    categories = reader.get_categories()
    
    featBuilder = FeatureBuilder(categories)        
    features = featBuilder.get_fetaures(100)