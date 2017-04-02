# -*- coding: utf-8 -*-
import os
import codecs
from bs4 import BeautifulSoup

from config_paths import IP_HTML_PATH, OUT_DIR

class DataReaderAndHtmlParser():
    
    def __init__(self):
        self.dataset = {}
        self.all_categories = []
        self.ip_html_dir = IP_HTML_PATH
        self.out_dir = OUT_DIR
        self._generate_dataset()
        
    """
    Input:  Directory where output txt files should be created.
    Processing:  Parse HTML, extract text, write text into files.
    """
    def transform_to_text(self):
        # Step 1: Create directory, if it does not exist
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
            
        # Step 2: Extract text from html & write to external file.
        for item  in self.dataset["filepath_to_category"].items():
            category= item[1]
            html_path= item[0]
            text_data= self._get_html_text_lines(html_path)        
            filename = html_path.split(category)[1].replace(os.pathsep,"")+".txt"
            file_path = self.out_dir+category+"_"+filename[1:]        
            self._write_extracted_text(file_path,text_data)
            print "Created: "+file_path
            
    def get_dataset(self):
        return self.dataset
        
    def get_categories(self):
        return self.all_categories
        
    def print_data_description(self, data):
        print "\nCOUNT IF DOCUMENTS: {0}".format(data["tot_docs"])
        print "\nCOUNT IF DISTINCT CATEGORIES:",data["categories_count"]
        print "\nDISTINCT CATEGORIES:"
        for cat in data["categories"]:
            print cat    
        print "\nDOCUMENT DISTRIBUTION:"
        for item in data["category_to_doc_count"]:
            print item[0], ":",item[1]

    def _get_html_text_lines(self,html_path):
        with open(html_path,'r') as fh:
            html = fh.read()
        soup = BeautifulSoup(html,'lxml')
        # Remove all script and style elements
        for script in soup(["script", "style"]):
            script.extract()       
        data = soup.get_text().lower()
        lines = (line.strip() for line in data.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk).splitlines()
        return text
        
    def _get_subdirs_list(self):
        sub_dirs = [x[0] for x in os.walk(self.ip_html_dir)]
        sub_dirs.remove(self.ip_html_dir)
        return sub_dirs
        
    def _generate_dataset(self):
        tot_docs = 0
        filepath_to_category = {}
        category_to_doc_count = {}
        # Build category-list and filepath-to-category-dict.
        for a_dir in self._get_subdirs_list():
            category = a_dir.split(self.ip_html_dir)[1]
            # Add categories
            if not self.all_categories.__contains__(category):
                self.all_categories.append(category)
                category_to_doc_count[category] = 0
            # Add category for each file
            for (dirpath, dirnames, filenames) in os.walk(a_dir):
                for fname in filenames:
                    tot_docs = tot_docs+1
                    filepath_to_category[a_dir+"/"+fname] = category
                    # Update document count for each category
                    category_to_doc_count[category] = \
                                category_to_doc_count[category] +1
        # Get data description
        self.dataset = {"tot_docs": tot_docs,
                   "categories": self.all_categories,\
                   "categories_count": len(self.all_categories),\
                   "filepath_to_category": filepath_to_category,\
                   "category_to_doc_count": \
                       sorted(category_to_doc_count.items(), key=lambda x: x[1])}
        return self.dataset
    
    def _write_extracted_text(self,filepath,file_lines):
        with codecs.open(filepath,'w','utf-8-sig') as outfile:
            for line in file_lines:
                try:
                    outfile.write(line)
                except:
                    print "Failed to write:"+line
               
if __name__=="__main__":
    dataParser = DataReaderAndHtmlParser()    
    dataParser.transform_to_text()        