import os            #for OS related activities fetching directory, searching ,marking existence of directory
from string import punctuation  #importing build in punctuation's
from nltk.tokenize import word_tokenize  #importing build in tokenizer
import contractions  #for removing contractions for I'll -> I will
from nltk.stem import PorterStemmer  #importing porterstemmer algorithm 
import json  #json for saving and loading index's
from math import log10
import numpy as np

ps=PorterStemmer()
DATASET_DIR=os.path.join(os.getcwd(),'dataset/ShortStories/')   #for dataset directory
TOTAL_DOCS=50 #number of docs
DISK_READ=False #flag to check if index were fetched from disk for not

class VectorSpaceModel:

    def __init__(self):
        global punctuation
        global DISK_READ
        punctuation+='“”’‘—'  #inorder to deal with punctuations of different unicode
        self.tf_df_index=dict()  #for inverted index
        self.tf_idf_index=dict()
        self.document_tf_idf_index=dict()
        self.stop_word=["a", "is", "the", "of", "all", "and", "to", "can", "be", "as", "once"
                        , "for", "at", "am", "are", "has", "have", "had", "up", "his", "her", "in", "on", "no", "we", "do"]
        
        if os.path.exists((os.path.join(os.getcwd(),'vsm_index.json'))):
            DISK_READ=True
        
        if DISK_READ:
            with open('vsm_index.json','r') as json_file:
                self.tf_idf_index=json.load(json_file)


    def pre_process(self,document):
        document=document.lower()  #lowers the text
        document=contractions.fix(document)  #remove contractions 
        document=document.translate(str.maketrans('','',punctuation))  #remove punctuations from text
        tokenize_word_list=word_tokenize(document) # make tokenizers 
        tokenize_word_list=[ word for word in tokenize_word_list if word not in self.stop_word ] #remove stop words
        tokenize_word_list=[ ps.stem(word) for word in tokenize_word_list ] #apply stemming 
        return tokenize_word_list

    def process_txt(self):
        global TOTAL_DOCS
        for txt_file in os.listdir(DATASET_DIR):   #going thorough all txt files in the folder and then calling pre_process function 
            if txt_file.endswith('.txt'):          #for pre processing and then creating index's 
                doc_id=int(txt_file.split('.')[0])
                f=open(os.path.join(DATASET_DIR,txt_file),'r',encoding='utf-8')
                word_list=self.pre_process(f.read())
                self.create_tf_df_index(word_list,str(doc_id))
        TOTAL_DOCS=len(os.listdir(DATASET_DIR))    
        self.create_tf_idf_index()
        self.write_file()

    def create_tf_df_index(self,words,DOC_ID):
        for word in words:
            if word not in self.tf_df_index:
                self.tf_df_index[word]=dict()

            if 'tf' not in self.tf_df_index[word]:
                self.tf_df_index[word]['tf']=dict()

            if DOC_ID not in self.tf_df_index[word]['tf']:
                self.tf_df_index[word]['tf'][DOC_ID]=1
            else:
                self.tf_df_index[word]['tf'][DOC_ID]+=1

    def create_tf_idf_index(self):
        for word in self.tf_df_index.keys():
            word_tfs=self.tf_df_index[word]['tf']
            df=len(word_tfs)
            idf=round(log10(TOTAL_DOCS/df),3)
            self.tf_idf_index[word]={'idf':idf,'df':df,'tf_idf':dict()}
            for doc_id in self.tf_df_index[word]['tf'].keys():
                tf=int(self.tf_df_index[word]['tf'][doc_id])
                tf_idf=round(tf*idf,3)
                self.tf_idf_index[word]['tf_idf'][doc_id]=tf_idf

    def process_document_tf_idf(self):

        for DOC_ID in range(1,TOTAL_DOCS+1):
             self.document_tf_idf_index[str(DOC_ID)]=[0]*len(self.tf_idf_index.keys())
        index=0
        for word in self.tf_idf_index.keys():
            tf_idf_dict=self.tf_idf_index[word]['tf_idf']
            for doc_id in tf_idf_dict.keys():
                self.document_tf_idf_index[doc_id][index]=float(self.tf_idf_index[word]['tf_idf'][doc_id])
            index+=1

    def process_query_vector(self,query):
        query=query.lower().split()
        query_tf_dict=dict()
        for word in query:
            word=ps.stem(word)
            if word not in query_tf_dict:
                query_tf_dict[word]=1
            else:
                query_tf_dict[word]+=1

        query_vector=list()
        for word in self.tf_idf_index.keys():
            idf=self.tf_idf_index[word]['idf']

            if word in query_tf_dict:
                tf_idf=round(query_tf_dict[word]*idf,3)
                query_vector.append(tf_idf)
            else:
                query_vector.append(0.0)
        
        return query_vector

    def compute_result(self,query_vector,alpha):
        query_vector=np.array(query_vector)
        result_set=list()
        for DOC_ID in self.document_tf_idf_index.keys():
            tf_idf_document_list=np.array(self.document_tf_idf_index[DOC_ID])
            dot_product=query_vector.dot(tf_idf_document_list)
            mag1=np.linalg.norm(query_vector)
            mag2=np.linalg.norm(tf_idf_document_list)
            cosine_similarity=round((dot_product/(mag1*mag2)),4)
            if cosine_similarity>=alpha:
                result_set.append((DOC_ID,cosine_similarity))
        
        result_set.sort(key=lambda x: x[1],reverse=True)
        return result_set

    def write_file(self):
        vsm_index_json=json.dumps(self.tf_idf_index)  #writes both file in json format
        with open('vsm_index.json','w') as json_file:
            json_file.write(vsm_index_json)
        
model=None
if __name__=='__main__':
    model=VectorSpaceModel()
    if not DISK_READ:
        print('Index Not Found Creating It...')
        model.process_txt()
        print('Index Created Successfully...')
    model.process_document_tf_idf()
    print("We are ready let's begin...")
    alpha=float(input('Enter value of alpha or enter -1 in order to set it to default(0.005): '))
    if alpha==-1:
        alpha=0.005
    while True:
        x=input('Enter Query: ')
        if x=='-1':
            break
        query_vector=model.process_query_vector(x)
        print(model.compute_result(query_vector,alpha))
else:
    model=VectorSpaceModel()
    if not DISK_READ:
        print('Index Not Found Creating It...')
        model.process_txt()
        print('Index Created Successfully...')
    model.process_document_tf_idf()
    print("We are ready let's begin...")
    