import csv
import os
import pickle
import re
from datetime import datetime

import numpy as np
import tensorflow
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


class HOF_Classifier:
    def __init__(self):
        self.classifier=[]
        self.vectorizer=[]
        self._class_vectorizer=None
        self._classes=['OT','anatomical']


    def read_scans_csv(self, file):
        with open(file,'r') as inf:
            reader = csv.DictReader(inf)
            scans=[{k: str(v) for k,v in row.items()} 
                      for row in csv.DictReader(inf,skipinitialspace=True)]
        return scans    
       
    '''
    Create vocabulary from the bag of words. These will act as features.
    '''
    def gen_vocabulary(self,scans):
        descs=self.prepare_descs(scans)
        vectorizer=CountVectorizer(min_df=0)
        vectorizer.fit(descs)
        self.vectorizer=vectorizer
        print('the length of vocabulary is ',len(vectorizer.vocabulary_))
     

    #for a NN output, categorical labels are stored as BOW over vocabulary of class labels.
    def prepare_training_vectors_nn(self,scans,gen_hofids=True):
        if self._class_vectorizer is None:
            vectorizer=CountVectorizer(min_df=0)
            vectorizer.fit(self._classes)
            self._class_vectorizer=vectorizer
        vectorizer=self._class_vectorizer
        vectorized_descs=self.gen_bow_vectors(scans)
        hofids=[ s['hof_id'] for s in scans ] if gen_hofids else []        
        return vectorized_descs,vectorizer.transform(hofids).toarray()
    
    def prepare_descs(self,scans):
        #descs are 'sentences' that contain series description and log-compressed number of frames.
        descs=[]
        for s in scans:
            series_desc = s['series_description']
            # handling series descriptions like 'Tumor_Metastasis_BJC/TRA_T', 'Tumor_Post_Op_BJC/T2_TRA_T', 'FL02/AXIAL T2 SE' etc.
            if '/' in series_desc: series_desc = series_desc.split('/')[1]
            
            desc=(re.sub('[^0-9a-zA-Z ]+',' ',series_desc)).split()
            #compressed representation of the number of frames.
            try:
                frames='frames{}'.format(str(int(np.around(np.log(1.0+float(s['frames']))*3.0))))
            except:
                frames='frames0'
            desc.append(frames)
            descs.append(' '.join([s for s in desc if ((not s.isdigit()) and (len(s)>1)) ]))
        return descs
        
    def gen_bow_vectors(self,scans):
        if not self.vectorizer: return []
        descs=self.prepare_descs(scans)
        return self.vectorizer.transform(descs).toarray()    
    
    def train_nn(self,X,y,test_split,epochs=10,batch_size=10, random_state=42):
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_split,random_state=random_state)
        input_dim=X_train.shape[1]
        
        model = Sequential()
        model.add(layers.Dense(36,input_dim=input_dim,activation='relu'))
        #model.add(layers.Dense(18,activation='relu'))
        model.add(layers.Dense(len(self._classes),activation='sigmoid'))
        
        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['categorical_accuracy'])
        model.summary()
        self.classifier=model
        
        checkpoint_filepath = "best_weights.hdf5"
        modelchkpt = tensorflow.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, 
                                                                monitor='val_categorical_accuracy', 
                                                                mode='max', 
                                                                save_best_only=True)

        hist=self.classifier.fit(X_train,y_train,
                                 epochs=epochs,verbose=True,
                                 validation_data=(X_test,y_test),
                                 batch_size=batch_size,
                                 callbacks=[modelchkpt])
        
        # The best model weights are loaded into the model.
        self.classifier.load_weights(checkpoint_filepath)
        return hist

    
    def _merge_hofids(self,scans,hofids):
        for s in scans:
            descr=re.sub(' ','',s['series_description'])
            cmd="slist qd "+"\"" + descr + "\""
            try:
                hof_id=os.popen(cmd).read().split()[1]
            except:
                hof_id=""
            #print(hof_id)
            s['hof_id']=hof_id
            # out.value="{}/{}".format(s['series_description'],hof_id)
        
    def save_model_nn(self,rt):
        pickle.dump(self.vectorizer,open(rt+'.vec','wb'))
        self.classifier.save(rt+'.hd5')


if __name__ == "__main__":
    
    hof_cl=HOF_Classifier()
    scans=hof_cl.read_scans_csv('sample.csv')

    #prepare the data vectors.
    hof_cl.gen_vocabulary(scans)

    descs,y=hof_cl.prepare_training_vectors_nn(scans)
    history = hof_cl.train_nn(descs, y, test_split = 0.1, epochs = 2, batch_size = 10, random_state = 42)

    # save the model files (scan_classifier_nn.*.{hd5,vec}) under root/trained_models/ and change the model reference inside configs/config_{docker,xnat}.py (config['trained_models']['classifier1'])
    hof_cl.save_model_nn(f"scan_classifier_nn.{datetime.today().strftime('%m.%d.%Y')}")