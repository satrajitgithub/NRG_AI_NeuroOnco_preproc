import warnings
warnings.simplefilter(action='ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

import csv
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import pprint
import pydicom
import re
import sys
import tensorflow as tf
import time
import yaml
from natsort import natsorted, index_natsorted, order_by_index
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential



sys.path.append(os.environ['SCRIPT_ROOT']) # Adds higher directory to python modules path.
from src.classification_aggregation import convert_df_meta_to_bash_parsable_text
from utils.utils import load_config

def get_image_orientation(dicom_slice):
    """
    This function will determine the orientation of MRI from DICOM headers.
    """

    # First check the Acquisition type, if it's 3D or 4D there's no orientation
    try:
        acquistion_type = dicom_slice[0x18, 0x23].value
    except KeyError:
        return 6

    if acquistion_type == '3D':
        return 0
    elif acquistion_type == '4D':
        return 5
    else:
        # Determine orientation from direction cosines
        if (0x20, 0x37) in dicom_slice:
            orientation = dicom_slice[0x20, 0x37].value
        else:
            return 6

        X_vector = np.abs(np.array(orientation[0:3]))
        Y_vector = np.abs(np.array(orientation[3:6]))

        X_index = np.argmax(X_vector)
        Y_index = np.argmax(Y_vector)

        if X_index == 0 and Y_index == 1:
            orientation = 1
        elif X_index == 0 and Y_index == 2:
            orientation = 2
        elif X_index == 1 and Y_index == 2:
            orientation = 3
        else:
            orientation = 4
    return orientation

class HOF_Classifier:
    def __init__(self):
        self.classifier=[]
        self.vectorizer=[]
        self._class_vectorizer=None
        #important: must be in alphabetical order for vectorizer to work correctly.
        # self._classes=['CBF','CBV','DSC','DWI','FA','MD','MPRAGE','MTT','OT','PBP','SWI','T1hi','T1lo','T2FLAIR','T2hi','T2lo','TRACEW','TTP']
        self._classes=['OT','OT','OT','OT','OT','OT','anatomical','OT','OT','OT','OT','anatomical','anatomical','anatomical','anatomical','anatomical','OT','OT']
        #self._scan_list=[]
    def load_json(self, json_file):
        with open(json_file, 'r') as fp:
            out_dict=json.loads(fp.read())
        return out_dict    
    def save_json(self, var, file):
        with open(file,'w') as fp:
            json.dump(var, fp) 
    '''
    Assign HOF ID's to scans using associative table look-up.
    '''
    def assign_hofids_slist(self,scans):
        for s in scans:
            descr=re.sub(' ','',s['series_description'])
            cmd="slist qd "+"\"" + descr + "\""
            try:
                hof_id=os.popen(cmd).read().split()[1]
            except:
                hof_id=""
            #print(hof_id)
            s['hof_id']=hof_id
            #out.value="{}/{}".format(s['series_description'],hof_id)
        return scans

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
     
    #for logreg/svm output, categorical labels are stored as strings
    def prepare_training_vectors(self,scans):
        #labels vector.
        vectorized_descs=self.gen_bow_vectors(scans)
        y=[ s['hof_id'] for s in scans ]
        return vectorized_descs,y
    
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
            # HotFix: Issue #10 : handling series descriptions like 'Tumor_Metastasis_BJC/TRA_T', 'Tumor_Post_Op_BJC/T2_TRA_T' etc.
            if 'BJC/' in series_desc: series_desc = series_desc.split('BJC/')[1]

            # HotFix: Issue #18 : handling series descriptions like 'FL02/AXIAL T2 SE' etc.
            if 'FL0' in series_desc: series_desc = series_desc.split('/')[1]
            
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
    
    def train_nn(self,X,y,test_split,epochs=10,batch_size=10, random_state=1000):
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_split,random_state=random_state)
        input_dim=X_train.shape[1]
        print('input_dim:',input_dim)
        model = Sequential()
        model.add(layers.Dense(36,input_dim=input_dim,activation='relu'))
        #model.add(layers.Dense(18,activation='relu'))
        model.add(layers.Dense(len(self._classes),activation='sigmoid'))
        print('output_dim:',len(self._classes))
        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy','categorical_accuracy'])
        model.summary()
        self.classifier=model
        #self.classifier.fit(X_train,y_train,epochs=10,verbose=True,validation_data=(X_test,y_test),batch_size=10)
        hist=self.classifier.fit(X_train,y_train,epochs=epochs,verbose=True,validation_data=(X_test,y_test),batch_size=batch_size)
        self.plot_nn_train_history(hist)
        
    def plot_nn_train_history(self,history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        x = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, acc, 'b', label='Training acc')
        plt.plot(x, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(x, loss, 'b', label='Training loss')
        plt.plot(x, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        
    def infer_nn(self,scans):
        vecs,ids=self.prepare_training_vectors_nn(scans,False)
        y_fit=self.classifier.predict(vecs)        
        hofids=[ self._classes[np.argmax(y_fit[i])] for i in range(len(y_fit)) ]
        return hofids        
        
    def train_classifier(self,X,y,test_split,random_state=1000):
        descs_train,descs_test,y_train,y_test=train_test_split(X,y,test_size=test_split,random_state=random_state)
        #classifier=LogisticRegression()
        classifier=LinearSVC()
        #classifier=SVC()
        classifier.fit(descs_train,y_train)
        scoreTest=classifier.score(descs_test,y_test)
        scoreTrain=classifier.score(descs_train,y_train)
        print('Test accuracy:', scoreTest, " train accuracy:",scoreTrain)        
        self.classifier=classifier
        
        return classifier
    
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
        
    def _predict_classifier(self,X):
        if not self.classifier: return []
        return self.classifier.predict(X)
    
    def predict_classifier_nn(self,scans):
        hofids=self.infer_nn(scans)
        for s,h in zip(scans,hofids):
            s['prediction']=h
        return scans
    
    def predict_classifier(self, scans):
        vectorized_descs=self.gen_bow_vectors(scans)
        labels=self._predict_classifier(vectorized_descs)
        for i,s in enumerate(scans):
            s['hof_id']=labels[i]
        return scans
    
    def is_valid_model(self):
        return (self.vectorizer and self.classifier)    
        
    def save_model_nn(self,rt):
        pickle.dump(self.vectorizer,open(rt+'.vec','wb'))
        self.classifier.save(rt+'.hd5')
        
    def load_model_nn(self,rt):
        self.vectorizer=pickle.load(open(rt+'.vec','rb'))
        self.classifier=tf.keras.models.load_model(rt+'.hd5')
    
    def save_model(self, file):
        pickle.dump([self.vectorizer,self.classifier],open(file,'wb'))
                    
    def load_model(self, file):
        self.vectorizer,self.classifier=pickle.load(open(file,'rb'))




if __name__ == '__main__':
    start_time = time.time()

    config = load_config(sys.argv[1])
    
    localmode = True
    if len(sys.argv) > 2 and 'xnat' in sys.argv[2]: localmode = False

    
    DICOM_FOLDER = config['preprocessing']['root_dicom_folder']
    orientation_names = config['labels']['orientation_labels']
    output_folder = config['testing']['root_output_folder']
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    print(f"dicom folder = {DICOM_FOLDER}")

    scans_list_of_dicts = []
    scans_skipped = []

    for root, dirs, files in os.walk(DICOM_FOLDER, topdown=False):
       # check if folder has dicom files
       if len(files)>0 and 'dcm' in list(set([os.path.basename(i).rsplit('.', 1)[-1] for i in files])):

            dicom_files = natsorted([i for i in files if os.path.basename(i).rsplit('.', 1)[-1] == 'dcm'])

            frames = len(dicom_files)
            print("Now reading:", os.path.join(root, dicom_files[0]))
            dcm_file = pydicom.read_file(os.path.join(root, dicom_files[0]))

            # check for DICOM Tag (0008,103E) Series description - if it does not have, then skip
            # some scans on XNAT (eg: 1-SR1) do not have series desc
            # enclose within try/except in case the tag is absent
            try:
                series_description = dcm_file.SeriesDescription
            except:
                # skip scan
                print(f"Skipped series - no series descriptions")
                continue

            orientation = orientation_names[get_image_orientation(dcm_file)]             

            if localmode:
                # Option1: extract series number from scan - this might lead to duplicate series numbers in cases where multiple scans have same series number (eg: TCGA-06-0158)
                series_number = dcm_file.SeriesNumber
            else:
                # Option2: extract series number from XNAT assigned scan folder names - in this scans with same series number are labeled as *-MR*
                # eg: root = /home/scans/SCANS/1-MR2/DICOM
                # we want to extract "1-MR2" from root
                series_number = os.path.basename(os.path.dirname(root))

            scan_dict = {'series_number': str(series_number), 'frames': str(frames), 'series_description': series_description, 'orientation': orientation}

            # check for DICOM Tag (0008,0008) Image Type - if its not ORIGINAL/PRIMARY, then skip
            # enclose within try/except in case the tag is absent
            try:
                if not all([i in dcm_file[0x08,0x08].value for i in ['ORIGINAL', 'PRIMARY']]): 
                    scan_dict['prediction'] = 'OT_derived'
                    scans_skipped.append(scan_dict)
                    continue
            except:
                pass

            # check for DICOM Tag (0018,0025) Angio Flag Attribute - if 'Y' then skip
            # enclose within try/except in case the tag is absent
            try:                       
                if dcm_file[0x18,0x25].value == 'Y': 
                    scan_dict['prediction'] = 'OT_angio'
                    scans_skipped.append(scan_dict)
                    continue
            except:
                pass


            scans_list_of_dicts.append(scan_dict)

           
    print("There are {} scans to be classified".format(len(scans_list_of_dicts)))
    # pprint.pprint(scans_list_of_dicts)
    hof_cl=HOF_Classifier()
    hof_cl.load_model_nn(config['trained_models']['classifier1'])
    scans_classified=hof_cl.predict_classifier_nn(scans_list_of_dicts)
    # pprint.pprint(scans_classified)

    #now write the classified data into a new file
    df = pd.DataFrame.from_dict(scans_classified+scans_skipped)


    ######################### Read classifier1 predictions and perform following heuristics (these were empirically determined)
    # if series description contains T2/FLAIR but scan has been classified OT by classifier1, then override that so that is fed to classifier2
    df.loc[(df['series_description'].str.contains('T2|FLAIR', na=False, case=False, regex=True)) & (df['prediction'] == 'OT'), 'prediction'] = 'anatomical'

    # hotfixed for now - todo - retrain classifier1
    df.loc[(df['series_description'].str.contains('TRA_T', na=False, case=False)) & (df['prediction'] == 'OT'), 'prediction'] = 'anatomical'

    # if series description contains following but scan has *NOT* been classified OT by classifier1, then override that with 'OT' so that it is *NOT* fed to classifier2
    df.loc[(df['series_description'].str.contains('task|lang|word|navigation|rest', na=False, case=False, regex=True)) & (df['prediction'] != 'OT'), 'prediction'] = 'OT'
    df.loc[(df['series_description'].str.contains('design|somersault|exorcist|bolus', na=False, case=False, regex=True)) & (df['prediction'] != 'OT'), 'prediction'] = 'OT'
    df.loc[(df['series_description'].str.contains('carotid|aahscout|plane_loc', na=False, case=False, regex=True)) & (df['prediction'] != 'OT'), 'prediction'] = 'OT'
    df.loc[(df['series_description'].str.contains('unknown|rois_of|localizer|document', na=False, case=False, regex=True)) & (df['prediction'] != 'OT'), 'prediction'] = 'OT'
    df.loc[(df['series_description'].str.contains('dti|dynamic|diffusion|diff', na=False, case=False, regex=True)) & (df['prediction'] != 'OT'), 'prediction'] = 'OT'
    df.loc[(df['series_description'].str.contains('perfusion|dwi|adc|swi', na=False, case=False, regex=True)) & (df['prediction'] != 'OT'), 'prediction'] = 'OT'
    df.loc[(df['series_description'].str.contains('tracew|posdisp|cow|orbits', na=False, case=False, regex=True)) & (df['prediction'] != 'OT'), 'prediction'] = 'OT'
    df.loc[(df['series_description'].str.contains('cbf|cbv|mag|pha', na=False, case=False, regex=True)) & (df['prediction'] != 'OT'), 'prediction'] = 'OT'
    df.loc[(df['series_description'].str.contains('fmri|subtract|collection|motor', na=False, case=False, regex=True)) & (df['prediction'] != 'OT'), 'prediction'] = 'OT'
    df.loc[(df['series_description'].str.contains('medic|dixon|trufi|msma|tip|stir', na=False, case=False, regex=True)) & (df['prediction'] != 'OT'), 'prediction'] = 'OT'
    df.loc[(df['series_description'].str.contains('3D Object Data', na=False, case=False)) & (df['prediction'] != 'OT'), 'prediction'] = 'OT'

    # if n_frames < *empirically determined number* but scan has *NOT* been classified OT by classifier1, then override that with 'OT' so that it is *NOT* fed to classifier2
    # This is done because otherwise the scans with low frames get propagated and ultimately gets messed up in coregistration stage
    df.loc[(df['frames'].astype(int) < 10) & (~df['prediction'].str.contains('OT')), 'prediction'] = 'OT_lowframes'
    ########################################################################


    # sort by series number
    df = df.reindex(index=order_by_index(df.index, index_natsorted(df.series_number)))
    print(f"\n\nClassification results (classifier1): \n {df}") 
    df.to_csv(config['testing']['root_pred_classifier1'], index=False)

    if not df[~df['prediction'].str.contains('OT')].empty:
        sys.exit(0)
    else:
        print(f"Classifier1 could not find any suitable scans to process from this session")
        convert_df_meta_to_bash_parsable_text(df, config['testing']['root_pred_classifier_meta_txt'])
        sys.exit(1)