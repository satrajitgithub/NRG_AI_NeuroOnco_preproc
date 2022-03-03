import glob
import os
from collections import defaultdict
import pprint

config = defaultdict(lambda: None)
config['labels'] = {}
config['labels']['scan_labels'] = ['T1', 'T1c', 'T2', 'OT', 'Flair', 'OT', 'OT', 'OT', 'OT']
config['labels']['orientation_labels'] = ['3D', 'Ax', 'Cor', 'Sag', 'Obl', '4D', 'UNKNOWN']

config['trained_models'] = {}
config['trained_models']['classifier1'] = os.path.join(os.environ['SCRIPT_ROOT'], 'trained_models', 'scan_classifier_nn.01.14.2022')
config['trained_models']['classifier2'] = os.path.join(os.environ['SCRIPT_ROOT'], 'trained_models', 'model_all_brain_tumor_data.hdf5')


config['preprocessing'] = {}
config['preprocessing']['root_dicom_folder'] = os.path.abspath('/input/SCANS')
config['preprocessing']['root_preproc_folder'] = os.path.join('/output', 'SCANTYPES_TEMP')
config['preprocessing']['dcm2niix_bin'] = os.environ['DCM2NIIX_PATH'] 
config['preprocessing']['fslreorient2std_bin'] = os.path.join(os.environ['FSL_PATH'], 'fslreorient2std')
config['preprocessing']['fslval_bin'] = os.path.join(os.environ['FSL_PATH'], 'fslval')

config['data_preparation'] = {}
config['data_preparation']['image_size_x'] = 256
config['data_preparation']['image_size_y'] = 256
config['data_preparation']['image_size_z'] = 25

config['testing'] = {}
config['testing']['root_output_folder'] = os.path.join('/output', 'SCANTYPES')
config['testing']['root_dcm_acq'] = os.path.join(config['testing']['root_output_folder'], 'acq_params.csv')
config['testing']['root_pred_classifier1'] = os.path.join(config['testing']['root_output_folder'], 'Predictions_classifier1.csv')
config['testing']['root_pred_classifier2'] = os.path.join(config['testing']['root_output_folder'], 'Predictions_classifier2.csv')
config['testing']['root_pred_classifier2_per_slice'] = os.path.join(config['testing']['root_output_folder'], 'Predictions_classifier2_per_slice.csv')
config['testing']['root_pred_classifier_meta'] = os.path.join(config['testing']['root_output_folder'], 'Predictions_classifier_meta.csv')
config['testing']['root_pred_classifier_meta_txt'] = os.path.join(config['testing']['root_output_folder'], 'Predictions_classifier_meta.txt')
config['testing']['root_pred_classifier_meta_with_provenance'] = os.path.join(config['testing']['root_output_folder'], 'Predictions_classifier_meta_with_provenance.csv')
config['testing']['root_anat_seg_patient'] = os.path.join('/output', 'ANAT_PATIENT_SPACE')

config['network'] = {}
config['network']['batch_size'] = 32
config['network']['nb_epoch'] = 100