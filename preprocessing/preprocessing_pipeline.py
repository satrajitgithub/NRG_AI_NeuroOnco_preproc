import os
import sys

import yaml
import dicom_handlers as DPF
import nifti_handlers as NPF
import time
import pandas as pd

import warnings
warnings.simplefilter(action='ignore')

sys.path.append(os.environ['SCRIPT_ROOT']) # Adds higher directory to python modules path.
from src.classification_aggregation import convert_df_meta_to_bash_parsable_text
from utils.utils import load_config

if __name__ == '__main__':
    start_time = time.time()

    config = load_config(sys.argv[1])

    localmode = True
    if len(sys.argv) > 2 and 'xnat' in sys.argv[2]: localmode = False


    x_image_size = config['data_preparation']['image_size_x']
    y_image_size = config['data_preparation']['image_size_y']
    z_image_size = config['data_preparation']['image_size_z']
    DICOM_FOLDER = config['preprocessing']['root_dicom_folder']
    PREPROC_FOLDER = config['preprocessing']['root_preproc_folder']
    DCM2NIIX_BIN = config['preprocessing']['dcm2niix_bin']
    FSLREORIENT_BIN = config['preprocessing']['fslreorient2std_bin']
    FSLVAL_BIN = config['preprocessing']['fslval_bin']
    output_folder = config['testing']['root_output_folder']

    DEFAULT_SIZE = [x_image_size, y_image_size, z_image_size]

    df = pd.read_csv(config['testing']['root_pred_classifier1'])

    # Filter out only anatomical scans - defined in yaml file
    df = df[df['prediction'] == 'anatomical']

    def create_directory(dir):
        if not os.path.exists(dir):
            os.makedirs(dir)


    def is_odd(number):
        return number % 2 != 0


    print('[Step 1/10] Sorting DICOM to structured folders....')
    structured_dicom_folder = DPF.sort_DICOM_to_structured_folders(DICOM_FOLDER, PREPROC_FOLDER, df['series_number'].tolist(), localmode = localmode)

    # Turn the following step on if you have problems running the pipeline
    # It will replaces spaces in the path names, which can sometimes
    # Cause errors with some tools
    print('Removing spaces from filepaths....')
    DPF.make_filepaths_safe_for_linux(structured_dicom_folder)
    #
    print('[Step 2/10] Checking and splitting for double scans in folders....')
    # Currently exclude T1hi/T1lo from this check for multiple echos.
    # DPF.split_in_series(structured_dicom_folder, exclusion_filter = df[df['prediction'].isin(['T1hi', 'T1lo'])]['series_number'].tolist())
    DPF.split_in_series(structured_dicom_folder, exclusion_filter = [])

    print('[Step 3/10] Converting DICOMs to NIFTI....')
    nifti_folder = NPF.convert_DICOM_to_NIFTI(structured_dicom_folder, DCM2NIIX_BIN)

    print('[Step 4/10] Moving RGB valued images.....')
    NPF.move_RGB_images(nifti_folder, FSLVAL_BIN)

    print('[Step 5/10] Extracting single point from 4D images....')
    images_4D_file = NPF.extract_4D_images(nifti_folder)

    print('[Step 6/10] Reorient to standard space....')
    NPF.reorient_to_std(nifti_folder, FSLREORIENT_BIN)

    print('[Step 7/10] Resampling images....')
    nifti_resampled_folder = NPF.resample_images(nifti_folder, DEFAULT_SIZE)

    print('[Step 8/10] Extracting slices from images...')
    nifti_slices_folder = NPF.slice_images(nifti_resampled_folder)

    print('[Step 9/10] Rescaling image intensity....')
    NPF.rescale_image_intensity(nifti_slices_folder)

    print('[Step 10/10] Creating label file....')
    NPF.create_label_file(nifti_slices_folder, images_4D_file)

    elapsed_time = time.time() - start_time

    print("Time taken =", elapsed_time, "seconds")
