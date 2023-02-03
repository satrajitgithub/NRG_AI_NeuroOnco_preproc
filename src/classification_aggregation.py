import warnings
warnings.simplefilter(action='ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import os
import shutil
import yaml
import json
from pathlib import Path
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import sys
from natsort import natsorted
from functools import reduce

sys.path.append(os.environ['SCRIPT_ROOT']) # Adds higher directory to python modules path.
from utils.utils import load_config

def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def mark_filtered_scans_in_meta(df1, df2, ref_series_number):
    df1['series_number'] = df1['series_number'].astype(str)
    df2['series_number'] = df2['series_number'].astype(str)
    
    # Mark the scans chosen for segmentation with "-seg"
    df1.loc[df1['series_number'].isin(df2['series_number']), 'prediction'] = df1.loc[df1['series_number'].isin(df2['series_number']), 'prediction'].astype(str) + '-seg'

    # Mark the scan chosen as reference for registration
    df1.loc[df1['series_number'] == ref_series_number, 'reference'] = True
    return df1


def merge_predictions_of_dual_scans(df1):
    # If we have 2 T2 and 2 PD, merge them as 2 T2-PD
    df1['series_number'] = df1['series_number'].astype(str).str.split('_', n = 1).str[0]

    df1.drop_duplicates(inplace=True)

    # df1 = df1.groupby(['series_number']).agg({'prediction': '-'.join, 'frames': 'first', 'series_description': 'first', 'orientation': 'first'})
    df1 = df1.groupby(['series_number', 'frames', 'series_description', 'orientation', 'reference'], dropna=False).agg({'prediction': '-'.join})
    df1 = df1.reindex(index=natsorted(df1.index))
    df1.reset_index(inplace=True)

    return df1

def add_provenance_to_meta(config, df1):
    df1['series_number'] = df1['series_number'].astype(str)

    # add classifier1 predictions
    cl1 = pd.read_csv(config['testing']['root_pred_classifier1'])
    cl1['series_number'] = cl1['series_number'].astype(str)
    cl1.rename(columns={'prediction': 'classifier1'}, inplace = True)
    cl1.drop(columns = ['frames', 'series_description', 'orientation'], inplace = True)

    # add classifier2 predictions
    cl2 = pd.read_csv(config['testing']['root_pred_classifier2'])
    cl2['series_number'] = cl2['series_number'].astype(str)
    # If we have 2 T2 and 2 PD, merge them as 2 T2-PD
    cl2['series_number'] = cl2['series_number'].astype(str).str.split('_', n = 1).str[0]
    cl2.drop_duplicates(inplace=True)
    cl2 = cl2.groupby(['series_number']).agg({'prediction': '-'.join}).reset_index()
    cl2 = cl2.reindex(index=natsorted(cl2.index))
    cl2.rename(columns={'prediction': 'classifier2'}, inplace = True)
        
    df1 = reduce(lambda  left,right: pd.merge(left,right, on=['series_number'], how = 'outer'), [df1, cl1, cl2])
    
    print(f"\n\nClassification results (meta - final): \n {df1}")

    return df1


def convert_df_meta_to_bash_parsable_text(df_meta, out_file_meta_txt):
    
    df_meta = df_meta[['series_number', 'prediction']]
    df_meta = df_meta['series_number'].astype(str)+':'+df_meta['prediction'].astype(str)

    bash_op = '\n'.join(df_meta.tolist()+['']) # joins each command with a newline
    print(f"bash_op:\n{bash_op}")
    file1 = open(out_file_meta_txt,"w")
    file1.writelines(bash_op)
    file1.close()

def conditionally_prune_cor_sag_sequences(df, series_class):

    # # Method 1: Naive - check for 'cor' or 'sag' in series description and exclude based on that
    # # Condition: check if after removing scans with 'cor'/'sag' in series description, we still have remaining scan(s) of that series class
    # # If yes, only then prune cor/sag scans, otherwise do not
    # if series_class in df[~((df.prediction == series_class) & (df.series_description.str.contains("cor|sag", case = False, regex=True)))]['prediction'].tolist():
    #     df = df[~((df.prediction == series_class) & (df.series_description.str.contains("cor|sag", case = False, regex=True)))]
    # return df

    # Method 2: Based on dicom metadata - check orientation of scan (determined previously) and exclude based on that
    if series_class in df[~((df.prediction == series_class) & (df.orientation.isin(['Cor', 'Sag', '3D'])))]['prediction'].tolist():
        df = df[~((df.prediction == series_class) & (df.orientation.isin(['Cor', 'Sag', '3D'])))]
    return df


if __name__ == '__main__':
    config = load_config(sys.argv[1])

    localmode = True
    if len(sys.argv) > 2 and 'xnat' in sys.argv[2]: localmode = False

    prediction_names = config['labels']['scan_labels']
    root_dicom_folder = config['preprocessing']['root_dicom_folder']
    root_preproc_folder = config['preprocessing']['root_preproc_folder']
    output_folder = config['testing']['root_output_folder']
    root_out_folder = config['testing']['root_anat_seg_patient']

    nifti_dir = os.path.join(root_preproc_folder, 'NIFTI')    

    out_file_meta = config['testing']['root_pred_classifier_meta']
    out_file_meta_txt = config['testing']['root_pred_classifier_meta_txt']

    Path(root_out_folder).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(out_file_meta)

    # ~~~~~~~~~~~~~~~~~~~ Selection of final 4 scans - T1,T1c,T2,FLAIR - based on following rules/heuristics ~~~~~~~~~~~~~~~~~~~

    # Rule: Keep only the scans of following types: 'T1', 'T1c', 'T2', 'Flair'
    df = df[df['prediction'].isin(['T1', 'T1c', 'T2', 'Flair'])]

    # Rule: Conditionally prune cor/sag scans
    for series_class in ['T1', 'T1c', 'T2', 'Flair']:
        df = conditionally_prune_cor_sag_sequences(df, series_class)

    # Heuristic: consider HEMO/T2_FFE/FLASH/GRECHO/T2*/T2star as T2 but keep some kind of preference (that if normal T2 exists, then prefer that)
    is_predicted_T2_filter = (df['prediction'] == 'T2') 
    is_bad_T2_filter = (df['series_description'].str.contains("hemo|ffe|flash|grecho|star|T2[*]", case = False, regex=True))
    if not df[is_predicted_T2_filter & ~is_bad_T2_filter].empty:
        drop_idx = df[is_predicted_T2_filter & is_bad_T2_filter].index
        print("Dropping bad T2: series_number=", df.loc[drop_idx]['series_number'].tolist())
        df.drop(index=drop_idx, inplace = True)

    # Rule: If there are multiple scans of same type, then keep the one with highest number of frames
    idx = df.groupby(['prediction'])['frames'].transform(max) == df['frames']
    df = df[idx]

    # Heuristic: If there are multiple scans of same type with same number of frames, then keep the one with highest series_number
    # Source: https://stackoverflow.com/questions/41525911/group-by-pandas-dataframe-and-select-latest-in-each-group
    df_final = df.groupby(['prediction']).tail(1)

    print("Classification results (filtered):\n", df_final)

    # ~~~~~~~~~~~~~~~~~~~ Determine reference scan for registration from the final 4 scans - based on following rules/heuristics ~~~~~~~

    # Preferably use only axial/3D scans as reference for registration. But sometimes, certain sessions do not contain
    # any Ax or 3D sequences. In that case, need to skip this constraint.
    if not df_final[df_final['orientation'].isin( ['Ax', '3D'])].empty:      
        df_constrained = df_final[df_final['orientation'].isin( ['Ax', '3D'])]
    else:
        df_constrained = df_final

    # check if either T1/T1c present
    if any([i in df_constrained['prediction'].tolist() for i in ['T1', 'T1c']]):
        # todo: if T1 and T1c match exactly in all parameters (#frames etc) then choose T1c as the reference
        ref_series_number = df_constrained[df_constrained['prediction'].isin(['T1', 'T1c'])].sort_values('frames', ascending = False).iloc[0].series_number
    # otherwise go for T2/Flair
    elif any([i in df_constrained['prediction'].tolist() for i in ['T2', 'Flair']]):
        ref_series_number = df_constrained[df_constrained['prediction'].isin(['T2', 'Flair'])].sort_values('frames', ascending = False).iloc[0].series_number

    print("Reference_scan:", ref_series_number)

    # ~~~~~~~~~~~~~~~~~~~ Determine reference scan for dicom-seg from the final 4 scans - preference in order: T1c - Flair - T2 - T1 ~~~~~~~
    # convert to categorical series for custom sorting
    # https://stackoverflow.com/questions/13838405/custom-sorting-in-pandas-dataframe/27009771
    df_final['prediction'] = pd.Categorical(df_final['prediction'], ["T1c", "Flair", "T2", "T1"])
    ref_series_number_dcmseg = df_final.sort_values('prediction').iloc[0].series_number
    print("ref_series_number_dcmseg:", ref_series_number_dcmseg)

    # ~~~~~~~~~~~~~~~~~~~ Add provenance to df_meta and save as final result  ~~~~~~~
    df_meta = pd.read_csv(out_file_meta)
    df_meta = mark_filtered_scans_in_meta(df_meta, df_final, str(ref_series_number))
    df_meta = merge_predictions_of_dual_scans(df_meta)
    df_meta = add_provenance_to_meta(config, df_meta)
    df_meta.to_csv(config['testing']['root_pred_classifier_meta_with_provenance'], index = False)
    print("df_meta\n", df_meta)
    
    convert_df_meta_to_bash_parsable_text(df_meta, out_file_meta_txt)


    '''
    df_final_dict is of form: 
    {4: {'prediction': 'Flair'},
     5: {'prediction': 'T1'},
     6: {'prediction': 'T2'},
     11: {'prediction': 'T1c'}}
    '''
    df_final_dict = df_final[['series_number', 'prediction']].set_index('series_number').T.to_dict()
    # print(df_final_dict)

    for series_id in df_final_dict.keys():
        scan_matcher = f'series{series_id}'
        scan_name = df_final_dict[series_id]['prediction']
        for path in Path(nifti_dir).rglob(f'*{scan_matcher}.nii.gz'):
            # Remove "Scan_1" or "Scan_2" suffixes from name (valid for dual acquisition scans eg: when T2 is taken from T2/PD)
            # print("scan_matcher", scan_matcher)
            if 'Scan_1' in scan_matcher: scan_matcher = scan_matcher.replace('_Scan_1','')
            if 'Scan_2' in scan_matcher: scan_matcher = scan_matcher.replace('_Scan_2','')

            out_file = f"{scan_matcher}_{scan_name}.nii.gz"
            out_path = os.path.join(root_out_folder, out_file)
            print(f"{os.path.basename(path)} --> {os.path.basename(out_path)}")
            shutil.copy(path, out_path)

        if series_id == str(ref_series_number):
            bash_op = '\n'.join([f"targ_scan_name={out_file}"]+['']) # joins each command with a newline
            print(f"bash_op for reference_scan.txt:\n{bash_op}")
            file1 = open(os.path.join(root_out_folder, 'reference_scan.txt'),"w")
            file1.writelines(bash_op)
            file1.close()

        if series_id == str(ref_series_number_dcmseg):
            bash_op = '\n'.join([f"targ_scan_name={out_file}"]+['']) # joins each command with a newline
            print(f"bash_op for reference_scan_dcmseg.txt:\n{bash_op}")
            file1 = open(os.path.join(root_out_folder, 'reference_scan_dcmseg.txt'),"w")
            file1.writelines(bash_op)
            file1.close()