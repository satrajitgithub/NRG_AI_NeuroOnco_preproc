import warnings
warnings.simplefilter(action='ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)


from tensorflow.keras.models import load_model
import numpy as np
import data_IO as data_IO
import tensorflow as tf
import yaml
import os
from pathlib import Path
import pandas as pd
import sys
from natsort import natsorted
import time

sys.path.append(os.environ['SCRIPT_ROOT']) # Adds higher directory to python modules path.
from src.classification_aggregation import convert_df_meta_to_bash_parsable_text
from utils.utils import load_config

def load_labels(label_file):
    labels = np.genfromtxt(label_file, dtype='str')
    if labels.size == 0:
        print(f"{test_label_file} is empty, probably because this session does not contain any required scans. Please check results from classifier1.")
        sys.exit(1)
    else:
        label_IDs = labels[:, 0]
        label_IDs = np.asarray(label_IDs)
        label_values = labels[:, 1].astype(int)
        extra_inputs = labels[:, 2:].astype(float)
        np.round(extra_inputs, 2)

        N_classes = len(np.unique(label_values))

        # Make sure that minimum of labels is 0
        label_values = label_values - np.min(label_values)

        return label_IDs, label_values, N_classes, extra_inputs

if __name__ == '__main__':
    start_time = time.time()

    config = load_config(sys.argv[1])

    localmode = True
    if len(sys.argv) > 2 and 'xnat' in sys.argv[2]: localmode = False

    batch_size = 1

    root_preproc_folder = config['preprocessing']['root_preproc_folder']
    test_label_file = os.path.abspath(os.path.join(root_preproc_folder, "DATA", 'labels.txt'))
    output_folder = config['testing']['root_output_folder']
    x_image_size = config['data_preparation']['image_size_x']
    y_image_size = config['data_preparation']['image_size_y']
    prediction_names = config['labels']['scan_labels']
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # model_name = os.path.basename(os.path.normpath(model_file)).split('.hdf5')[0]
    out_file = os.path.join(output_folder, 'Predictions.csv')
    out_file_clsfr1 = config['testing']['root_pred_classifier1']
    out_file_clsfr2 = config['testing']['root_pred_classifier2']
    out_file_clsfr2_per_slice = config['testing']['root_pred_classifier2_per_slice']
    out_file_meta = config['testing']['root_pred_classifier_meta']

    test_image_IDs, test_image_labels, _, extra_inputs = load_labels(test_label_file)

    optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0, amsgrad=False)

    model = load_model(config['trained_models']['classifier2'])
    model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['categorical_accuracy']
    )

    NiftiGenerator_test = data_IO.NiftiGenerator2D_ExtraInput(batch_size,
                                                               test_image_IDs,
                                                               test_image_labels,
                                                               [x_image_size, y_image_size],
                                                               extra_inputs)

    header = ("prediction", "i_label")
    rows = list()
    i_files = list()

    with open(out_file, 'w') as prediction_csv:
        for i_file, i_label, i_extra_input in zip(test_image_IDs, test_image_labels, extra_inputs):
            print(i_file)

            image = NiftiGenerator_test.get_single_image(os.path.join(root_preproc_folder, 'NIFTI_SLICES', i_file))
            supplied_extra_input = np.zeros([1, 1])
            supplied_extra_input[0, :] = i_extra_input
            prediction = model.predict([image, supplied_extra_input])
            prediction_csv.write(i_file + '\t' + str(np.argmax(prediction) + 1) + '\t' + str(i_label) + '\n')


            #
            i_files.append(i_file)
            rows.append([str(np.argmax(prediction) + 1) , str(i_label)])


    df = pd.DataFrame.from_records(rows, columns=header, index=i_files)
    df.index.name = 'slices'
    df.to_csv(out_file_clsfr2_per_slice)

    df = pd.read_csv(out_file_clsfr2_per_slice)
    df[['scan', 'slice_number']] = df['slices'].str.rsplit('+', 1, expand=True)
    df.drop(columns = ['i_label', 'slices'], inplace=True)
    df = df[['scan', 'slice_number', 'prediction']]
    df_agg = df.groupby(['scan'])['prediction'].agg(lambda x:x.value_counts().index[0]).reset_index()
    prediction_names_dict = {idx+1:i for idx,i in enumerate(prediction_names)}
    df_agg.replace({"prediction": prediction_names_dict}, inplace = True)
    df_agg[['scan', 'series_number']] = df_agg['scan'].str.rsplit('+', 1, expand=True)
    df_agg.drop(columns = ['scan'], inplace=True)
    df_agg = df_agg[['series_number', 'prediction']]
    df_agg['series_number'] = df_agg['series_number'].apply(lambda x: str(x).strip('series'))

    df_agg.to_csv(out_file_clsfr2, index = False)
    print(f"\n\nClassification results (classifier2): \n {df_agg}") 

    # ~~~~~~~~~~~~~~~~~~~~~~~~~ Aggregation from classifier1 and classifier2 and creating df_meta ~~~~~~~~~~~~~~~~~~~~~~~~~
    df1 = pd.read_csv(out_file_clsfr1, index_col = 'series_number')
    df2 = pd.read_csv(out_file_clsfr2, index_col = 'series_number')

    df1.index = df1.index.map(str)
    df2.index = df2.index.map(str)

    # ~~~~~~~~~~~~~~~~ Handling scans which has double/multiple acquisitions (eg: single T2 has PD+T2)~~~~~~~~~~~~~~~~~
    f = []
    for sn in df2.index: 
        if sn in df1.index:
            df1.at[sn, 'prediction'] = df2.at[sn, 'prediction']
        # if there are two scans per series (eg: T2 has PD+T2), then scans are named as 2_scan_1, 2_scan_2. in that case this will go into the else block
        else:
            f.append(sn)

    for a in f:
        a1 = a.split('_')[0]
        d = dict(df1.loc[a1])
        d['prediction'] = df2.loc[a]['prediction']
        df1.loc[a] = d
        
    for a in list(set([i.split('_')[0] for i in f])):
        df1.drop([a], inplace=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Issue: https://github.com/satrajitgithub/NRG_AI_Neuroonco_preproc/issues/20
    # Following checks if there are still any anatomical scans in meta results which have predictions from classifier1 which might
    # have trickled down. These are the scans for which classifier2 makes no predictions (i.e. skips them) due to some reason - most
    # often because dcm2niix fails due to these being derived scans. If there are any such scans, we hard assign the 'OT' label to them
    # because there should be no anatomical labels of classifier1 present in the final predictions.
    df1.loc[df1['prediction'] == 'anatomical', 'prediction'] = 'OT'

    df1 = df1.reindex(index=natsorted(df1.index))

    df1.to_csv(out_file_meta)

    df1.reset_index(inplace = True)
    print(f"\n\nClassification results (meta): \n {df1}")

    if not df_agg[~df_agg['prediction'].str.contains('OT')].empty:
        if not df_agg[~(df_agg['prediction'] == 'T1')].empty:
            sys.exit(0)
        else:
            print(f"Classifier2 could find only pre-contrast T1 scan from this session, which cannot be segmented")
            convert_df_meta_to_bash_parsable_text(df1, config['testing']['root_pred_classifier_meta_txt'])
            sys.exit(1)
    else:
        print(f"Classifier2 could not find any suitable scans to process from this session")
        convert_df_meta_to_bash_parsable_text(df1, config['testing']['root_pred_classifier_meta_txt'])
        sys.exit(1)
        