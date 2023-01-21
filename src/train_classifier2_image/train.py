import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
import numpy as np
import yaml
import os
import datetime

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import PReLU

from ..data_IO import data_IO

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


def load_labels(label_file):
    labels = np.genfromtxt(label_file, dtype='str')
    label_IDs = labels[:, 0]
    label_IDs = np.asarray(label_IDs)
    label_values = labels[:, 1].astype(np.int)
    extra_inputs = labels[:, 2:].astype(np.float)
    np.round(extra_inputs, 2)

    N_classes = len(np.unique(label_values))

    # Make sure that minimum of labels is 0
    label_values = label_values - np.min(label_values)

    one_hot_labels = get_one_hot(label_values, N_classes)

    return label_IDs, one_hot_labels, N_classes, extra_inputs

def DDS_model(img_rows, img_cols, img_channels):
    model = Model()
    image_input = Input(shape=(img_rows, img_cols, img_channels), dtype='float32')

    dicom_tag_input = Input(shape=(1, ), dtype='float32')
    dicom_tag_dropped = Dropout(0.25)(dicom_tag_input)

    conv1 = Conv2D(32, (5, 5))(image_input)
    batch1 = BatchNormalization(axis=-1)(conv1)
    relu1 = PReLU()(batch1)
    conv2 = Conv2D(32, (5, 5))(relu1)
    batch2 = BatchNormalization(axis=-1)(conv2)
    relu2 = PReLU()(batch2)
    pooling1 = MaxPooling2D(pool_size=(3, 3))(relu2)

    conv3 = Conv2D(64, (5, 5))(pooling1)
    batch3 = BatchNormalization(axis=-1)(conv3)
    relu3 = PReLU()(batch3)
    conv4 = Conv2D(64, (5, 5))(relu3)
    batch4 = BatchNormalization(axis=-1)(conv4)
    relu4 = PReLU()(batch4)
    pooling2 = MaxPooling2D(pool_size=(3, 3))(relu4)

    conv5 = Conv2D(64, (5, 5))(pooling2)
    batch5 = BatchNormalization(axis=-1)(conv5)
    relu5 = PReLU()(batch5)
    conv6 = Conv2D(64, (5, 5))(relu5)
    batch6 = BatchNormalization(axis=-1)(conv6)
    relu6 = PReLU()(batch6)
    pooling3 = MaxPooling2D(pool_size=(3, 3))(relu6)

    flattened_CNN_output = Flatten()(pooling3)

    merged_inputs = concatenate([flattened_CNN_output, dicom_tag_dropped])

    dropout_inputs = Dropout(0.4)(merged_inputs)
    dense = Dense(1024, activation='relu')(dropout_inputs)

    dropout_dense = Dropout(0.4)(dense)

    predictions = Dense(8, activation='softmax')(dropout_dense)

    model = Model(inputs=[image_input, dicom_tag_input], outputs=predictions)

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0, amsgrad=False),
                  metrics=['categorical_accuracy'])

    return model


def train_model(train_image_IDs, batch_size, model, nb_epoch, data_generator):
    steps_per_epoch = int(np.floor(len(train_image_IDs)/float(batch_size)))
    print("%d steps per epoch" % steps_per_epoch)


    lr_callback = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, min_lr=0.000001, verbose=1)
    stopping_callback = EarlyStopping(monitor='loss', patience=6, verbose=1)


    model.fit_generator(data_generator,
                        epochs=nb_epoch,
                        steps_per_epoch=steps_per_epoch,
                        callbacks=[lr_callback, stopping_callback],
                        use_multiprocessing=False,
                        workers=4)

    return model


if __name__ == "__main__":
    
    '''
    train_label_file should be of the format:

    location_of_nifti                           class               is_4D
    location_of_nifti_scan1_slice1.nii.gz       0                   0
    location_of_nifti_scan1_slice2.nii.gz       0                   0
    location_of_nifti_scan1_slice1.nii.gz       0                   0
    etc.

    the scans need to be first pre-processed using the NRG_AI_NeuroOnco_preproc/preprocessing/preprocessing_pipeline.py 
    so that they are of dimension 256 x 256 x 25

    Details about 'class' column can be found in https://github.com/Svdvoort/DeepDicomSort#running-deepdicomsort

    'is_4D' is typically 0 for all scans but can be 1 in some scans that are 4-dimensional 
    (e.g., some DWI scans with multiple b-values and potentially b-vectors, and for some PWI-DSC scans, which contain multiple time points.) 

    More details of this can be found at:
    van der Voort, S.R., Smits, M., Klein, S. et al. DeepDicomSort: An Automatic Sorting Algorithm for Brain Magnetic Resonance Imaging Data. 
    Neuroinform 19, 159–184 (2021). 
    https://doi.org/10.1007/s12021-020-09475-7
    '''
    train_label_file = "/path/to/train/label/file.txt"
    output_folder = "/path/to/output/folder/"
    
    # match these values with configs/config*.py    
    x_image_size = 256
    y_image_size = 256
    batch_size = 32
    nb_epoch = 100

    model_args = {'img_rows': x_image_size, 'img_cols': y_image_size, 'img_channels': 1}
    model = models.DDS_model(**model_args)

    now = str(datetime.datetime.now()).replace(' ', '_')
    model_name = 'DDS_model_epochs' + str(nb_epoch) + '_time_' + now


    train_image_IDs, train_image_labels, N_train_classes, extra_inputs = load_labels(train_label_file)

    print("Detected %d classes in training data" % N_train_classes)


    NiftiGenerator_train = data_IO.NiftiGenerator2D_ExtraInput(batch_size,
                                                               train_image_IDs,
                                                               train_image_labels,
                                                               [x_image_size, y_image_size],
                                                               extra_inputs)

    
    model = train_model(train_image_IDs, batch_size, model, nb_epoch, NiftiGenerator_train)

    # save the new model under root/trained_models/ and change the model reference inside configs/config_{docker,xnat}.py (config['trained_models']['classifier2'])
    model.save(os.path.join(output_folder, model_name + '.hdf5'))