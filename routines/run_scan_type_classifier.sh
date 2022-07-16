#!/bin/bash

source $SCRIPT_ROOT/utils/bash_utils.sh

banner "Starting Scan-type Classifier.."
segmentationready="false"

# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Scan-type Classifier ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if [[ $1 == *xnat* ]];then 
  path_to_config=$SCRIPT_ROOT/configs/config_xnat.py
else
  path_to_config=$SCRIPT_ROOT/configs/config_docker.py
fi


banner "classifier1_metadata"
python -u $SCRIPT_ROOT/src/classifier1_metadata.py $path_to_config $1

# if there are not only 'OT' scans but also anatomical scans present
if [ $? -eq 0 ]; then

  banner "preprocessing_pipeline"
  python -u $SCRIPT_ROOT/preprocessing/preprocessing_pipeline.py $path_to_config $1

  banner "classifier2_image"
  python -u $SCRIPT_ROOT/src/classifier2_image.py $path_to_config $1

  # if there are not only 'T1' scans but also T1c/T2/Flair scans present
  if [ $? -eq 0 ]; then

    banner "classification_aggregation"
    python -u $SCRIPT_ROOT/src/classification_aggregation.py $path_to_config $1

    segmentationready="true"
  fi

  # cleanup
  rm -rf /output/SCANTYPES_TEMP
fi

if [[ $1 == *xnat* ]];then 

  echo "Processing finished. Now uploading scantypes to XNAT:"
  
  echo "Getting JSESSIONID by following command:"
  echo curl -u $XNAT_USER:$XNAT_PASS -X GET $XNAT_HOST/data/JSESSION
  
  jsess=$(xnat_jsess)
  project=$2
  subject=$3
  session=$4

  echo "Giving following inputs to upload_scantypes() func:"
  echo "jsess="$jsess
  echo "project="$project
  echo "subject="$subject
  echo "session="$session
  
  upload_scantypes /output/SCANTYPES/Predictions_classifier_meta.txt $jsess $project $subject $session
  set_custom_flag_xnat $segmentationready $jsess $project $subject $session

else
  echo "Setting segmentationready flag="$segmentationready
  set_custom_flag 'segmentationready' $segmentationready
fi
