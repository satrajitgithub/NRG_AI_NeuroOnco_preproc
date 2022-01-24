#!/bin/bash

source $SCRIPT_ROOT/utils/bash_utils.sh
banner "Converting seg-mask from atlas space-->patient space (nifti)"

if [[ $1 == *xnat* ]];then 
  project=$2
  subject=$3
  session=$4

  segmentationready=$(get_custom_flag_xnat 'segmentationready' $project $subject $session)
else
  segmentationready=$(get_custom_flag)
fi

if [[ $segmentationready != *"true"* ]];then 
  echo "Sorry! This session is NOT segmentationready"
else
  $SCRIPT_ROOT/wrapper_scripts/prediction_to_patient_space.sh
fi