#!/bin/bash

source $SCRIPT_ROOT/utils/bash_utils.sh
banner "Converting seg (patient space) nifti-->dicom-seg"

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
  $SCRIPT_ROOT/wrapper_scripts/patient_space_to_dcmseg.sh
fi