#!/bin/bash

source $SCRIPT_ROOT/utils/bash_utils.sh
banner "Uploading dicom-seg as roi-assessor"

if [[ $1 == *xnat* ]];then 
  project=$2
  subject=$3
  session=$4

  segmentationready=$(get_custom_flag_xnat 'segmentationready' $project $subject $session)
else
  echo "Sorry! This operation is ONLY for XNAT mode"
  exit 0
fi

if [[ $segmentationready != *"true"* ]];then 
  echo "Sorry! This session is NOT segmentationready"
else
  $SCRIPT_ROOT/wrapper_scripts/dcmseg_to_roiassessor.sh $project $subject $session 
fi