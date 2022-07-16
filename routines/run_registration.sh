#!/bin/bash

source $SCRIPT_ROOT/utils/bash_utils.sh
banner "Starting Registration"

echo "Checking for segmentationready flag..."
if [[ $1 == *xnat* ]];then 
  project=$2
  subject=$3
  session=$4

  segmentationready=$(get_custom_flag_xnat 'segmentationready' $project $subject $session)
else
  segmentationready=$(get_custom_flag)
fi

echo "segmentationready="$segmentationready

if [[ $segmentationready != *"true"* ]];then 
  echo "Sorry! This session is NOT segmentationready"
else
  echo "This session is segmentationready - performing registration:"
  $SCRIPT_ROOT/wrapper_scripts/registration.sh "${@:1}"
fi