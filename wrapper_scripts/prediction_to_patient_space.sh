#!/bin/bash

ANAT_PATIENT_SPACE=/input_scantype/ANAT_PATIENT_SPACE
ANAT_COREGISTERED=/input_registration/ANAT_COREGISTERED
ANAT_REGISTERED=/input_registration/ANAT_REGISTERED
SEGMENTATION=/input_segmentation

prediction=`ls $SEGMENTATION/prediction.nii.gz`
omat=`ls $ANAT_COREGISTERED/SRI_to_patient_omat`
source $ANAT_PATIENT_SPACE/reference_scan.txt
targ_scan=$ANAT_PATIENT_SPACE"/"$targ_scan_name

echo -e "prediction="$prediction
echo -e "omat="$omat
echo -e "targ_scan="$targ_scan

echo "[INFO] Using omat_patient to convert prediction.nii.gz to patient space --> prediction_patient.nii.gz.."
echo flirt -v -in $prediction -ref $targ_scan -out /output/prediction_patient.nii.gz -init $omat -applyxfm -interp nearestneighbour
flirt -v -in $prediction -ref $targ_scan -out /output/prediction_patient.nii.gz -init $omat -applyxfm -interp nearestneighbour
