#!/bin/bash

# echo "DEBUG"
# echo "input_session"`ls input_session/*`
# echo "input_scantype"`ls input_scantype/*`
# echo "input_registration"`ls input_registration/*`
# echo "input_segmentation"`ls input_segmentation/*`

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Structural scan check ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Reason: if (0008,0102)/(0008,0104) absent in dicom metadata, itkimage2segimage fails
# Solution: We need to inject back the tag (0008,0102)/(0008,0104) with dummy values. This is done using dcmodify from dcmtk 
# For more details, check: https://qiicr.gitbook.io/dcmqi-guide/faq#conversion-fails-with-the-missing-attribute-s-error-what-should-i-do

ANAT_PATIENT_SPACE=/input_scantype/ANAT_PATIENT_SPACE
ANAT_COREGISTERED=/input_registration/ANAT_COREGISTERED
ANAT_REGISTERED=/input_registration/ANAT_REGISTERED
SEGMENTATION=/input_segmentation_patient

STRUCT_SCAN=/output/structural_scan

source $ANAT_PATIENT_SPACE/reference_scan_dcmseg.txt
echo "Target scan ="$targ_scan_name
A="$(echo $targ_scan_name | cut -d'_' -f1)"
dcm_seriesnumber=${A#"series"}
echo "dcm_seriesnumber="$dcm_seriesnumber

# for xnat
if [ -d /input_session/SCANS ];then
	dcm_dir=`ls -1d /input_session/SCANS/$dcm_seriesnumber*/`
# for docker/local
else
	dcm_dir=`ls -1d /input_session/$dcm_seriesnumber*/`
fi
echo "dcm_dir="$dcm_dir

mkdir -p $STRUCT_SCAN
find $dcm_dir -type f -name "*.dcm" -exec cp -r {} $STRUCT_SCAN/ \;
echo "Found this number of dicom files in output:" 
ls $STRUCT_SCAN/*.dcm | wc -l
echo $DCMTK_PATH/dcmodify --version -v -i "ProcedureCodeSequence[0].CodingSchemeDesignator=99UNKNOWN" $STRUCT_SCAN/*.dcm
$DCMTK_PATH/dcmodify -nb -i "ProcedureCodeSequence[0].CodingSchemeDesignator=99UNKNOWN" $STRUCT_SCAN/*.dcm
$DCMTK_PATH/dcmodify -nb -i "ProcedureCodeSequence[0].CodeMeaning=99UNKNOWN" $STRUCT_SCAN/*.dcm
$DCMTK_PATH/dcmodify -nb -i "ProcedureCodeSequence[0].CodeMeaning=99UNKNOWN" $STRUCT_SCAN/*.dcm


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Actual conversion ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dcm_ref=`ls -1d $STRUCT_SCAN/`
inputimage=${SEGMENTATION}/prediction_patient.nii.gz

echo "inputImageList="$inputimage
echo "dcm_ref="$dcm_ref
echo "Found followig files in dcm_ref:"
ls $dcm_ref/*

echo $DCMQI_PATH/itkimage2segimage --inputImageList $inputimage --inputDICOMDirectory $dcm_ref/ --inputMetadata $SCRIPT_ROOT/wrapper_scripts/patient_space_to_dcmseg.json --outputDICOM /output/prediction_patient.dcm
$DCMQI_PATH/itkimage2segimage --inputImageList $inputimage --inputDICOMDirectory $dcm_ref/ --inputMetadata $SCRIPT_ROOT/wrapper_scripts/patient_space_to_dcmseg.json --outputDICOM /output/prediction_patient.dcm

# Cleanup
rm -rf $STRUCT_SCAN

