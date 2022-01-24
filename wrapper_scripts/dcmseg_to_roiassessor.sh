#!/bin/bash

# uploading file to XNAT as ROI_COLLECTION under ASSESSORS
ROI_LABEL=prediction_patient_"`date +"%m%d%Y_%H%M%S"`"
DICOMSEG="`ls /input/prediction_patient.dcm`"

echo "DICOM-SEG file: " $DICOMSEG
echo "Uploading to XNAT at: " $XNAT_HOST
echo "Project: " $1
echo "SESSION: " $3
echo "ROI_LABEL: " $ROI_LABEL


echo "Executing the following command: "
echo 'curl -u $XNAT_USER:"*******" --data-binary "@$DICOMSEG" -H "Content-Type: application/octet-stream" -X PUT $XNAT_HOST/xapi/roi/projects/$1/sessions/$3/collections/$ROI_LABEL?type=SEG'
curl -u $XNAT_USER:$XNAT_PASS --data-binary "@$DICOMSEG" -H "Content-Type: application/octet-stream" -X PUT $XNAT_HOST/xapi/roi/projects/$1/sessions/$3/collections/$ROI_LABEL?type=SEG