#!/bin/bash

# ~~~~~~~~~~~~~~~~~~~~ Set paths ~~~~~~~~~~~~~~~~~~~~~~~~
echo "Updated: 10/06/2022"
# export TERM=xterm
export SCRIPT_ROOT=/NRG_AI_NeuroOnco_preproc
export DCM2NIIX_PATH=/dcm2niix
export FSL_PATH=/opt/fsl-5.0.11/bin
export ATLAS_PATH=/atlas
export ROBEX_PATH=/opt/ROBEX_linux
export DCMTK_PATH=/opt/dcmtk366/bin
export DCMQI_PATH=/opt/dcmqi/bin


print_help() {
  echo $"Error in input!
Usage (docker): docker run [-v <HOST_DIR>:<CONTAINER_DIR>] satrajit2012/nrg_ai_neuroonco_preproc:v0 <command> --docker [args]
Usage (XNAT):  <command> --xnat #PROJECT# #SUBJECT# #SESSION# [args] 
Choose <command> from: {scan_type_classifier|registration|skullstrip|prediction_to_patient_space|patient_space_to_dcmseg|dcmseg_to_roiassessor}"
}

case "$1" in
    scan_type_classifier|registration|skullstrip|prediction_to_patient_space|patient_space_to_dcmseg|dcmseg_to_roiassessor) arg1_check=true;;
    *) print_help
            exit 1
esac

if [[ $arg1_check == *true* ]];then
  case "$2" in
      --docker) $SCRIPT_ROOT/routines/run_$1.sh "${@:2}";;
      --xnat) $SCRIPT_ROOT/routines/run_$1.sh "${@:2}";;
    *) print_help
              exit 1
  esac
fi