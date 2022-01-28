#!/bin/bash

# ~~~~~~~~~~~~~~~~~~~~ Set paths ~~~~~~~~~~~~~~~~~~~~~~~~
echo "Updated: 01/28/2022 - 3:20pm"
# export TERM=xterm
export SCRIPT_ROOT=/NRG_AI_NeuroOnco_preproc
export DCM2NIIX_PATH=/dcm2niix
export FSL_PATH=/opt/fsl-5.0.11/bin
export ATLAS_PATH=/atlas
export ROBEX_PATH=/opt/ROBEX_linux
export DCMTK_PATH=/opt/dcmtk366/bin
export DCMQI_PATH=/opt/dcmqi/bin


# export SCRIPT_ROOT=/home/satrajit.chakrabarty/NRG_AI_Neuroonco_preproc
# export SESSION_PATH=/home/satrajit.chakrabarty/NRG_AI_Neuroonco_preproc_workspace/subject1/session1
# export OUTPUT_PATH=/home/satrajit.chakrabarty/NRG_AI_Neuroonco_preproc_workspace/resources
# export DCM2NIIX_PATH=/export/mricron/mricron-20190902/dcm2niix
# export FSL_PATH=/export/fsl/fsl-5.0.10/bin
# export ATLAS_PATH=/scratch/satrajit.chakrabarty/fsl_scripts

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