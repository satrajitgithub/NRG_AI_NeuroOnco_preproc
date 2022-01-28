#!/bin/bash

# while getopts r OPT
# do
#     case "$OPT" in
#         r) REG=true;; # if this flag is true, then perform registration after coregistration; else stop at coregistration
#         *) echo "invalid flag. Pass [-r] for performing registration after coregistration." >&2
#             exit 1 ;;
#     esac
# done

# if [ "$REG" = true ] ; then 
#     echo 'Registration: TRUE'
# else
# 	echo 'Registration: FALSE. Will only perform Co-registration.'
# fi

REG=true
ANAT_PATIENT_SPACE=/input/ANAT_PATIENT_SPACE
ANAT_COREGISTERED=/output/ANAT_COREGISTERED
ANAT_REGISTERED=/output/ANAT_REGISTERED

source $ANAT_PATIENT_SPACE/reference_scan.txt # setting targ_scan_name variable from here 
targ_scan=$ANAT_PATIENT_SPACE"/"$targ_scan_name

echo "ANAT_PATIENT_SPACE="$ANAT_PATIENT_SPACE
echo "ANAT_COREGISTERED="$ANAT_COREGISTERED
echo "ANAT_REGISTERED="$ANAT_REGISTERED
echo "targ_scan="$targ_scan

mkdir -p $ANAT_COREGISTERED
SECONDS=0 ; 

# Target scan: determined in $SCRIPT_ROOT/utils/classification_aggregation.py
targ_scan_sequence=`echo $targ_scan_name | cut -d'_' -f2-` # if targ_scan_name=series801_T1c.nii.gz then this is T1c.nii.gz
targ_scan_sequence=${targ_scan_sequence%%.*} # T1c.nii.gz --> T1c

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~ Co-registration ~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cp $targ_scan $ANAT_COREGISTERED/"$targ_scan_sequence"_coreg.nii.gz  # copied as T1c_coreg.nii.gz

for f in $ANAT_PATIENT_SPACE/*.nii.gz;do 
	if [ $f != $targ_scan ]; then
		echo "Now coregistering: "$f
		f_basename=`echo "${f##*/}"`
		f_seq=`echo $f_basename | cut -d'_' -f2-` # if f=series801_T1c.nii.gz then this is T1c.nii.gz
		f_seq=${f_seq%%.*} # T1c.nii.gz --> T1c
		# echo $f_seq
		flirt -in $f -ref $targ_scan -out $ANAT_COREGISTERED/"$f_seq"_coreg.nii.gz -dof 6 -cost mutualinfo
	fi
done

python -u $SCRIPT_ROOT/utils/QC_visual.py $ANAT_COREGISTERED

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~ Registration ~~~~~~~~~~~~~~~~~~~~~~~~~~~"
# Registering targ_scan to SRI24
if [ "$REG" = true ] ; then 
	mkdir -p $ANAT_REGISTERED

	if [ $targ_scan_sequence != "T2" ]; then
		SRI_PATH=$ATLAS_PATH/spgr_unstrip.nii.gz
	else
		SRI_PATH=$ATLAS_PATH/late_unstrip.nii.gz
	fi

	echo "Atlas="$SRI_PATH

	echo "Now registering targ_scan=$targ_scan to SRI"

	# # Method-0:
	# flirt -in $targ_scan -ref $SRI_PATH -out $ANAT_REGISTERED/"$targ_scan_sequence"_registered.nii.gz -omat $ANAT_COREGISTERED/patient_to_SRI_omat -cost mutualinfo

	# Method-1:
	flirt -in $targ_scan -ref $SRI_PATH -dof 6 -omat $ANAT_COREGISTERED/patient_to_SRI_dof6_omat -cost mutualinfo
	flirt -in $targ_scan -ref $SRI_PATH -dof 9 -applyxfm -init $ANAT_COREGISTERED/patient_to_SRI_dof6_omat -omat $ANAT_COREGISTERED/patient_to_SRI_dof6_dof9_omat -cost mutualinfo
	flirt -in $targ_scan -ref $SRI_PATH -dof 12 -applyxfm -init $ANAT_COREGISTERED/patient_to_SRI_dof6_dof9_omat -omat $ANAT_COREGISTERED/patient_to_SRI_omat -cost mutualinfo
	flirt -in $targ_scan -ref $SRI_PATH -applyxfm -init $ANAT_COREGISTERED/patient_to_SRI_omat -out $ANAT_REGISTERED/"$targ_scan_sequence"_registered.nii.gz	-cost mutualinfo	

	# # Method-2: we have intermediate images like T1c_on_SRI24_dof*.nii.gz
	# flirt -in $targ_scan                                        -ref $SRI_PATH -out $ANAT_COREGISTERED/patient_on_SRI_dof6.nii.gz  -dof 6 -omat $ANAT_COREGISTERED/patient_to_SRI_dof6_omat
	# flirt -in $ANAT_COREGISTERED/patient_on_SRI_dof6.nii.gz -ref $SRI_PATH -out $ANAT_COREGISTERED/patient_on_SRI_dof9.nii.gz  -dof 9 -omat $ANAT_COREGISTERED/patient_to_SRI_dof9_omat
	# flirt -in $ANAT_COREGISTERED/patient_on_SRI_dof9.nii.gz -ref $SRI_PATH -out $ANAT_COREGISTERED/patient_on_SRI_dof12.nii.gz -dof 12 -omat $ANAT_COREGISTERED/patient_to_SRI_dof12_omat
	# convert_xfm -omat $ANAT_COREGISTERED/patient_to_SRI_dof6_dof9_omat -concat $ANAT_COREGISTERED/patient_to_SRI_dof9_omat $ANAT_COREGISTERED/patient_to_SRI_dof6_omat
	# convert_xfm -omat $ANAT_COREGISTERED/patient_to_SRI_omat -concat $ANAT_COREGISTERED/patient_to_SRI_dof12_omat $ANAT_COREGISTERED/patient_to_SRI_dof6_dof9_omat
	# flirt -in $targ_scan -ref $SRI_PATH -applyxfm -init $ANAT_COREGISTERED/patient_to_SRI_omat -out $ANAT_REGISTERED/"$targ_scan_sequence"_registered.nii.gz
	
	convert_xfm -omat $ANAT_COREGISTERED/SRI_to_patient_omat -inverse $ANAT_COREGISTERED/patient_to_SRI_omat


	for f in $ANAT_COREGISTERED/*.nii.gz;do 
		echo "Now registering: "$f
		f_basename=`echo "${f##*/}"`
		f_seq=`echo $f_basename | cut -d'_' -f1` # if f=T1c_coreg.nii.gz then this is T1c
		if [ $f_seq != $targ_scan_sequence ]; then
			# echo $f_seq
			flirt -in $f -ref $SRI_PATH -applyxfm -init $ANAT_COREGISTERED/patient_to_SRI_omat -out $ANAT_REGISTERED/"$f_seq"_registered.nii.gz -dof 6
		fi
	done

	python -u $SCRIPT_ROOT/utils/QC_visual.py $ANAT_REGISTERED

fi

echo "It took" $SECONDS "seconds"

# explicitly setting permissions to handle XNAT's "failed (upload)" error
cd /output/
chmod -R 755 .
