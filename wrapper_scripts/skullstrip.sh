#!/bin/bash

t1w=`ls /input/ANAT_REGISTERED/T1_registered.nii.gz`
t1c=`ls /input/ANAT_REGISTERED/T1c_registered.nii.gz`
t2=`ls /input/ANAT_REGISTERED/T2_registered.nii.gz`
flair=`ls /input/ANAT_REGISTERED/Flair_registered.nii.gz`
SKULL_STRIPPED=/output

printf "[INFO] Received following files as input:\nT1w = ${t1w}\nT1c = ${t1c}\nT2 = ${t2}\nFlair = ${flair}\n"
printf "[INFO] Saving output at: ${SKULL_STRIPPED}\n"

# Preference to be used as the main scan for producing brainmask.nii.gz - T1c, T1w, T2, Flair
if [ ! -z $t1c ]; then
  	printf "[INFO] Skull-stripping ${t1c} using ROBEX..\n"	 	
	$ROBEX_PATH/runROBEX.sh $t1c $SKULL_STRIPPED/T1c_stripped.nii.gz $SKULL_STRIPPED/brainmask.nii.gz
elif [ ! -z $t1w ]; then
  	printf "[INFO] Skull-stripping ${t1w} using ROBEX..\n"	 	
	$ROBEX_PATH/runROBEX.sh $t1w $SKULL_STRIPPED/T1_stripped.nii.gz $SKULL_STRIPPED/brainmask.nii.gz
elif [ ! -z $t2 ]; then
  	printf "[INFO] Skull-stripping ${t2} using ROBEX..\n"	 	
	$ROBEX_PATH/runROBEX.sh $t2 $SKULL_STRIPPED/T2_stripped.nii.gz $SKULL_STRIPPED/brainmask.nii.gz
elif [ ! -z $flair ]; then
  	printf "[INFO] Skull-stripping ${flair} using ROBEX..\n"	 	
	$ROBEX_PATH/runROBEX.sh $flair $SKULL_STRIPPED/Flair_stripped.nii.gz $SKULL_STRIPPED/brainmask.nii.gz
fi

if [ ! -f $SKULL_STRIPPED/T1_stripped.nii.gz ];then
	printf "[INFO] Using brainmask.nii.gz to strip ${t1w} scan..\n"	 	
	python3 $SCRIPT_ROOT/utils/brainStrip.py ${t1w} $SKULL_STRIPPED/brainmask.nii.gz $SKULL_STRIPPED/T1_stripped.nii.gz
fi

if [ ! -f $SKULL_STRIPPED/T2_stripped.nii.gz ];then
	printf "[INFO] Using brainmask.nii.gz to strip ${t2} scan..\n"	 	
	python3 $SCRIPT_ROOT/utils/brainStrip.py ${t2} $SKULL_STRIPPED/brainmask.nii.gz $SKULL_STRIPPED/T2_stripped.nii.gz
fi

if [ ! -f $SKULL_STRIPPED/Flair_stripped.nii.gz ];then
	printf "[INFO] Using brainmask.nii.gz to strip ${flair} scan..\n"	 	
	python3 $SCRIPT_ROOT/utils/brainStrip.py ${flair} $SKULL_STRIPPED/brainmask.nii.gz $SKULL_STRIPPED/Flair_stripped.nii.gz
fi
 
echo -e "Session Done!"		

python -u $SCRIPT_ROOT/utils/QC_visual.py $SKULL_STRIPPED/

# explicitly setting permissions to handle XNAT's "failed (upload)" error
cd $SKULL_STRIPPED/
chmod -R 755 .
