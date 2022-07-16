#!/bin/bash

banner()
{
  echo "+------------------------------------------+"
  printf "| %-40s |\n" "`date`"
  echo "|                                          |"
  printf "|`tput bold` %-40s `tput sgr0`|\n" "$@"
  echo "+------------------------------------------+"
}


xnat_jsess() {
  jsess=`curl -u $XNAT_USER:$XNAT_PASS -X GET $XNAT_HOST/data/JSESSION`
  echo $jsess
}


upload_scantypes() {
  while read -r line;do
    # echo $line;
    series_number=$(echo $line|cut -d':' -f1)
    predicted_type=$(echo $line|cut -d':' -f2)
    echo curl -k --cookie JSESSIONID=$2 -X PUT $XNAT_HOST'/data/projects/'$3'/subjects/'$4'/experiments/'$5'/scans/'$series_number'?xnat:mrScanData/series_class='$predicted_type 
    curl -k --cookie JSESSIONID=$2 -X PUT $XNAT_HOST'/data/projects/'$3'/subjects/'$4'/experiments/'$5'/scans/'$series_number'?xnat:mrScanData/series_class='$predicted_type
    # break
  done < $1
}


set_custom_flag_xnat() {
    echo curl -k --cookie JSESSIONID=$2 -X PUT $XNAT_HOST'/data/projects/'$3'/subjects/'$4'/experiments/'$5'?xnat:experimentData/fields/field%5Bname%3Dsegmentationready%5D/field='$1
    curl -k --cookie JSESSIONID=$2 -X PUT $XNAT_HOST'/data/projects/'$3'/subjects/'$4'/experiments/'$5'?xnat:experimentData/fields/field%5Bname%3Dsegmentationready%5D/field='$1
}

get_custom_flag_xnat () {
  # parse segmentationready flag with python
  # echo python -u $SCRIPT_ROOT/utils/parse_XNAT_custom_flag.py $2 $3 $4 $1
  python -u $SCRIPT_ROOT/utils/parse_XNAT_custom_flag.py $2 $3 $4 $1

  if [ $? -eq 0 ]; then
      echo "true"
  else
      echo "false"
  fi
}


set_custom_flag() {
    echo $1'='$2 >> /output_root/custom_flag.txt
}

get_custom_flag() {
    source /output_root/custom_flag.txt
    echo $segmentationready
}