# Running using docker
### 1. Installation and Requirements
#### 1.1. Required software
This requires an installation of docker. Instructions can be found at https://docs.docker.com/get-docker/ 
Test installation by running this:
```
docker
```
#### Latest versions tested:  
As of Jan 2022, this software has been tested on CentOS 7.9.2009 using Docker version 20.10.7, build f0df350.

#### 1.2. Pulling docker containers
The two required containers can be pulled using:
```
docker pull satrajit2012/scanclsfrmeta:v3
docker pull satrajit2012/segmenter:v3
```
#### 1.3. Creating input folder structure
The containers operate on a session-level. Arrange folder structure as follows:
```
sample_subject/
└── sample_session
    ├── scan1
    │   ├── 1-01.dcm
    │   ├── 1-02.dcm
    │   ├── 1-03.dcm
    │   └── ...
    ├── scan2
    │   ├── 1-01.dcm
    │   ├── 1-02.dcm
    │   ├── 1-03.dcm
    │   ├── ...
    └── scan3
        ├── 1-01.dcm
        ├── 1-02.dcm
        ├── 1-03.dcm
        ├── ...

```
#### 1.4. Removing space from folder names
To prevent the code from breaking on linux systems, it is preferred to remove all spaces from file and foldernames.
For that, run the following script to replace all spaces with underscore :
```
for f in *\ *; do mv "$f" "${f// /_}"; done
```
For example, if you had data of a subject as follows:
```
subject1/
└── session1
    ├── 1.000000-3 PLANE LOC-58068
    ├── 2.000000-ASSET cal-86691
    ├── 3.000000-SAG T1 FLAIR-91896
    ...
```
after running the script, it should change to:
```
subject1/
└── session1
    ├── 1.000000-3_PLANE_LOC-58068
    ├── 2.000000-ASSET_cal-86691
    ├── 3.000000-SAG_T1_FLAIR-91896
    ...
```
#### 1.5. Setting input and output paths
Set the rootpath to the session as the `input_path`:
```
input_path=/data/xnat/foo/subject1/session1
```
Set and create the directories that you want to store your outputs in:
```
output_path=/data/xnat/foo/resources
mkdir -p $output_path/{SCANTYPES,REGISTERED,SKULL_STRIPPED,SEGMENTATION,SEGMENTATION_DCMSEG}
```
Now we are ready to run the dockers.
### 2. Running the containers
We have x commands that we need to execute **sequentially** on the session.
#### 2.1. Scan-type classifier
In this step we take as input the raw dicom files from the session and classify them.
```
docker run --rm \
-v $input_path:/home/scans/SCANS/ \
-v $output_path/SCANTYPES:/home/output \
scanclsfrmeta:v3 /bin/bash -c 'cd /NRG_AI_Neuroonco_preproc/; ./run_local.sh'
```
#### 2.2. Registration
```
docker run 
-v $output_path/SCANTYPES:/home/input \
-v $output_path/REGISTERED:/home/output \
scanclsfrmeta:v3 /bin/bash -c '/NRG_AI_Neuroonco_preproc/flirt_wrapper/register_to_SRI24_local.sh -r'
```
#### 2.3. Skull-stripping
```
docker run 
-v $output_path/REGISTERED/:/home/input/ \
-v $output_path/SKULL_STRIPPED:/home/output \
scanclsfrmeta:v3 /bin/bash -c '/NRG_AI_Neuroonco_preproc/robex_wrapper/script_robex_local.sh'
```
#### 2.4. Segmentation
```
docker run 
-v $output_path/SKULL_STRIPPED/:/workspace/data \
-v $output_path/SEGMENTATION:/output \
segmenter:v3 /bin/bash -c '/run_local.sh session1'
```
#### 2.5. Segmentation to patient space
```
docker run --rm \
-v $output_path/SCANTYPES/:/scantypes \
-v $output_path/REGISTERED/:/registered \
-v $output_path/SEGMENTATION/:/segmentation \
scanclsfrmeta:v3 /bin/bash -c '/NRG_AI_Neuroonco_preproc/flirt_wrapper/prediction2patientspace_local.sh'
```
#### 2.6. Patient space to dicom-seg
```
docker run --rm \
-v $output_path/SEGMENTATION_DCMSEG/:/resource_structscan \
-v $output_path/SEGMENTATION/:/resource_segmentation \
-v $output_path/SEGMENTATION_DCMSEG/:/output \
scanclsfrmeta:v3 /bin/bash -c '/NRG_AI_Neuroonco_preproc/dcmqi_wrapper/run_dcmqi_local.sh'
```