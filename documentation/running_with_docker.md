# Running using docker
<!-- TOC -->
- [1. Installation and Requirements](#1-installation-and-requirements)
  - [1.1. Required software](#11-required-software)
    - [Latest versions tested](#latest-versions-tested)
  - [1.2. Pulling docker containers](#12-pulling-docker-containers)
  - [1.3. Creating input folder structure](#13-creating-input-folder-structure)
  - [1.4. Removing space from folder names](#14-removing-space-from-folder-names)
  - [1.5. Setting input and output paths](#15-setting-input-and-output-paths)
- [2. Running the containers](#2-running-the-containers)
  - [2.1. Scan-type classifier](#21-scan-type-classifier)
  - [2.2. Registration](#22-registration)
  - [2.3. Skull-stripping](#23-skull-stripping)
  - [2.4. Segmentation](#24-segmentation)
  - [2.5. Segmentation to patient space](#25-segmentation-to-patient-space)
  - [2.6. Patient space to dicom-seg](#26-patient-space-to-dicom-seg)

<!-- /TOC -->
### 1. Installation and Requirements
#### 1.1. Required software
This requires an installation of docker. Instructions can be found at https://docs.docker.com/get-docker/
Test installation by running this:
```
docker
```
##### Latest versions tested  
As of Jan 2022, this software has been tested on CentOS 7.9.2009 using Docker version 20.10.7, build f0df350.

#### 1.2. Pulling docker containers
The two required containers can be pulled using:
* `satrajit2012/nrg_ai_neuroonco_preproc:v0`
* `satrajit2012/nrg_ai_neuroonco_segment:v0`

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
The code assumes that no directory or filenames will have spaces in them, so please remove all spaces from file and folder names. An example script which does this (replaces all spaces with underscore in all directory and filenames recursively):
```
find . -depth -name '* *' | while IFS= read -r f ; do mv -i "$f" "$(dirname "$f")/$(basename "$f"|tr ' ' _)" ; done
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
Set the rootpath to the session (not the subject) as the `input_path`:
```
input_path=/absolute/path/to/subject1/session1
```
Set and create the directories that you want to store your outputs in:
```
output_path=/absolute/path/to/output
mkdir -p $output_path/{1-scan_type_classifier,2-registration,3-skullstripped,4-segmentation,5-segmentation_patient_space,6-patient_space_to_dicom_seg}
```
Now we are ready to run the dockers.
### 2. Running the containers
We have 6 commands that we need to execute **sequentially** on the session.
#### 2.1. Scan-type classifier
In this step we take as input the raw dicom files from the session and classify them.
```
docker run --rm -v $input_path:/input -v $output_path:/output_root -v $output_path/1-scan_type_classifier:/output satrajit2012/nrg_ai_neuroonco_preproc:v0 scan_type_classifier --docker
```
#### 2.2. Registration
```
docker run --rm -v $output_path/1-scan_type_classifier:/input -v $output_path:/output_root -v $output_path/2-registration:/output satrajit2012/nrg_ai_neuroonco_preproc:v0 registration --docker
```
#### 2.3. Skull-stripping
```
docker run --rm -v $output_path/2-registration:/input -v $output_path:/output_root -v $output_path/3-skullstripped:/output satrajit2012/nrg_ai_neuroonco_preproc:v0 skullstrip --docker
```
#### 2.4. Segmentation
```
docker run --rm -v $output_path/3-skullstripped:/input -v $output_path:/output_root -v $output_path/4-segmentation:/output satrajit2012/nrg_ai_neuroonco_segment:v0 segmentation --docker [--evaluate] [--radiomics]
```
#### 2.5. Segmentation to patient space
```
docker run --rm -v $output_path/1-scan_type_classifier:/input_scantype -v $output_path/2-registration:/input_registration -v $output_path/4-segmentation:/input_segmentation -v $output_path:/output_root -v $output_path/5-segmentation_patient_space:/output satrajit2012/nrg_ai_neuroonco_preproc:v0 prediction_to_patient_space --docker
```
#### 2.6. Patient space to dicom-seg
```
docker run --rm -v $input_path:/input_session -v $output_path/1-scan_type_classifier:/input_scantype -v $output_path/2-registration:/input_registration -v $output_path/5-segmentation_patient_space:/input_segmentation_patient -v $output_path:/output_root -v $output_path/6-patient_space_to_dicom_seg:/output satrajit2012/nrg_ai_neuroonco_preproc:v0 patient_space_to_dcmseg --docker
```
Note that, the step [7. Uploading dicom-seg segmentation as ROI-assessor](workflow_step_by_step.md/#7-uploading-dicom-seg-segmentation-as-roi-assessor) is only for XNAT and not performed for general docker usage.
