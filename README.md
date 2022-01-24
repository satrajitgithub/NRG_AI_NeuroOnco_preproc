# Name_to_be_decided
This project aims to offer a completely automated and end-to-end artificial intelligence (AI)-driven workflow for curation and preprocessing of large-scale MRI neuro-oncology studies, with subsequent extraction of quantitative phenotypes. 
![](documentation/figures/pipeline.png)
The workflow i) uses natural language processing and convolutional neural network to classify MRI scans into different anatomical and non-anatomical types; ii) preprocesses the data in a reproducible way; and iii) uses AI to delineate tumor tissue subtypes, enabling extraction of volumetric information along with shape and texture-based radiomic features. Moreover, it is robust to missing MR sequences and adopts an expert-in-the-loop approach, where segmentation results may be manually refined by radiologists. 

Note that, this is Repository 1/2 of a two-part repository. Repository 2/2 can be found at [NRG_AI_NeuroOnco_segment](https://github.com/satrajitgithub/NRG_AI_NeuroOnco_segment)

This project is created by the Neuroinformatics Research Group ([NRG](https://nrg.wustl.edu/)) at the Computational Imaging Research Center (Washington University School of Medicine).


## Citation
## About
Check https://github.com/QTIM-Lab/DeepNeuro

## Usage
The workflow can be used either using [docker](https://www.docker.com/) or within [XNAT](https://www.xnat.org/) which is an extensible open-source imaging informatics software platform dedicated to imaging-based research. Both of these running modes require minimal to no setup. Detailed documentations for both can be found at: 
1. [running with docker](documentation/running_with_docker.md)
2. [running with XNAT](documentation/running_with_XNAT.md)

For a general overview of each step of the workflow, please refer to our [step-by-step walkthrough](documentation/workflow_step_by_step.md).

## Acknowledgements
## Disclaimer
**Only intended for research purposes and not for clinical use.**