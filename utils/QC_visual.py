import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import os 
import sys
import glob
from pathlib import Path
import shutil
import yaml

def show_axial_sagittal_coronal_allmods(filepath, filename, suptitle = None):
    
    n_cols = 3 # three views axial, sag, coronal
    n_rows = len(glob.glob(filepath)) # as many as the num of modalities that session has
    size_of_each_fig = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*size_of_each_fig,n_rows*size_of_each_fig))


    for idx, modality in enumerate(glob.glob(filepath)):

        data = np.squeeze(nib.load(modality).get_fdata())

        num_slices_sag = data.shape[0]
        num_slices_cor = data.shape[1]
        num_slices_ax = data.shape[2]

        slices = [data[:,:,num_slices_ax//2], data[num_slices_sag//2,:,:], data[:,num_slices_cor//2,:]]

        #  If there is only one modality available
        if n_rows == 1:
            ax_idx = "[i]"
            title_idx = "[0]"
        else:
            ax_idx = "[idx, i]"
            title_idx = "[idx, 0]"

        for i, slice in enumerate(slices):
            eval("axes"+ax_idx).imshow(slice.T, cmap="gray", origin="lower")
            eval("axes"+ax_idx).axis('off')
            eval("axes"+title_idx).set_title(modality.split(os.sep)[-1], fontsize = 15, color='r')
    
    if suptitle:
        fig.suptitle(suptitle, fontsize = 20)
    
    plt.savefig(filename, bbox_inches = 'tight')
    plt.close()


target_folder = os.path.abspath(sys.argv[1])
print(f"[QC] target_folder = {target_folder}")

show_axial_sagittal_coronal_allmods(os.path.join(target_folder, '*.nii.gz'), os.path.join(target_folder, 'QC.png'))


