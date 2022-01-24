import glob
import os
import numpy as np
import nibabel as nib
import sys

# This function is for saving the output as a nifti file
def save_numpy_2_nifti(image_numpy, reference_nifti_filepath, output_path):
    nifti_image = nib.load(reference_nifti_filepath)
    new_header = header=nifti_image.header.copy()
    image_affine = nifti_image.affine
    output_nifti = nib.nifti1.Nifti1Image(image_numpy, None, header=new_header)
    nib.save(output_nifti, output_path)


input_path = sys.argv[1]
brainmask_path = sys.argv[2]
output_path=sys.argv[3] 

input_path_numpy = nib.load(input_path).get_fdata()
brainmask_path_numpy = nib.load(brainmask_path).get_fdata()

if len(input_path_numpy.shape) > 3 :
    # 4D volumes could have been given. Often 3Ds are stored as 4Ds with 4th dim == 1.
    assert input_path_numpy.shape[3] == 1
    input_path_numpy = input_path_numpy[:,:,:,0]

output_path_numpy = np.multiply(input_path_numpy, brainmask_path_numpy)

print(f"Saving the output file at: {output_path}")
save_numpy_2_nifti(output_path_numpy, input_path, output_path) # Saving the output file using the function defined above
print("Done!")