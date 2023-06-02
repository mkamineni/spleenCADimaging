import SimpleITK as sitk
import argparse
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import pandas as pd
import pickle
import multiprocessing as mp
import os
from collections import ChainMap

# for the extraction of features using radiomics
import six
from radiomics import featureextractor, getTestCase
import csv

# go through outputs folder and then go through stitched_data folder
stitched_dir = '/home/jupyter/MRI-spleen/stitched_data'
nnunet_folder = '/home/jupyter/MRI-spleen/nnunet_data'
#stitched_exts = ['opp.nii.gz', 'fat.nii.gz', 'inp.nii.gz', 'wat.nii.gz']
stitched_exts = ['opp.nii.gz','wat.nii.gz']

directory = '/home/jupyter/MRI-spleen/outputs'
segments = '/home/jupyter/MRI-spleen/segments'
        
def load_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        pkl_file = pickle.load(f)
    return pkl_file


def array_from_image(filename):
    # Read the .nii image containing the volume with SimpleITK:
    sitk_t1 = sitk.ReadImage(filename)

    # and access the numpy array:
    t1 = sitk.GetArrayFromImage(sitk_t1)
    return t1

def segment_one_org(array, org_label = 5):
    def map_label(i, org_label = org_label):
        if i == org_label:
            return 1
        else:
            return 0
        

    vec_map = np.vectorize(map_label)
    updatedArray = vec_map(array)
    
    print("Number of pixels for label %d: %d" %(org_label, np.sum(updatedArray)))
    return updatedArray


def get_orig_image(stitched_dir, subject_id, i, stitched_ext, conversion_map):
    stitched_vol_fn = '/'.join([stitched_dir, subject_id, stitched_ext])
    #print(stitched_vol_fn)
    if stitched_vol_fn in conversion_map:
        nnunet_image = conversion_map[stitched_vol_fn]
        orig_image = array_from_image(nnunet_image)
        orig_image = np.array(orig_image)
        return orig_image             
    else:
        print(subject_id + "is missing")
        return None

    
def extract_voxel_helper(subject_id, counter, conversion_map):
    print("Subject %s" %subject_id)
    pred_file = directory+'/'+subject_id+'/prd.nii.gz'
    segment_map = array_from_image(pred_file)
    segment_map = np.array(segment_map)

    dict1, dict2 = {}, {}
    for i, stitched_ext in enumerate(stitched_exts):
        try:
            orig_image = get_orig_image(stitched_dir, subject_id, i, stitched_ext, conversion_map)
            if orig_image is None:
                continue
            # get both liver and spleen
            for org_label in [1]:
                os.makedirs(os.path.join(segments, subject_id), exist_ok=True)
                segment_org = segment_one_org(segment_map, org_label)
                #print(segment_org.shape)
                segment_org_nifti = nib.Nifti1Image(segment_org, affine=np.eye(4), dtype = float)
                segment_org_path = os.path.join(segments, subject_id, str(org_label)+'_seg_org_'+stitched_ext)
                nib.save(segment_org_nifti, segment_org_path)

                desired_voxel = np.multiply(segment_org, orig_image)     
                #print(desired_voxel.shape)

                final_img = nib.Nifti1Image(desired_voxel, affine=np.eye(4))
                final_img_path = os.path.join(segments, subject_id, str(org_label)+'_'+stitched_ext)
                nib.save(final_img, final_img_path)
                #os.remove(segment_org_path)
                #os.remove(final_img_path)

                if org_label == 1:
                     dict1[str(counter)+stitched_ext] = [final_img_path, segment_org_path, subject_id]
                else:
                     dict2[str(counter)+stitched_ext] = [final_img_path, segment_org_path, subject_id]

        except:
            print(subject_id)
    return dict1, dict2
            
            
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--increment', required=True, help='Folder that contains downloaded zip files for each subject')
    parser.add_argument('--counter', required=True, help='Folder that contains subjects with stitched volumes as .nii.gz files')
    
    args = parser.parse_args()

    increment = int(args.increment)
    counter = int(args.counter)

    orig_conversion_map = load_pickle(os.path.join(nnunet_folder, 'conversion.pkl'))
    conversion_map = {}
    for entry in orig_conversion_map:
        entry = entry[0]    
        for img_path in entry['img_paths']:
            conversion_map[img_path['orig']] = img_path['nnunet'] 

    args = []
    
    dirs = [x[0] for x in os.walk(segments)]
    
    if increment*counter > len(dirs):
        if increment*counter < len(dirs):
            dirs = dirs[increment*counter: -1]
        else:
            dirs = 0
    else:
        dirs = dirs[increment*(counter-1):increment*counter]

    for subject_id in dirs:
        subject_id = subject_id.split('/')[-1]
        if len(subject_id) == 7:
            args.append((subject_id, counter, conversion_map))
            counter +=1

    pool =  mp.Pool(mp.cpu_count())
    res = pool.starmap(extract_voxel_helper, args)
    dict1_list, dict2_list = zip(*res)
    dict1 = dict(ChainMap(*list(dict1_list)))
    dict2 = dict(ChainMap(*list(dict2_list)))

    df1 = pd.DataFrame.from_dict(dict1,orient='index', columns = ["Image", "Mask", "ID"])
    df2 = pd.DataFrame.from_dict(dict2,orient='index', columns = ["Image", "Mask", "ID"])
    df1.to_csv("radiomics/radiomics_filenames_1.csv")
    df2.to_csv("radiomics/radiomics_filenames_2.csv")

     




if __name__ == '__main__':
    main()
