import scipy.io as sio
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

#creates a dict of subdir paths to store extracted imgs according to type
def mk_output_subdirs(output_parent_folder):
    if output_parent_folder.exists() != True:
        output_parent_folder.mkdir()
        
    car_types = ['suv', 'sedan', 'coupe', 'hatchback', 'van', 'convertible', 'wagon', 'pickup']
    output_subdirs = dict(zip(car_types, [output_parent_folder.joinpath(car_type) for car_type in car_types]))
    
    for subfolder in output_subdirs.values():
        if subfolder.exists() != True:
            subfolder.mkdir()
            
    return output_subdirs

def extract_car_type(name):
    if 'suv' in name:
        return 'suv'
    
    elif 'hatchback' in name:
        return 'hatchback'
    
    elif 'van' in name:
        return 'van'
    
    elif 'convertible' in name:
        return 'convertible'
    
    elif 'coupe' in name:
        return 'coupe'
    
    elif 'cab' in name:
        return 'pickup'
    
    elif 'sedan' in name:
        return 'sedan'
    
    elif 'wagon' in name:
        return 'wagon'

    else:
        return np.nan

#creates a df containing the class names and type
def mk_class_names_df(mat_path):
    mat = sio.loadmat(mat_path, simplify_cells=True)

    class_names = pd.DataFrame(mat['class_names'], columns=['class_name']) #get class names from mat file
    class_names.index += 1
    class_names['class_name'] = class_names['class_name'].str.lower()

    # insert type into names w/o type
    missing_type_names = ['acura tl type-s 2008', 'acura integra type r 2001',
           'buick regal gs 2012', 'chevrolet corvette zr1 2012',
           'chevrolet corvette ron fellows edition z06 2007',
           'chevrolet hhr ss 2010', 'chevrolet cobalt ss 2010',
           'chevrolet trailblazer ss 2009', 'chrysler 300 srt-8 2010',
           'dodge challenger srt8 2011', 'dodge charger srt-8 2009',
           'fiat 500 abarth 2012', 'jaguar xk xkr 2012',
           'lamborghini gallardo lp 570-4 superleggera 2012']
    
    added_type_names = ['acura tl type-s sedan 2008', 'acura integra type-r coupe 2001',
           'buick regal gs sedan 2012', 'chevrolet corvette zr1 coupe 2012',
           'chevrolet corvette ron fellows edition z06 coupe 2007',
           'chevrolet hhr ss suv 2010', 'chevrolet cobalt ss coupe 2010',
           'chevrolet trailblazer ss suv 2009', 'chrysler 300 srt-8 sedan 2010',
           'dodge challenger srt8 coupe 2011', 'dodge charger srt-8 sedan 2009',
           'fiat 500 abarth hatchback 2012', 'jaguar xk xkr coupe 2012',
           'lamborghini gallardo lp 570-4 superleggera coupe 2012']

    #update df
    class_names[class_names.isin(missing_type_names)] = added_type_names

    #extract type from name
    class_names['class_type'] = class_names['class_name'].apply(extract_car_type)
    return class_names

#create a df containing the img paths and x,y coordinates
def mk_annotations_df(mat_path, class_names):
    mat = sio.loadmat(mat_path, simplify_cells=True)
    annotations = pd.DataFrame(mat['annotations']) 
 
    #get type for each img
    annotations['type'] = annotations['class'].apply(lambda img_class: class_names.loc[img_class, 'class_type'])
    
    #convert each path str to path obj
    annotations['relative_im_path'] = annotations.relative_im_path.apply(Path)
    
    #drop no longer needed cols
    annotations.drop(['class','test'], axis = 1, inplace = True)

    return annotations

def img_extract(img_path, xmin, ymin, xmax, ymax, car_type, output_subdirs):
    img = cv2.imread(str(img_path))
    cropped_img = img[ymin:ymax, xmin:xmax]
    
    # save img in respesctive folders
    output_folder = output_subdirs[car_type]
    cv2.imwrite(str(output_folder.joinpath(f'{img_path.name}')), cropped_img)

def main():
    #define paths
    output_parent_folder = Path('./cropped')
    cars_mat = Path('./cars_annos.mat')
    
    #make folders
    output_paths = mk_output_subdirs(output_parent_folder)
    
    #make class_names df and annotations df
    class_names = mk_class_names_df(cars_mat)
    annotations = mk_annotations_df(cars_mat, class_names)
    
    #start
    tqdm.pandas(desc='Categorising')
    annotations.progress_apply(lambda row: img_extract(row[0], row[1], row[2], row[3], row[4], row[5], output_paths)
                               ,axis=1)
if __name__ == '__main__':
    main()