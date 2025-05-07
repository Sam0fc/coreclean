import pandas as pd
from . import file_utils
import matplotlib.pyplot as plt
import cv2

def get_core_sections(tops = './chronology/tops.csv', bottoms = './chronology/bottoms.csv', path='./Dataset/Cropped/', composite_path='./Composite', leg=339):
    '''selects and crops cores from a list of the tops and bottoms of splice sections'''
    tops = pd.read_csv(tops)
    bottoms = pd.read_csv(bottoms)

    cores=[]
    for i in range(len(tops)):
        top = tops.iloc[i]
        bottom = bottoms.iloc[i]
        print(bottom)
        needed_sections = range(top['Section'], bottom['Section'] + 1)
        print(f'we need, {needed_sections}')
        top_of_top = top['TopOffset (cm)']
        bottom_of_bottom = bottom['BottomOffset (cm)']
        print(f'we need, {top_of_top} to {bottom_of_bottom}')

        full_needed = []
        for section in needed_sections:
            full_needed.append( str(leg) + '_' + top['site'] + top['hole'] + '_' + f"{int(top['core']):02d}" + top['Type'] + '_' + str(section) )
        print(full_needed)

        for index, core in enumerate(full_needed):
            cores.append(core)
            if index == 0:
               file_utils.save_image(get_core(core, top=top_of_top, path = path), path = composite_path + '/' + str(i) +'-' +  str(index) +'-' +core + '.tif')
            elif index == len(full_needed) - 1:
               file_utils.save_image(get_core(core, bottom=bottom_of_bottom, path = path), path = composite_path + '/' + str(i) +'-' +str(index) +'-'+  core + '.tif')
            else:
               file_utils.save_image(get_core(core, path = path), path = composite_path +'/' + str(i) +'-' + str(index) +'-' +core + '.tif')
    
    combine_cores(cores, path = composite_path)
    

def get_core(core, top=None, bottom=None, path = './Dataset/Cropped/'):
    '''gets a core from the path and crops it to the top and bottom'''
    print(core)
    core = file_utils.read_image(path + core + '.tif')[0]

    print(core.shape)
    if core.shape[0] > core.shape[1]:
        # rotate clockwise
        core = cv2.rotate(core, cv2.ROTATE_90_CLOCKWISE)
        print(core.shape)
    
    if top is not None:
        top = int(200*top)
        core = core[:,top:,:]
    if bottom is not None:
        bottom = int(200*bottom)
        core = core[:,:bottom,:]


    
    #top = top * 200
    #bottom = bottom * 200

    #if top is not None:
    #    core = core[top:]
    #if bottom is not None:
    #    core = core[:bottom]
    
    return core

def combine_cores(cores,composite_path = './Composite'):
        # Find the minimum height (shape[0]) among all cores
        min_height = min(core.shape[0] for core in cores)

        # Crop all cores to the minimum height, taking from the middle of larger sections
        cropped_cores = []
        for core in cores:
            if core.shape[0] > min_height:
                start = (core.shape[0] - min_height) // 2
                end = start + min_height
                cropped_cores.append(core[start:end, :, :])
            else:
                cropped_cores.append(core)

        # Concatenate cores along the width (shape[1])
        combined_core = cv2.hconcat(cropped_cores)

        # Display the combined core
        plt.imshow(combined_core)
        plt.show()

        return combined_core

if __name__ == "__main__":
    get_core_sections()