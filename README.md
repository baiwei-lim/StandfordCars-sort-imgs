# Purpose
This script only purpose is to categorise the images according to their type provided in the dataset. All images dataset was used instead of the split train/test dataset.

## Instructions
1. Place categorise.py in a folder with the extracted dataset folder and mat file
```
.
├── car_ims
├── cars_annos.mat
├── categorise.py
```

2. Install necessary packages (use pip if not sure)
    * conda: ``` conda install -c conda-forge scipy pandas numpy opencv tqdm ```
    * pip: ``` pip install scipy pandas numpy opencv-python tqdm ```

3. Open a terminal/cmd in the folder containing the files and run the following command
``` 
python categorise.py 
```

4. A new folder named cropped should appear and contain all the categorised images
```
.
├── car_ims
├── cars_annos.mat
├── categorise.py
├── cropped
```
