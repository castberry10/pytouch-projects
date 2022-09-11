import math
import os
import shutil

for clss in classes_list:
    path = os.path.join(original_dataset_dir, clss)
    fnames = os.listdir(path)
    
    train_size = math.floor(len(fnames) * 0.6)
    validation_size = math.floor(len(fnames) * 0.2)
    test_size = math.floor(len(fnames) * 0.2)
    
    train_fnames = fnames[:train_size]
    print('Train size(', clss , '): ', len(train_fnames))
    