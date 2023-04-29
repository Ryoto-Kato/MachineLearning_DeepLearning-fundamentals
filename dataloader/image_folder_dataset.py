"""
Definition of ImageFolderDataset dataset class
"""

import os
import pickle

import numpy as np
from PIL import Image

from .base_dataset import Dataset

class rk_ImageFolderDataset(Dataset):
    """Data archive extraction and folder saving
    
    Args:
    self.images:    list
                    file path to images

    self.labels:    list, int(class index)
                    list of image class index [0, 9]
    
    """
    def __init__(self, root, *args, transform = None,
                  download_url = 'https://i2dl.vc.in.tum.de/static/data/cifar10.zip',
                  **kwargs):
        super().__init__(root, *args, download_url, **kwargs)

        self.classes, self.class_to_idx = self._find_classes(self.root_path)
        self.images, self.labels = self.make_dataset(
            directory = self.root_path,
            class_to_idx = self.class_to_idx
        )

        self.transform = transform

    @staticmethod
    def _find_classes(directory):
        """Finds the class folders in a dataset

        Args:
        -----
        directory: root.directory of the dataset
        
        Returns:
        -------
        classes:        list
                        contains all classes found
        class_to_idx:   dictionary
                        maps class to label (class index, class label[0, 9])
        """

        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))} # dictionary
        return classes, class_to_idx
    
    @staticmethod
    def make_dataset(directory, class_to_idx):
        """Create the image dataset by preparing a list of samples
        Images are sorted in an ascending order by class and file name

        Args:
        ----
        dictionary:     string (root_path)
                        root directory of the dataset which is downloaded and stored

        class_to_idx:   dictionary
                        maps class label to index
        
        Returns:
        -------
        images:         list
                        all images in the dataset
        
        labels:         list
                        one label per image
        """

        images, labels = [], []

        for target_class in sorted(class_to_idx.keys()): # sort in alphabet orders
            label = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class) # directory to the class image folder

            for root, _, fnames in sorted(os.walk(target_dir)): # return root, directory, files
                for fname in sorted(fnames):
                    if fname.endswith(".png"):
                        path = os.path.join(root, fname)
                        images.append(path)
                        labels.append(label)

        assert len(images) == len(labels)
        return images, labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        """Return dictionary

        Args:
        index:  int
                image index

        Returns:
        --------
        data_dict:  dictionary
                    ["image": .png, "label": (int) class index]
        """
        _image_path = self.images[index]
        _label = self.labels[index]

        _image = self.load_image_as_numpy(_image_path)

        if self.transform is not None:
            _image = self.transform(_image)
        data_dict = {"image": _image, "label": _label}

        return data_dict
    
    def damp_pckl(self, save_root, pickel_fname):
        if not save_root.endswith("memory"):
            save_root+='/memory'

        if not os.path.exists(save_root):
            os.makedirs(save_root, exist_ok=True)

        pickel_file_path = os.path.join(save_root, pickel_fname)


        if not os.path.exists(pickel_file_path):
            f = open(pickel_file_path, 'x')
            f.close()

        with open(pickel_file_path, 'wb') as f:
            pickle.dump({"images": self.images, "labels": self.labels, 
                         "class_to_idx": self.class_to_idx, "classes": self.classes}, f)
        f.close()
        
    
    @staticmethod # this method can be called without instantiation/object of this class
    def load_image_as_numpy(image_path):
        """Load image as numpy
        Args:
        ----
        image_path: path to .png
                    self.images[index]

        Returns:
        -------
        array_im;   np.array
                    np.array of image.png
        """

        array_im = np.asarray(Image.open(image_path),dtype=float)

        return array_im

class rk_MemoryImageFolderDataset(rk_ImageFolderDataset):
    def __init__(self, root, pckl_fname, *args,
                 transform=None,
                 download_url="https://i2dl.vc.in.tum.de/static/data/cifar10memory.zip",
                 **kwargs):
        
        super().__init__(
            root, *args, download_url=download_url, **kwargs)
        
        with open(os.path.join(self.root_path, pckl_fname), 'rb') as f:
            save_dict = pickle.load(f)
        
        self.images = save_dict['images']
        self.labels = save_dict['labels']
        self.class_to_idx = save_dict['class_to_idx']
        self.classes = save_dict['classes']
        self.transform = transform

    # def load_image_as_numpy(self, image_path):
    #     """Here we already have everything in memory, 
    #     so we can just return the image        
    #     """
    #     return image_path