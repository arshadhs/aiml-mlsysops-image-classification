# validate the datasets

import pytest
import torch
from mlsysops_cifar10.data import load_cifar10

@pytest.fixture(scope="module")
def datasets():
    return load_cifar10()
  
def test_dataset_size(datasets):
    train, test = datasets
    assert len(train) == 50000
    assert len(test) == 10000

def test_image_shape(datasets):
    train, _ = datasets
    image, label = train[0]
    
    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert 0 <= label < 10

# train, test = load_cifar10(); 
# print('Training:', len(train)); 
# print('Testing:', len(test)); 
# print('Image shape:', train[0][0].shape); 
# print('Classes:', train.classes)

def test_pixel_values(datasets):
    train, _ = datasets
    image, _ = train[0]
    
    assert image.min() >= 0
    assert image.max() <= 1
    
def test_classes(datasets):
    train, _ = datasets
    
    assert len(train.classes) == 10