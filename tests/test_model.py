import pytest
import torch

from mlsysops_cifar10.model import CIFAR10CNN

@pytest.fixture
def model():
    return CIFAR10CNN()

def test_model_output_shape(model):
    images = torch.randn(4, 3, 32, 32)
    
    outputs = model(images)
    
    assert outputs.shape == (4, 10)
    
def test_single_image(model):
    """Verify CNN accepts a batch containing 1 image"""
    image = torch.randn(1, 3, 32, 32)
    
    output = model(image)
    
    assert output.shape == (1,10)
    
def test_model_parameters(model):
    """Verify CNN has trainable parameters"""
    parameter_cout = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    
    assert parameter_cout == 545098
    
def test_backward_pass(model):
    """Verify that gradients are calculated using backprop"""
    images = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])
    
    outputs = model(images)
    
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    
    loss.backward()
    
    assert model.features[0].weight.grad is not None
    assert model.classifier[1].weight.grad is not None