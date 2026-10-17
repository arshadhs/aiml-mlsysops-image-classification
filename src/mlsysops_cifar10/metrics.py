import os, torch; 

print('Logical CPUs:', os.cpu_count()); 
print('PyTorch threads:', torch.get_num_threads()); 
print('PyTorch inter-op threads:', torch.get_num_interop_threads())"