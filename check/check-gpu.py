import torchvision
import torch

print('CUDA.is available: ', torch.cuda.is_available())
print('N of GPU devices: ', torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_properties(i).name)
