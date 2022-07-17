import os
import torch

print('sudo PYTHONPATH=' + os.path.abspath(torch.__file__).split('/torch')[0])
