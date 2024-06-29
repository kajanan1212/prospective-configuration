from torchvision import datasets
import os

[eval(f"datasets.{dataset}")("./data", download=True) for dataset in ["CIFAR10"]]
