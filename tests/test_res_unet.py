import torch
import sys
# sys.path.append("D:\_Work\_Research\ResUnet") 
sys.path.append("/home/grad/Shilong/ResUnet_for_Estimation/") 

from core.res_unet import ResUnet, ResidualConv, Upsample
from core.res_unet_plus import ResUnetPlusPlus

def test_resunet():
    img = torch.ones(1, 3, 224, 224)
    resunet = ResUnet(3)
    assert resunet(img).shape == torch.Size([1, 1, 224, 224]) 
    
    
def test_residual_conv():
    x = torch.ones(1, 64, 224, 224)
    res_conv = ResidualConv(64, 128, 2, 1) 
    assert res_conv(x).shape == torch.Size([1, 128, 112, 112]) 
    

def test_upsample():
    x = torch.ones(1, 512, 28, 28)
    upsample = Upsample(512, 512, 2, 2)
    assert upsample(x).shape == torch.Size([1, 512, 56, 56])

def test_resunet_mine(): # 测试能否输入热图大小[672,224]，然后得到[672,224]的mask
    img = torch.ones(1, 1, 672, 224)
    resunet = ResUnet(1)
    if resunet(img).shape == torch.Size([1, 1, 672, 224]):
        print("yes")
    else:
        print("no")

def test_resunet_plus_mine(): # 测试能否输入热图大小[672,224]，然后得到[672,224]的mask
    img = torch.ones(1, 1, 672, 224)
    resunet_Plus = ResUnetPlusPlus(1)
    output = resunet_Plus(img)
    if output.shape == torch.Size([1, 1, 672, 224]):
        print("yes")
    else:
        print("no")
        print(output.shape)

test_resunet_plus_mine()
    