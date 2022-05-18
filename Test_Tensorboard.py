from tensorboardX import SummaryWriter
writer = SummaryWriter("./Test_TensorboardLog")
from dataset import dataloader
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils.logger import MyWriter
import os
import torch
from core.res_unet import ResUnet
from core.res_unet_plus import ResUnetPlusPlus
from utils import metrics




dataset_val = dataloader.ImageDataset_mine(
        False, transform=transforms.Compose([dataloader.ToTensorTarget()])
    )
val_dataloader = DataLoader(
        dataset_val, batch_size=3, num_workers=0, shuffle=False
    )

model = ResUnet(1).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    
writer = MyWriter("./logs/Test_Tensorboard")
criterion = metrics.BCEDiceLoss()

resume = "/home/grad/Shilong/ResUnet_for_Estimation/checkpoints/test_train/test_train_checkpoint_3370.pt"
if resume:
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)

        start_epoch = checkpoint["epoch"]

        best_loss = checkpoint["best_loss"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(
            "=> loaded checkpoint '{}' (epoch {})".format(
                resume, checkpoint["epoch"]
            )
        )
    else:
        print("=> no checkpoint found")
print("we arrive here1")
for idx, data in enumerate(tqdm(val_dataloader, desc="validation")):
    # get the inputs and wrap in Variable
    inputs = data["sat_img"].cuda()
    labels = data["map_img"].cuda()
    if idx == 0:print("we arrive here2")
    outputs = model(inputs)
    loss = criterion(outputs, labels)
        
    if idx == 0:
        writer.log_images(inputs.cpu(), labels.cpu(), outputs.cpu(), 1)
        

# from tensorboard import version; 
# print(version.VERSION)

# from tqdm import tqdm
# import time

# bar = tqdm(range(1000))
# bar.set_description("now < 500")

# for i in bar:
#     time.sleep(0.01)
#     if i == 501:
#         bar.set_description("now > 500")


# import tensorflow as tf

# print(tf.__version__)
