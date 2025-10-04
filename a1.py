import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, default_collate
import torch.optim as optim
from tqdm import tqdm

from torchvision import transforms

import json
import cv2
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

from torchmetrics.classification import BinaryJaccardIndex

class ImageDataset(Dataset):
    def __init__(self, begin, end, device):
        super(ImageDataset, self).__init__()
        self.labeled_data = self.get_labeled_data(begin, end, device)

    def get_file_names(self, begin, end):
        file_names = []
        
        with open("/srv/data/lt2326-h25/a1/info.json", "r") as info_file: 
            meta_data = json.load(info_file)["train"]
            for image_data in meta_data:
                file_name = image_data["file_name"]
                file_path = Path("/srv/data/lt2326-h25/a1/images/" + file_name)
                if file_path.exists():
                    file_names.append(file_name)   
                    
        return file_names[int(len(file_names)*begin):int(len(file_names)*end)]

    def get_images(self, file_names):
        images = {}

        for file_name in file_names:
            file_path = Path("/srv/data/lt2326-h25/a1/images/" + file_name)
            image = cv2.imread(file_path)
            resized_img = cv2.resize(image, dsize=(128, 128))
            images[file_name] = resized_img
            

        return images

    def get_anno_data(self, file_names):
        anno_data = {}
        
        with open("/srv/data/lt2326-h25/a1/train.jsonl") as anno_file:
            full_anno = anno_file.readlines()
            for anno in full_anno: 
                json_obj = json.loads(anno)
                if json_obj["file_name"] in file_names:
                    anno_data[json_obj["file_name"]] = json_obj["annotations"]

        return anno_data

    def get_labeled_data(self, begin, end, device):
        """
        Creates a binary mask for each image based on the annotations. 
        Commented out is the original solution resizing before calculating the mask.
        """
        file_names = self.get_file_names(begin, end)
        images = self.get_images(file_names)
        anno_data = self.get_anno_data(file_names)

        labeled_data = []
        
        for file_name in file_names:
            image = images[file_name]
            labeled_pixels = np.zeros((2048,2048))
            annotations = anno_data[file_name]
            for sentence in annotations:
                for instance in sentence:
                    #xmin = int((instance["adjusted_bbox"][0] * 128) / 2048)       # int so we can use as index for pixels
                    #ymin = 128 - int((instance["adjusted_bbox"][1] * 128) / 2048)  # 128 - ymin because coordinates y start from bottom
                    #xmax = int((xmin + instance["adjusted_bbox"][2] * 128) / 2048)
                    #ymax = 128 - int((ymin + instance["adjusted_bbox"][3] * 128) / 2048)
        #
                    #for x in range(xmin, (xmax if xmax <= 128 else 128) ):
                    #    for y in range(ymin, (ymax if ymax <= 128 else 128)):
                    #        labeled_pixels[x][y] = 1


                    xmin = int(instance["adjusted_bbox"][0])       # int so we can use as index for pixels
                    ymin = 2028 - int(instance["adjusted_bbox"][1]) # 2048 - ymin because coordinates y start from bottom
                    xmax = int(xmin + instance["adjusted_bbox"][2])
                    ymax = 2048 - int(ymin + instance["adjusted_bbox"][3])
        
                    for x in range(xmin, (xmax if xmax <= 2048 else 2048) ):
                        for y in range(ymin, (ymax if ymax <= 2048 else 24048)):
                            labeled_pixels[x][y] = 1
            
            resized_mask = cv2.resize(labeled_pixels, dsize=(128, 128))
            for i in range(len(resized_mask)):
                for j in range(len(resized_mask[0])):
                    if resized_mask[i][j] < 0.5:
                        resized_mask[i][j] = 0
                    else:
                        resized_mask[i][j] = 1
                    
            labeled_data.append((torch.Tensor(image).permute(2, 0, 1).to(device), torch.Tensor(resized_mask).to(device)))

        return labeled_data        

    def __getitem__(self, x):
        return self.labeled_data[x]


    def __len__(self):
        return len(self.labeled_data) 


class SimpleModel(nn.Module):
    """
    Very simple model including one convolutional layer, one pooling, and upsampling.
    """
    
    def __init__(self, ):
        super(SimpleModel, self).__init__()
        self.conv2d = nn.Conv2d(3, 3, (4, 4), stride=2)
        self.p1 = nn.MaxPool2d(4)
        self.tanh = nn.Tanh() # not relu because kinda irrelevant if data not centered?
        self.flatten = nn.Flatten(1, 3)
        self.linear = nn.Linear(6 * 6 * 3, 6 * 6 * 1) 
        self.unflatten = nn.Unflatten(1, (1, 6, 6))   # only 1 channel now
        self.upsample = nn.Upsample(size=(128, 128)) 
        
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, items):
        output = self.conv2d(items)
        output = self.p1(output)
        output = self.conv2d(output)
        output = self.tanh(output)

        output = self.flatten(output)
        output = self.linear(output)
        output = self.unflatten(output)
        
        output = self.upsample(output)
        output = self.sigmoid(output)
        return output

def trainSimple(dataset, batch_size=24, epochs=3, device="cpu"):
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    model = SimpleModel().to(device)
    model.train()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        epoch_loss = 0
        
        for batch_index, batch in enumerate(tqdm(dataloader)):
            X, y = batch          # X size: batch_size * 2048 * 2048 * 3; Y size: batch_size * 2048 * 2048 * 1
            output = model(X)
            loss = criterion(torch.squeeze(output),y.float())

            epoch_loss += loss.item()

        print(f"Epoch loss: {epoch_loss}")

    return model


class UNetModel(nn.Module):
    """
    Model inspired by UNet. I first implemented all the original layers of the UNet architecture, before downsizing.
    Commented out are the layers not used for the sake of memory space preservation.
    """
    def __init__(self):
        super(UNetModel, self).__init__()
        self.conv2d1 = nn.Conv2d(3, 32, (4, 4), stride=1, padding="same")
        self.conv2d32 = nn.Conv2d(32, 32, (4, 4), stride=1, padding="same")
        self.conv2d2 = nn.Conv2d(32, 64, (4, 4), stride=1, padding="same")
        self.conv2d64 = nn.Conv2d(64, 64, (4, 4), stride=1, padding="same")
        
        #self.conv2d3 = nn.Conv2d(64, 128, (4, 4), stride=1, padding="same")
        #self.conv2d128 = nn.Conv2d(128, 128, (4, 4), stride=1, padding="same")
        #self.conv2d4 = nn.Conv2d(128, 256, (4, 4), stride=1, padding="same")
        #self.conv2d256 = nn.Conv2d(256, 256, (4, 4), stride=1, padding="same")
        
        self.conv2d5 = nn.Conv2d(64, 128, (4, 4), stride=1, padding="same")
        #self.conv2d5 = nn.Conv2d(64, 128, (4, 4), stride=1, padding="same")
        self.conv2d128 = nn.Conv2d(128, 128, (4, 4), stride=1, padding="same")
        
        self.relu = nn.ReLU()       
        self.pool = nn.MaxPool2d(2, 2)
        
        #self.upsample1 = nn.ConvTranspose2d(512, 256, (2,2), stride=2)
        #self.upsample2 = nn.ConvTranspose2d(256, 128, (2,2), stride=2)
        self.upsample3 = nn.ConvTranspose2d(128, 64, (2,2), stride=2)
        self.upsample4 = nn.ConvTranspose2d(64, 32, (2,2), stride=2)

        #self.conv2d6 = nn.Conv2d(512, 256, (4, 4), stride=1, padding="same")
        #self.conv2d7 = nn.Conv2d(256, 128, (4, 4), stride=1, padding="same")
        
        self.conv2d8 = nn.Conv2d(128, 64, (4, 4), stride=1, padding="same")
        self.conv2d9 = nn.Conv2d(64, 32, (4, 4), stride=1, padding="same")

        self.conv2dFinal = nn.Conv2d(32, 1, (1, 1), stride=1, padding="same")
        self.sigmoid = nn.Sigmoid()

    def forward(self, item):
        conv1 = self.conv2d1(item)
        conv1 = self.relu(conv1)
        conv1 = self.conv2d32(conv1)
        conv1 = self.relu(conv1)
        pool1 = self.pool(conv1)

        conv2 = self.conv2d2(pool1)
        conv2 = self.relu(conv2)
        conv2 = self.conv2d64(conv2)
        conv2 = self.relu(conv2)
        pool2 = self.pool(conv2)

        #conv3 = self.conv2d3(pool2)
        #conv3 = self.relu(conv3)
        #conv3 = self.conv2d128(conv3)
        #conv3 = self.relu(conv3)
        #pool3 = self.pool(conv3)
#
        #conv4 = self.conv2d4(pool3)
        #conv4 = self.relu(conv4)
        #conv4 = self.conv2d256(conv4)
        #conv4 = self.relu(conv4)
        #pool4 = self.pool(conv4)

        bottom = self.conv2d5(pool2)
        bottom = self.relu(bottom)
        bottom = self.conv2d128(bottom)
        bottom = self.relu(bottom)

        #upconv1 = self.upsample1(bottom)        
        #cat1 = torch.cat((upconv1, conv4), dim=1)
        #conv5 = self.conv2d6(cat1)
        #conv5 = self.relu(conv5)
        #conv5 = self.conv2d256(conv5)
        #conv5 = self.relu(conv5)
#
        #upconv2 = self.upsample2(conv5)
        #cat2 = torch.cat((upconv2, conv3), dim=1)
        #conv6 = self.conv2d7(cat2)
        #conv6 = self.relu(conv6)
        #conv6 = self.conv2d128(conv6)
        #conv6 = self.relu(conv6)

        upconv3 = self.upsample3(bottom)       
        cat3 = torch.cat((upconv3, conv2), dim=1)
        conv7 = self.conv2d8(cat3)
        conv7 = self.relu(conv7)
        conv7 = self.conv2d64(conv7)
        conv7 = self.relu(conv7)

        upconv4 = self.upsample4(conv7)
        cat4 = torch.cat((upconv4, conv1), dim=1)
        conv8 = self.conv2d9(cat4)
        conv8 = self.relu(conv8)
        conv8 = self.conv2d32(conv8)
        conv8 = self.relu(conv8)

        convfinal = self.conv2dFinal(conv8)
        
        output = self.sigmoid(convfinal)

        return output


def trainUNet(dataset, batch_size=24, epochs=3, device="cpu"):
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    model = UNetModel().to(device)
    model.train()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        epoch_loss = 0
        
        for batch_index, batch in enumerate(tqdm(dataloader)):
            X, y = batch          # X size: batch_size * 128 * 128 * 3; Y size: batch_size * 128 * 128 * 1
            output = model(X)
            loss = criterion(torch.squeeze(output),y.float())

            epoch_loss += loss.item()

        print(f"Epoch loss: {epoch_loss}")

    return model


def eval_model(test_dataset, model):
    model.eval()
    testX, testy = zip(*test_dataset.labeled_data)
    testX = torch.stack(list(testX), 0)
    testy = torch.stack(list(testy), 0)

    with torch.no_grad():
        testoutput = model(testX)

    testy = torch.stack(list(testy), 0)
    jaccard = BinaryJaccardIndex()
    jaccard(testoutput.squeeze(), testy)
            
            
    
