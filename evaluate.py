import torch
import torchvision.transforms as transforms

import cv2
import numpy as np
import matplotlib.pyplot as plt


from Perception.utils.utils import find_edge_channel


def evaluate(model,image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges,edges_inv = find_edge_channel(image)
    output_image = np.zeros((gray.shape[0],gray.shape[1],3),dtype=np.uint8)
    output_image[:,:,0] = gray
    output_image[:,:,1] = edges
    output_image[:,:,2] = edges_inv
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((180,330)),
                                    transforms.ToTensor()])
    
    test_img = transform(output_image).unsqueeze(0)
    output = model(test_img)
    pred = torch.sigmoid(output)
    pred = torch.where(pred > 0.5 , 1, 0)
    pred = pred.detach().cpu().squeeze().numpy()

    return pred




