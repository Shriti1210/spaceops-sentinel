import torch
import numpy as np
import cv2
from deep_change_model import ChangeCNN

DEVICE = torch.device("cpu")


def load_deep_model():

    model = ChangeCNN()
    model.load_state_dict(
        torch.load("models/deep_change_cnn.pth", map_location=DEVICE)
    )
    model.eval()

    return model


def get_deep_change_heatmap(img1, img2):

    model = load_deep_model()

    img1 = img1[:, :, :3]   # FORCE RGB
    img2 = img2[:, :, :3]

    img1_res = cv2.resize(img1, (256,256))
    img2_res = cv2.resize(img2, (256,256))

    img1_res = img1_res.transpose(2,0,1) / 255.0
    img2_res = img2_res.transpose(2,0,1) / 255.0

    x = np.concatenate([img1_res, img2_res], axis=0)

    tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred = model(tensor)

    heatmap = pred.squeeze().numpy()

    heatmap = cv2.resize(
        heatmap,
        (img1.shape[1], img1.shape[0])
    )

    return heatmap