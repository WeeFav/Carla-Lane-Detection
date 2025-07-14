import torch
import os
import cv2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import scipy.special
from tqdm import tqdm
import numpy as np
import yaml

from model import UFLDNet
from dataset import DemoDataset
from utils import culane_row_anchor, tusimple_row_anchor

def main(cfg):
    # number of row anchors
    if cfg['dataset'] == 'CULane':
        cls_num_per_lane = 18
        img_w, img_h = 1640, 590
        row_anchor = culane_row_anchor
    elif cfg['dataset'] == 'Tusimple':
        cls_num_per_lane = 56
        img_w, img_h = 1280, 720
        row_anchor = tusimple_row_anchor
    else:
        raise NotImplementedError

    # define model
    # we dont need auxiliary segmentation in testing
    model = UFLDNet(
        pretrained=False,
        backbone=cfg['backbone'],
        cls_dim=(cfg['griding_num'] + 1, cls_num_per_lane, 4),
        use_aux=False
    )
    model.cuda()

    # load model weights
    # If your model was trained with torch.nn.DataParallel or DistributedDataParallel, the state_dict keys are prefixed with 'module.'
    # If you’re now loading into a single-GPU or CPU model, the keys need to match exactly. So this loop removes the 'module.' prefix from the keys.
    state_dict = torch.load(cfg['model_path'], map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v
            
    # strict=False means won’t complain if some layers are missing or if there are extra layers. Useful for fine-tuning 
    model.load_state_dict(compatible_state_dict, strict=False)
    model.eval()

    # transform input image
    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    dataset = DemoDataset(cfg['img_folder'], img_transform=img_transforms)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    for batch_idx, (imgs, img_paths) in enumerate(tqdm(dataloader)):
        imgs = imgs.cuda()
        # predict
        with torch.no_grad():
            out = model(imgs) # (batch_size, num_gridding, num_cls_per_lane, num_of_lanes)

        col_sample = np.linspace(0, 800 - 1, cfg['griding_num'])
        col_sample_w = col_sample[1] - col_sample[0]

        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :] # flips rows
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0) # removes the last class, which is often reserved for no lane / background.
        idx = np.arange(cfg['griding_num']) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0) # expectation / avg idx
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == cfg['griding_num']] = 0
        out_j = loc

        vis = cv2.imread(img_paths[0])
        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                        cv2.circle(vis,ppp,5,(0,255,0),-1)
        
        img_name = os.path.splitext(os.path.basename(img_paths[0]))[0]
        cv2.imwrite(os.path.join(cfg['out_folder'], img_name + '.jpg'), vis)

if __name__ == "__main__":
    with open("demo.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    main(cfg)