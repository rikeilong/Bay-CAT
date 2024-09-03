import os
import torch
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import glob
from decord import VideoReader
import cv2
import decord
import clip
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import random as rnd
device='cuda:1'

# clip_model,preprocess = clip.load('ViT-B/32',jit=False)
# clip_model.eval()
# clip_model = clip_model.cuda()

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)
from moviepy.editor import AudioFileClip


# def clip_feat_extract(img):

#     image = img.unsqueeze(0).cuda()
#     with torch.no_grad():
#         image_features = clip_model.encode_image(image)
#     return image_features

# tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
# frms = tensor_frms.permute(0, 3, 1, 2).float()  # (T, C, H, W)
# total_feats = []

def Image_feat_extract(video_path, height,width, n_frms, sampling="uniform"):
    temp_outfile = '/home/qilang/PythonProjects/ECCV_LLMs/temp_video'
    temp_paths = []
    vc = cv2.VideoCapture(video_path)
    fps = vc.get(cv2.CAP_PROP_FPS)
    c = 1
    rval=vc.isOpened()
    while rval:
        rval, frame = vc.read()  
        save_path = os.path.join(temp_outfile,str(c)+'.jpg')
        if rval:
            if (c % round(fps) == 0):
                cv2.imwrite(save_path, frame)
                temp_paths.append(save_path)
            c = c + 1
        else:
            break

    vc.release()

    inputs = {
        ModalityType.VISION: data.load_and_transform_vision_data(temp_paths, device),
    }
    with torch.no_grad():
        img_features = model(inputs)

    return img_features[ModalityType.VISION]

def audio_feat_extract(video_path):
    temp_outfile = 'temp_audio.wav'
    my_audio_clip = AudioFileClip(video_path)
    my_audio_clip.write_audiofile(temp_outfile)
    inputs = {
            ModalityType.AUDIO: data.load_and_transform_audio_data([temp_outfile], device),
        }
    with torch.no_grad():
        audio_features = model(inputs)
    audio_features = audio_features[ModalityType.AUDIO]

    return audio_features

# a = ImageClIP_feat_extract(video_path="/home/qilang/MUCIS-AVQA-videos-Synthetic/va00005060.mp4", height=224, width=224, n_frms=60, sampling="uniform")
# print(a.shape)
# audio_feat_extract("/home/qilang/MUCIS-AVQA-videos-Synthetic/esa00000307.mp4")