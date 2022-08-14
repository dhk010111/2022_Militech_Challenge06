import torch as ch
import dill
from IPython import embed
from torchvision import transforms
from os import path

CLASS = 0 #patch 생성할 때 클래스 나눠줌
PATCH_SIZE = 150 # 25,50,100,150
ROBUST = True


to_pil = transforms.ToPILImage()
center_crop = transforms.CenterCrop(PATCH_SIZE)

#Unadversarial folder 기본 경로 추가
base_path = 'BASEPATH ADD'

eps = '3' if ROBUST else '0'
#colab으로 README 따라서 학습시키면 OUT_DIR 폴더에 pt.best 파일 생성되는 경로 추가()
model_path = f'checkpoint model path ADD'

checkpoint = ch.load(path.join(base_path, model_path), pickle_module=dill)
sd = checkpoint['model']
patches = sd['module.booster.patches'].cpu()

im = center_crop(to_pil(patches[CLASS]))
im.save(f'class_{CLASS}.png')
