"""Independent Torch-only LPIPS-Alex v0.1 formula; no pip/torchvision/network.

Architecture/calibration definition: richzhang/PerceptualSimilarity, commit
082bb24f84c091ea94de2867d34c4544f68e0963. Not a new learned metric.
"""
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from docci_text_guided_image_v20.core import sha

ASSETS = {
    'alexnet.pth':'7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02',
    'linear.pth':'df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0'
}

class Perceptual(nn.Module):
    def __init__(self, folder):
        super().__init__()
        folder = Path(folder)
        for name,h in ASSETS.items():
            if sha(folder/name) != h:
                raise ValueError('Official offline perceptual asset differs: '+name)
        alex = torch.load(str(folder/'alexnet.pth'),map_location='cpu')
        linear = torch.load(str(folder/'linear.pth'),map_location='cpu')
        channels = [64,192,384,256,256]
        self.convs = nn.ModuleList([
            nn.Conv2d(3,64,11,4,2), nn.Conv2d(64,192,5,1,2),
            nn.Conv2d(192,384,3,1,1), nn.Conv2d(384,256,3,1,1),
            nn.Conv2d(256,256,3,1,1)])
        if set(linear) != {'lin%d.model.1.weight'%i for i in range(5)}:
            raise ValueError('LPIPS v0.1 calibration schema differs')
        for i,(conv,index) in enumerate(zip(self.convs,(0,3,6,8,10))):
            conv.load_state_dict({k:alex['features.%d.%s'%(index,k)] for k in ('weight','bias')},strict=True)
            weight = linear['lin%d.model.1.weight'%i]
            if weight.shape != (1,channels[i],1,1) or not torch.isfinite(weight).all():
                raise ValueError('LPIPS calibration tensor differs')
            self.register_buffer('weight%d'%i,weight.clone())
        self.register_buffer('shift',torch.tensor([-.030,-.088,-.188])[None,:,None,None])
        self.register_buffer('scale',torch.tensor([.458,.448,.450])[None,:,None,None])
        self.requires_grad_(False).eval()

    def features(self, x):
        x = (x*2-1-self.shift)/self.scale
        values = []
        for i,conv in enumerate(self.convs):
            if i in (1,2):
                x = F.max_pool2d(x,3,2)
            x = F.relu(conv(x))
            values.append(x/(x.square().sum(1,keepdim=True).sqrt()+1e-10))
        return values

    def forward(self, x, target):
        if x.shape != target.shape or min(x.shape[-2:]) < 32:
            raise ValueError('LPIPS needs same RGB view and both sides >=32')
        return sum(F.conv2d((a-b).square(),getattr(self,'weight%d'%i)).mean((1,2,3))
                   for i,(a,b) in enumerate(zip(self.features(x),self.features(target))))
