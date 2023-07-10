import torch
import torchvision.models as models


def main():
    mod = models.vgg16(weights='IMAGENET1K_V1')
    torch.save(mod.state_dict(), 'mod_weights.pth')
    mod = models.vgg16()
    mod.load_state_dict(torch.load('mod_weights.pth'))
    mod.eval()
    torch.save(mod, 'mod.pth')
    mod = torch.load('mod.pth')
                        
