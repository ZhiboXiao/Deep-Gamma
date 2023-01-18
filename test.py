import os
import sys
import numpy as np
import torch
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable
from model.model import Finetunemodel
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from multi_read_data import MemoryFriendlyLoader_test
from torchsummary import summary
from model.model import *

parser = argparse.ArgumentParser("SCI")
parser.add_argument('--data_path', type=str, default='./data/data3/',
                    help='location of the data corpus')
parser.add_argument('--save_path', type=str, default='./results/data3', help='location of the data corpus')
parser.add_argument('--model', type=str, default='./weights/weights_116.pt', help='location of the data corpus')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')

args = parser.parse_args()
save_path = args.save_path
os.makedirs(save_path, exist_ok=True)

TestDataset = MemoryFriendlyLoader_test(img_dir=args.data_path, task='test')

test_queue = torch.utils.data.DataLoader(
    TestDataset, batch_size=1,
    pin_memory=True, num_workers=0)


def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    # image_numpy = np.clip(np.transpose(image_numpy, (1, 2, 0))*255, 0, 255).astype('uint8')
    # # im_rgb = cv2.cvtColor(image_numpy, cv2.COLOR_HSV2RGB)
    # cv2.imwrite(path, image_numpy)

    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))#((image_numpy-np.min(image_numpy))/(np.max(image_numpy)-np.min(image_numpy)))
    im.save(path, 'png')

def save_images_gray(tensor, path):
    tensor = tensor.squeeze(0)
    image_numpy = tensor[0].cpu().float().numpy()

    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')


def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    # model = Finetunemodel(args.model)
    model = Network().cuda()
    model.load_state_dict(torch.load(
        './weights/weights_55.pth',
        map_location=lambda storage, loc: storage))
    # model = model.cuda()
    model.eval()
    with torch.no_grad():
        for _, (input, label, image_name) in enumerate(test_queue):
            input = Variable(input, volatile=True).cuda()
            input = F.interpolate(input, size=[1024, 1024], mode="bilinear")
            image_name = image_name[0].split('\\')[-1].split('.')[0]
            enhance_image, i, i_enhance, r, i_h = model(input, input)
            u_name = '%s.png' % (image_name)
            print('processing {}'.format(u_name))
            u_path = save_path + '/' + u_name
            u_path1 = save_path + '/i' + u_name
            u_path2 = save_path + '/i_e' + u_name
            u_path3 = save_path + '/r' + u_name
            # out1 = torch.cat([input[:, 0, :, :].unsqueeze(1), input[:, 1, :, :].unsqueeze(1), out], 1)
            save_images(enhance_image, u_path)
            save_images_gray(i, u_path1)
            save_images_gray(i_enhance, u_path2)
            save_images(r, u_path3)



if __name__ == '__main__':
    main()
