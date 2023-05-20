import os
import sys
import time
import glob
import numpy as np
import torch
import utils
from PIL import Image
import logging
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from model.model import *
from model.xiaoye import *
from multi_read_data import MemoryFriendlyLoader_train, MemoryFriendlyLoader_test
import kornia
import torch.nn.functional as F
import cv2

from histloss import HistogramLoss
from skimage.metrics.simple_metrics import peak_signal_noise_ratio as compare_psnr


parser = argparse.ArgumentParser("SCI")
parser.add_argument('--batch_size', type=int, default=2, help='batch size')
parser.add_argument('--cuda', default=True, type=bool, help='Use CUDA to train model')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--epochs', type=int, default=50, help='epochs')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--stage', type=int, default=3, help='epochs')
parser.add_argument('--save', type=str, default='EXP/', help='location of the data corpus')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.save = args.save + '/' + 'Train-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('./model/*.py'))
model_path = args.save + '/model_epochs/'
os.makedirs(model_path, exist_ok=True)
image_path = args.save + '/image_epochs/'
os.makedirs(image_path, exist_ok=True)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info("train file name = %s", os.path.split(__file__))

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    # image_numpy = np.clip(np.transpose(image_numpy, (1, 2, 0))*255, 0, 255).astype('uint8')
    # # im_rgb = cv2.cvtColor(image_numpy, cv2.COLOR_HSV2RGB)
    # cv2.imwrite(path, image_numpy)

    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(((image_numpy-np.min(image_numpy))/(np.max(image_numpy)-np.min(image_numpy))) * 255.0, 0, 255.0).astype('uint8'))#((image_numpy-np.min(image_numpy))/(np.max(image_numpy)-np.min(image_numpy)))
    im.save(path, 'png')

def save_images_gray(tensor, path):
    # tensor = tensor.squeeze(0)
    image_numpy = tensor[0].cpu().float().numpy()

    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(((image_numpy-np.min(image_numpy))/(np.max(image_numpy)-np.min(image_numpy))) * 255.0, 0, 255.0).astype('uint8'))
    im = im.convert('L')
    im.save(path, 'png')

def Psnr(img1, img2):
   mse = np.mean( (img1 - img2) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %s' % args.gpu)
    logging.info("args = %s", args)
    ssim_loss = kornia.losses.SSIMLoss(5)
    psnr_loss = kornia.losses.PSNRLoss(1)
    l1loss = torch.nn.L1Loss()
    histloss = HistogramLoss()
    l2loss = nn.MSELoss()

    model = Network()
    #
    # model.net.in_conv.apply(model.weights_init)
    # model.net.conv.apply(model.weights_init)
    # model.net.out_conv.apply(model.weights_init)

    # model = Wnet(1)

    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=3e-4)
    MB = utils.count_parameters_in_MB(model)
    logging.info("model size = %f", MB)
    print(MB)


    train_low_data_names = './data/train/'      ######训练集位置
    TrainDataset = MemoryFriendlyLoader_train(img_dir=train_low_data_names, task='train')


    test_low_data_names = './data/test/'          ########测试集位置
    TestDataset = MemoryFriendlyLoader_test(img_dir=test_low_data_names, task='test')

    train_queue = torch.utils.data.DataLoader(
        TrainDataset, batch_size=args.batch_size,
        pin_memory=True, num_workers=0, shuffle=True, generator=torch.Generator(device='cuda'))

    test_queue = torch.utils.data.DataLoader(
        TestDataset, batch_size=1,
        pin_memory=True, num_workers=0, shuffle=True, generator=torch.Generator(device='cuda'))
    total_step = 0
    best_epoch = 0
    best_psnr = 100
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch_idx, (low, high, _) in enumerate(train_queue):
            total_step += 1
            low = Variable(low, requires_grad=False).cuda()
            high = Variable(high, requires_grad=False).cuda()
            # n, c, h, w = high.shape()
            # maskh = torch.zeros(n, c, h, w)
            # maskl = torch.zeros(n, c, h, w)

            # high[:, :, 0:511] = 0
            # low[:, :, 0:511] = 0
            high = F.interpolate(high, size=[1024, 1024], mode="bilinear")
            low = F.interpolate(low, size=[1024, 1024], mode="bilinear")
            optimizer.zero_grad()
            loss = model._loss(low, high, epoch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            losses.append(loss.item())
            # logging.info('train-epoch %03d %03d %f', epoch, batch_idx, loss)



        logging.info('train-epoch %03d %f', epoch, np.average(losses))
        torch.save(model.state_dict(), os.path.join(model_path, 'weights_%d.pth' % epoch))

        if epoch % 1 == 0 and total_step != 0:
            ssim = 0
            psnr = 0
            logging.info('train %03d %f', epoch, loss)
            model.eval()
            with torch.no_grad():
                for _, (input, label, image_name) in enumerate(test_queue):
                    input = Variable(input, volatile=True).cuda()
                    label = Variable(label, volatile=True).cuda()
                    input = F.interpolate(input, size=[1024, 1024], mode="bilinear")
                    label = F.interpolate(label, size=[1024, 1024], mode="bilinear")
                    image_name = image_name[0].split('\\')[-1].split('.')[0]
                    out, i, i_e, r, i_h = model(input, label)

                    out = (out - torch.min(out)/(torch.max(out)-torch.min(out)))
                    ssim += 1 - 2*ssim_loss(out, label)
                    psnr += compare_psnr(out.cpu().numpy(), label.cpu().numpy(), data_range=1)
                    u_name = '%s.jpg' % (image_name)

                    image_path1 = image_path + str(epoch)
                    os.makedirs(image_path1, exist_ok=True)
                    u_path = image_path1 + '/' + u_name
                    save_images(out, u_path)

            metric_path = image_path1 + '/ssim' + str(ssim/len(test_queue)) + 'psnr' + str(psnr/len(test_queue)) + '/'
            if best_psnr > np.average(losses) and epoch != 0:
                best_psnr = np.average(losses)
                best_epoch = epoch
            os.makedirs(metric_path, exist_ok=True)
        print(best_epoch)

if __name__ == '__main__':
    main()
