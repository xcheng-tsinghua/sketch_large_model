import argparse

import numpy as np
import torch
from tqdm import tqdm
from encoders.ImageEncoder_ULIP import create_pretrained_imageencoder
from encoders.TextEncoder_ULIP import create_pretrained_textencoder
from encoders.PointBERT_ULIP2 import create_pretrained_pointbert
from encoders.sketch_encoder import SketchBERT
from data.SLMDataset import SLMataset
# from encoders.loss_func import ULIPWithImageLoss as SketchLoss
from encoders.loss_func import MSELoss as SketchLoss
import torch.amp as amp
from colorama import Fore, Back, init


def parse_args():
    parser = argparse.ArgumentParser('training')

    parser.add_argument('--bs', type=int, default=100, help='batch size in training')
    parser.add_argument('--epoch', default=2000, type=int, help='number of epoch in training')
    parser.add_argument('--is_load_weight', type=str, default='True', choices=['True', 'False'], help='is load pretrained weight')
    parser.add_argument('--n_skh_pnt', type=int, default=256, help='is load pretrained weight')
    parser.add_argument('--n_pcd_pnt', type=int, default=2048, help='is load pretrained weight')

    parser.add_argument('--local', type=str, default='False', choices=['True', 'False'], help='local or sever?')
    parser.add_argument('--root_sever', type=str, default=rf'/root/my_data/data_set/sketch_large_model_dataset', help='data root in sever')
    parser.add_argument('--root_local', type=str, default=rf'D:\document\DeepLearning\DataSet\sketch_large_model_dataset', help='data root in local')

    return parser.parse_args()


def main(args):

    # 设置数据集
    if args.local == 'True':
        data_root = args.root_local
    else:
        data_root = args.root_sever

    train_dataset = SLMataset(root=data_root, is_train=True, n_skh_points=args.n_skh_pnt, n_pcd_points=args.n_pcd_pnt)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4)

    # 加载模型
    pre_img_enc = create_pretrained_imageencoder().cuda()
    pre_txt_enc = create_pretrained_textencoder().cuda()
    pre_pnt_enc = create_pretrained_pointbert().cuda()

    sketch_enc = SketchBERT(args.n_skh_pnt).cuda()

    '''加载权重'''
    if args.is_load_weight == 'True':
        try:
            sketch_enc.load_state_dict(torch.load('./weights/sketch_encoder.pth'))
            print(Fore.GREEN + 'training from exist model: ./weights/sketch_encoder.pth')
        except:
            print(Fore.GREEN + 'no existing model, training from scratch')
    else:
        print(Fore.BLACK + Back.BLUE + 'does not load state dict, training from scratch')

    '''定义优化器'''
    optimizer = torch.optim.AdamW(
        sketch_enc.parameters(), lr=0.003, betas=(0.9, 0.98),
        eps=1e-08, weight_decay=0.1
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    criterion = SketchLoss()
    # scaler = amp.GradScaler(enabled=not args.disable_amp)

    for epoch in range(args.epoch):
        sketch_enc = sketch_enc.train()

        loss_list = []

        for batch_id, data in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader)):
            text_data, pcd_data, skh_data, mask_data, tensor_image = data[0].long().cuda(), data[1].float().cuda(), data[2].float().cuda(), data[3].float().cuda(), data[4].float().cuda()

            optimizer.zero_grad()
            # with amp.autocast('cuda', enabled=True):
            #
            #     text_embed = pre_txt_enc(text_data).detach()
            #     pcd_embed = pre_pnt_enc(pcd_data).detach()
            #     image_embed = pre_img_enc(tensor_image).detach()
            #     sketch_embed, logit_scale = sketch_enc(skh_data)  # 需要训练
            #
            #     loss_dict = criterion(pcd_embed, text_embed, image_embed, sketch_embed, logit_scale)
            #     loss = loss_dict['loss']

            text_embed = pre_txt_enc(text_data).detach().clone()
            pcd_embed = pre_pnt_enc(pcd_data).detach().clone()
            image_embed = pre_img_enc(tensor_image).detach().clone()
            sketch_embed = sketch_enc(skh_data, mask_data)  # 需要训练

            loss_dict = criterion(pcd_embed, text_embed, image_embed, sketch_embed, 1 / 0.07)
            loss = loss_dict['loss']

            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

        scheduler.step()
        torch.save(sketch_enc.state_dict(), './weights/sketch_encoder.pth')
        print(Back.BLUE + f'{epoch} / {args.epoch}: save sketch weights at: ./weights/sketch_encoder.pth, loss: {np.mean(loss_list)}')


if __name__ == '__main__':
    init(autoreset=True)
    main(parse_args())





