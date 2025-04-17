import argparse
import torch
from tqdm import tqdm
from encoders.ImageEncoder_ULIP import create_pretrained_imageencoder
from encoders.TextEncoder_ULIP import create_pretrained_textencoder
from encoders.PointBERT_ULIP2 import create_pretrained_pointbert
from encoders.sketch_encoder import SketchEncoder
from data.ulip_dataset import UlipDataset
from encoders.loss_func import ULIPWithImageLoss as SketchLoss
import torch.amp as amp


def parse_args():
    parser = argparse.ArgumentParser('training')

    parser.add_argument('--bs', type=int, default=10, help='batch size in training')
    parser.add_argument('--epoch', default=2000, type=int, help='number of epoch in training')

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

    train_dataset = UlipDataset(root=data_root, is_train=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4)

    # 加载模型
    pre_img_enc = create_pretrained_imageencoder().cuda()
    pre_txt_enc = create_pretrained_textencoder().cuda()
    pre_pnt_enc = create_pretrained_pointbert().cuda()

    sketch_enc = SketchEncoder().cuda()

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

        for batch_id, data in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader)):
            text_data, pcd_data, skh_data, tensor_image = data[0].long().cuda(), data[1].float().cuda(), data[2].float().cuda(), data[3].float().cuda()

            optimizer.zero_grad()
            with amp.autocast('cuda', enabled=True):

                text_embed = pre_txt_enc(text_data).detach()
                pcd_embed = pre_pnt_enc(pcd_data).detach()
                image_embed = pre_img_enc(tensor_image).detach()
                sketch_embed, logit_scale = sketch_enc(skh_data)  # 需要训练

                loss_dict = criterion(pcd_embed, text_embed, image_embed, sketch_embed, logit_scale)
                loss = loss_dict['loss']

            loss.backward()
            optimizer.step()

        scheduler.step()
        torch.save(sketch_enc.state_dict(), './weights/sketch_encoder.pth')


if __name__ == '__main__':
    main(parse_args())














