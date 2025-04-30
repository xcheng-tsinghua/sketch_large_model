import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(
            tensor_all,
            tensor,
            async_op=False  # performance opt
        )

        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


class ULIPWithImageLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, pc_embed, text_embed, image_embed, sketch_embed, logit_scale):
        # pc_embed = outputs['pc_embed']
        # text_embed = outputs['text_embed']
        # image_embed = outputs['image_embed']
        # logit_scale = outputs['logit_scale']
        local_batch_size = pc_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * get_rank() + torch.arange(
                local_batch_size, device=pc_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # normalized features
        pc_embed = F.normalize(pc_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        sketch_embed = F.normalize(sketch_embed, dim=-1, p=2)

        # gather features from all GPUs
        pc_embed_all, text_embed_all, image_embed_all, sketch_embed_all = \
            all_gather_batch([pc_embed, text_embed, image_embed, sketch_embed])

        # cosine similarity as logits
        # logits_per_pc_text = logit_scale * pc_embed @ text_embed_all.t()
        # logits_per_text_pc = logit_scale * text_embed @ pc_embed_all.t()
        #
        # logits_per_pc_image = logit_scale * pc_embed @ image_embed_all.t()
        # logits_per_image_pc = logit_scale * image_embed @ pc_embed_all.t()

        logits_per_sketch_image = logit_scale * sketch_embed @ image_embed_all.t()
        logits_per_image_sketch = logit_scale * image_embed @ sketch_embed_all.t()

        logits_per_sketch_pc = logit_scale * sketch_embed @ pc_embed_all.t()
        logits_per_pc_sketch = logit_scale * pc_embed @ sketch_embed_all.t()

        logits_per_sketch_text = logit_scale * sketch_embed @ text_embed_all.t()
        logits_per_text_sketch = logit_scale * text_embed @ sketch_embed_all.t()

        # loss = (F.cross_entropy(logits_per_pc_text, self.labels) + \
        #         F.cross_entropy(logits_per_text_pc, self.labels)) / 2 + \
        #         (F.cross_entropy(logits_per_pc_image, self.labels) +
        #          F.cross_entropy(logits_per_image_pc, self.labels)) / 2

        loss = (F.cross_entropy(logits_per_sketch_image, self.labels) + \
                F.cross_entropy(logits_per_image_sketch, self.labels)) / 2 + \
                (F.cross_entropy(logits_per_sketch_pc, self.labels) +
                F.cross_entropy(logits_per_pc_sketch, self.labels)) / 2 + \
                (F.cross_entropy(logits_per_sketch_text, self.labels) +
                F.cross_entropy(logits_per_text_sketch, self.labels)) / 2

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_sketch_text, dim=-1)
            correct = pred.eq(self.labels).sum()
            sketch_text_acc = 100 * correct / local_batch_size

            pred = torch.argmax(logits_per_sketch_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            sketch_image_acc = 100 * correct / local_batch_size

            pred = torch.argmax(logits_per_sketch_pc, dim=-1)
            correct = pred.eq(self.labels).sum()
            sketch_pc_acc = 100 * correct / local_batch_size

        return {'loss': loss, 'ulip_loss': loss, 'sketch_image_acc': sketch_image_acc, 'sketch_text_acc': sketch_text_acc, 'sketch_pc_acc': sketch_pc_acc}


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pcd_embed, text_embed, image_embed, sketch_embed, logit_scale):
        loss_skh_txt = F.mse_loss(sketch_embed, text_embed)
        loss_skh_pcd = F.mse_loss(sketch_embed, pcd_embed)
        loss_skh_img = F.mse_loss(sketch_embed, image_embed)
        loss = loss_skh_txt + loss_skh_pcd + loss_skh_img

        return {'loss': loss}


def constructive_loss(x, y, margin=1.0, lambda_=1.0):
    """
    对比损失
    :param x: [bs ,emb]
    :param y: [bs ,emb]
    :param margin:
    :param lambda_:
    :return:
    """
    # x, y: tensors of shape (N, D)
    N = x.size(0)

    # 计算对应行之间的距离
    pos_dist = F.pairwise_distance(x, y, p=2)
    pos_loss = torch.mean(pos_dist ** 2)

    # 计算 x 与 y 中所有不同行之间的距离
    x_exp = x.unsqueeze(1)  # (N, 1, D)
    y_exp = y.unsqueeze(0)  # (1, N, D)
    dist_matrix = torch.norm(x_exp - y_exp, dim=2, p=2)  # (N, N)

    # 创建掩码，排除对角线（即对应行）
    mask = ~torch.eye(N, dtype=torch.bool, device=x.device)
    neg_dist = dist_matrix[mask]

    # 计算不同行之间的损失
    neg_loss = torch.mean(F.relu(margin - neg_dist) ** 2)

    # 总损失
    loss = pos_loss + lambda_ * neg_loss
    return loss

