import numpy as np
from torch.utils.data import Dataset
import os
from transformers import AutoTokenizer
from pathlib import Path
from torchvision import transforms
from PIL import Image


def get_subdirs(dir_path):
    """
    获取 dir_path 的所有一级子文件夹
    仅仅是文件夹名，不是完整路径
    """
    path_allclasses = Path(dir_path)
    directories = [str(x) for x in path_allclasses.iterdir() if x.is_dir()]
    dir_names = [item.split(os.sep)[-1] for item in directories]

    return dir_names


def get_allfiles(dir_path, suffix='txt', filename_only=False):
    '''
    获取dir_path下的全部文件路径
    '''
    filepath_all = []

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.split('.')[-1] == suffix:
                if filename_only:
                    current_filepath = file
                else:
                    current_filepath = str(os.path.join(root, file))
                filepath_all.append(current_filepath)

    return filepath_all


class UlipDataset(Dataset):
    """
    定位文件的路径如下：
    root
    ├─ train
    │   ├─ Bushes
    │   │   ├─model.STEP
    │   │   ├─image.png
    │   │   ├─sketch.txt
    │   │   ├─point_cloud.txt
    │   │   ...
    │   │
    │   ├─ Clamps
    │   │   ├─model.STEP
    │   │   ├─image.png
    │   │   ├─sketch.txt
    │   │   ├─point_cloud.txt
    │   │   ...
    │   │
    │   ...
    │
    ├─ test
    │   ├─ Bushes
    │   │   ├─model.STEP
    │   │   ├─image.png
    │   │   ├─sketch.txt
    │   │   ├─point_cloud.txt
    │   │   ...
    │   │
    │   ├─ Clamps
    │   │   ├─model.STEP
    │   │   ├─image.png
    │   │   ├─sketch.txt
    │   │   ├─point_cloud.txt
    │   │   ...
    │   │
    │   ...
    │

    """
    def __init__(self,
                 root=r'D:\document\DeepLearning\DataSet\root',
                 is_train=True,
                 n_pcd_points=2048,  # 点云中的点数
                 n_skh_points=256,  # 草图中的点数
                 data_argumentation=False
                 ):

        print('sketch dataset, from:' + root)
        self.data_augmentation = data_argumentation
        self.n_pcd_points = n_pcd_points
        self.n_skh_points = n_skh_points

        if is_train:
            inner_root = os.path.join(root, 'train')
        else:
            inner_root = os.path.join(root, 'test')

        # 获取全部类别列表，即 inner_root 内的全部文件夹名
        category_all = get_subdirs(inner_root)
        self.datapath = []

        for c_class in category_all:
            class_root = os.path.join(inner_root, c_class)
            c_text = f'an image of {c_class}'

            # 找到该文件夹下所有的文件夹，每个文件夹里包含成组的数据
            pair_data = get_subdirs(class_root)

            for c_pair_data in pair_data:
                pair_data_root = os.path.join(class_root, c_pair_data)

                # 点云数据路径：
                c_pcd_path = os.path.join(pair_data_root, c_pair_data + '.txt')

                # 草图及图片数据
                c_skh_name_list = get_allfiles(pair_data_root, 'txt', True)
                c_skh_name_list = [c_name.replace('.txt', '') for c_name in c_skh_name_list if c_pair_data + '_' in c_name]

                for c_skh_name in c_skh_name_list:
                    # 草图数据
                    c_skh_path = os.path.join(pair_data_root, c_skh_name + '.txt')

                    # 图片数据
                    c_img_path = os.path.join(pair_data_root, c_skh_name + '.png')

                    self.datapath.append((c_text, c_pcd_path, c_skh_path, c_img_path))

        # 将文本转化为Tensor
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        print('number of instance all:', len(self.datapath))

    def __getitem__(self, index):
        """
        :return: [stroke1, stroke2, ..., stroke_n] (list)
        stroke = [n_stroke_point, 2] (numpy.ndarray)
        """
        fn = self.datapath[index]  # (‘plane’, Path1)

        # text embedding
        c_text = fn[0]
        token_book = self.tokenizer.encode_plus(c_text, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
        text_data = token_book["input_ids"].squeeze()

        # point cloud embedding
        c_pcd_path = fn[1]
        pcd_data = np.loadtxt(c_pcd_path, delimiter='\t')[:, :3]
        # 保证点数相同
        choice = np.random.choice(pcd_data.shape[0], self.n_pcd_points, replace=True)
        pcd_data = pcd_data[choice, :]

        # sketch embedding
        c_skh_path = fn[2]
        skh_data = np.loadtxt(c_skh_path, delimiter=',')[:, :2]
        # 保证点数相同
        choice = np.random.choice(skh_data.shape[0], self.n_skh_points, replace=True)
        skh_data = skh_data[choice, :]

        # image embedding
        c_img_path = fn[3]
        image = Image.open(c_img_path).convert("RGB")  # 确保是 RGB 模式
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()  # 转换为 [C, H, W] 格式的 Tensor，值在 [0, 1] 之间
        ])
        tensor_image = transform(image)

        return text_data, pcd_data, skh_data, tensor_image

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    # # 加载预训练模型和分词器
    # model_name = "bert-base-uncased"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # # model = AutoModel.from_pretrained(model_name)
    #
    # # 示例文本
    # text = "an image of a gear"
    #
    # # 将文本转换为 Token IDs
    # inputs = tokenizer.encode_plus(text, max_length=10, padding="max_length", truncation=True, return_tensors="pt")
    #
    # # 查看输出
    # print("Input IDs:", inputs["input_ids"].squeeze())
    # print("Attention Mask:", inputs["attention_mask"])
    # print("token_type_ids:", inputs["token_type_ids"])

    adataset = UlipDataset()
    adata = adataset[0]

    print('text emb: ', adata[0].size())
    print('point cloud: ', adata[1].shape)
    print('sketch points: ', adata[2].shape)
    print('image: ', adata[3].size())

    pass









