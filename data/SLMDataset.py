import numpy as np
from torch.utils.data import Dataset
import os
from pathlib import Path
from torchvision import transforms
from PIL import Image
import torch
from functools import lru_cache
import ftfy
import regex as re
import html
import gzip


@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

    def __call__(self, texts, context_length=77):
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.encoder["<|startoftext|>"]
        eot_token = self.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            tokens = tokens[:context_length]
            result[i, :len(tokens)] = torch.tensor(tokens)

        if len(result) == 1:
            return result[0]
        return result


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


class SLMataset(Dataset):
    """
    Sketch Large Model Dataset

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
        # self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer = SimpleTokenizer()

        print('number of instance all:', len(self.datapath))

    @staticmethod
    def sketch_std(sketch):
        """
        草图质心移动到原点，大小缩放到 [-1, 1]^2
        Args:
            sketch: [n_pnts, 4]

        Returns:

        """
        coordinates = sketch[:, :2]

        mean_coor = np.mean(coordinates, axis=0)
        coordinates = coordinates - mean_coor  # 实测是否加expand_dims效果一样
        dist = np.max(np.sqrt(np.sum(coordinates ** 2, axis=1)), 0)
        coordinates = coordinates / dist

        sketch[:, :2] = coordinates
        return sketch

    def get_sketch_and_mask(self, sketch_root):
        data_raw = np.loadtxt(sketch_root, delimiter=',')
        data_raw = data_raw[:, :3]

        # 多于指定点数则进行采样
        n_point_raw = len(data_raw)
        if n_point_raw > self.n_skh_points:
            choice = np.random.choice(n_point_raw, self.n_skh_points, replace=True)
            data_raw = data_raw[choice, :]

        data_raw = self.sketch_std(data_raw)

        # 将16， 17 改为 0， 1
        c_sketch_len = len(data_raw)
        data_raw[data_raw == 16] = 0
        data_raw[data_raw == 17] = 1
        data_raw = torch.from_numpy(data_raw)

        data_cube = torch.zeros(self.n_skh_points, 5, dtype=torch.float)
        mask = torch.zeros(self.n_skh_points, dtype=torch.float)

        data_cube[:c_sketch_len, :2] = data_raw[:, :2]
        data_cube[:c_sketch_len, 2] = data_raw[:, 2]
        data_cube[:c_sketch_len, 3] = 1 - data_raw[:, 2]
        data_cube[-1, 4] = 1

        mask[:c_sketch_len] = 1

        return data_cube, mask

    def __getitem__(self, index):
        """
        :return: [stroke1, stroke2, ..., stroke_n] (list)
        stroke = [n_stroke_point, 2] (numpy.ndarray)
        """
        fn = self.datapath[index]  # (‘plane’, Path1)

        # text embedding
        c_text = fn[0]
        # token_book = self.tokenizer.encode_plus(c_text, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
        # text_data = token_book["input_ids"].squeeze()
        text_data = self.tokenizer(c_text)

        # point cloud embedding
        c_pcd_path = fn[1]
        pcd_data = np.loadtxt(c_pcd_path, delimiter='\t')[:, :3]
        # 保证点数相同
        choice = np.random.choice(pcd_data.shape[0], self.n_pcd_points, replace=True)
        pcd_data = pcd_data[choice, :]

        # sketch embedding
        c_skh_path = fn[2]
        sketch, mask = self.get_sketch_and_mask(c_skh_path)
        # skh_data = np.loadtxt(c_skh_path, delimiter=',')[:, :2]
        # # 保证点数相同
        # choice = np.random.choice(skh_data.shape[0], self.n_skh_points, replace=True)
        # skh_data = skh_data[choice, :]

        # image embedding
        c_img_path = fn[3]
        image = Image.open(c_img_path).convert("RGB")  # 确保是 RGB 模式
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()  # 转换为 [C, H, W] 格式的 Tensor，值在 [0, 1] 之间
        ])
        tensor_image = transform(image)

        return text_data, pcd_data, sketch, mask, tensor_image

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









