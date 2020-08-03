from PIL import Image
import torch
import torchvision
import glob

class Chairs(torch.utils.data.Dataset):
    def __init__(self, root_path, is_train = True):
        super(Chairs, self).__init__()

        img_paths = glob.glob(root_path + '/*/*/*.png')

        trainset_size = int(len(img_paths) * 0.8)
        self.img_paths = img_paths[:trainset_size] if is_train else img_paths[trainset_size:]
        # self.img_paths = img_paths[:64]
        
        self.tsfm = torchvision.transforms.Compose([
            torchvision.transforms.Resize(64),
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        return self.tsfm(img)
