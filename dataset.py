import pandas as pd
import numpy as np
import re
import cv2
import torch
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2
import evaluation as ev

DIR_INPUT = 'data'
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_TEST = f'{DIR_INPUT}/test'
valid_units = 665


class DataGenerator(Dataset):

    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)

        boxes = records[['x', 'y', 'w', 'h']].values

        area = boxes[:, 3] * boxes[:, 2]
        area = torch.as_tensor(area, dtype=torch.float32)

        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        # there is only one class
        labels = torch.ones((records.shape[0],), dtype=torch.int64)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']

            target['boxes'] = torch.tensor(sample['bboxes'])

        target['boxes'] = target['boxes'].type(torch.float32)
        target['boxes'] = target['boxes'].reshape(-1, 4)
        return image, target, image_id

    def __len__(self):
        return self.image_ids.shape[0]


# Albumentations
def get_train_transform():
    return A.Compose([
        A.CLAHE(clip_limit=3, p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Resize(512, 512),
        A.ChannelDropout(p=0.01),
        A.ShiftScaleRotate(
            shift_limit=0.5,
            scale_limit=(0, 0.5),
            rotate_limit=90,
            border_mode=0,
            value=0
        ),
        A.OneOf([
            A.RandomBrightnessContrast(p=1),
            A.RandomGamma(p=1),
        ], p=0.2),
        A.Normalize(mean=0.0, std=1.0, p=1),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_valid_transform():
    return A.Compose([
        # A.Resize(512, 512),
        A.Normalize(mean=0.0, std=1.0, p=1),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r


def get_train_valid_data():
    train_df = pd.read_csv(f'{DIR_INPUT}/train.csv')

    train_df['x'] = -1
    train_df['y'] = -1
    train_df['w'] = -1
    train_df['h'] = -1

    train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))
    train_df.drop(columns=['bbox'], inplace=True)
    train_df['x'] = train_df['x'].astype(np.float)
    train_df['y'] = train_df['y'].astype(np.float)
    train_df['w'] = train_df['w'].astype(np.float)
    train_df['h'] = train_df['h'].astype(np.float)

    # remove boxes over 20000
    train_df['area'] = train_df['w'] * train_df['h']
    train_df = train_df.drop(train_df[train_df.area > 20000].index)

    image_ids = train_df['image_id'].unique()
    # split the dataset in train and test set
    # indices = torch.randperm(len(image_ids)).tolist()
    # valid_ids = image_ids[indices[-valid_units:]]
    # train_ids = image_ids[indices[:-valid_units]]

    train_ids, valid_ids = ev.split_same_loc_dist(train_df)
    valid_df = train_df[train_df['image_id'].isin(valid_ids)]
    train_df = train_df[train_df['image_id'].isin(train_ids)]

    return train_df, valid_df
