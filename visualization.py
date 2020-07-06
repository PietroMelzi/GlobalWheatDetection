import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns


TEST_DIR = "data/test/"


def get_all_bboxes(df, image_id):
    image_bboxes = df[df.image_id == image_id]

    bboxes = []
    # for _, row in image_bboxes.iterrows():
    #     bboxes.append((row.bbox_xmin, row.bbox_ymin, row.bbox_width, row.bbox_height))
    image_bboxes = image_bboxes['PredictionString'].iloc[0]
    image_bboxes = image_bboxes.split()

    a = len(image_bboxes)

    for i in range(len(image_bboxes)//5):
        bboxes.append((int(image_bboxes[5*i + 1]), int(image_bboxes[5*i + 2]), int(image_bboxes[5*i + 3]),
                       int(image_bboxes[5*i + 4])))

    return bboxes


def plot_image_examples(df, rows=3, cols=4, title='Image examples'):
    fig, axs = plt.subplots(rows, cols, figsize=(10, 10))
    for row in range(rows):
        for col in range(cols):
            # idx = np.random.randint(len(df), size=1)[0]
            idx = (row * cols + col) % 10
            img_id = df.iloc[idx].image_id

            img = Image.open(TEST_DIR + img_id + '.jpg')
            axs[row, col].imshow(img)
            bboxes = get_all_bboxes(df, img_id)

            for bbox in bboxes:
                rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r',
                                         facecolor='none')
                axs[row, col].add_patch(rect)

            axs[row, col].axis('off')

    plt.suptitle(title)
    plt.show()


path = "data/submission/submission_t=0.45.csv"
df = pd.read_csv(path)
plot_image_examples(df)
