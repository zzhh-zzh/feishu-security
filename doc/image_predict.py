import pandas as pd
import torch
import csv
import glob
from PIL import Image

# load your model
model = Model().cuda()
id2label = {0: 'cat', 1: 'dog', 2: 'other'}

res = {'image_path': [], 'label': []}
for image_path in glob.glob('data/test/*'):
    scores = model(Image.open(image_path))
    score, label = torch.max(scores, dim=-1)
    res['image_path'].append(image_path)
    res['label'].append(id2label[label])

image_df = pd.DataFrame(data=res, dtype=str)
image_df.to_csv('res.csv', index=False, quoting=csv.QUOTE_ALL)