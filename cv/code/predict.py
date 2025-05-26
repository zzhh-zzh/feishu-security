import pandas as pd
import torch
import csv
import glob
from model import build_model
from PIL import Image
from torchvision import transforms

# load your model
model = build_model()
model.load_state_dict(torch.load("cv\model_save\mobilenet_v3_small.pth"))
model.to("cuda")
model.eval()

transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.486, 0.456, 0.406],
                [0.229, 0.224, 0.225])
        ])

id2label = {0: 'cat', 1: 'dog', 2: 'other'}

res = {'image_path': [], 'label': []}
# for image_path in glob.glob('data/test/*'):
for image_path in glob.glob('cv/trainset/base_val/val/cat/*'):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to("cuda")
    
    outputs = model(image_tensor)
    score, label = torch.max(outputs, dim=-1)
    res['image_path'].append(image_path)
    res['label'].append(id2label[label.item()])

image_df = pd.DataFrame(data=res, dtype=str)
image_df.to_csv('res.csv', index=False, quoting=csv.QUOTE_ALL)