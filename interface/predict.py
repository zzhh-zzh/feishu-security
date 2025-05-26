import argparse
import json
import torch
from model import RobertaClassifier,build_model
from transformers import AutoTokenizer
from PIL import Image
from torchvision import transforms
from io import BytesIO
from urllib.request import urlopen
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def predict_text(text):
    # load your model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    model = RobertaClassifier(
        model_name="model_save/text/roberta",
        num_classes=3,
        dropout=0.1
    )
    model.load_state_dict(torch.load("model_save/text/model_state_dict.pth",map_location=torch.device(device)))
    tokenizer = AutoTokenizer.from_pretrained("model_save/text/")
    model.to(device)
    model.eval()

    id2label = {0: '消极', 1: '中性', 2: '积极'}
    
    with torch.no_grad():
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        scores = outputs['logits']
        _, label = torch.max(scores, dim=-1)
        return {"type":id2label[label.item()]}
    
def predict_cv(text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # load your model
    model = build_model()
    model.load_state_dict(torch.load("model_save\cv\mobilenet_v3_small.pth",map_location=torch.device(device)))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.486, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
            ])

    id2label = {0: '猫', 1: '狗', 2: '其他动物'}
    
    with torch.no_grad():
        response = urlopen(text)
        image = Image.open(BytesIO(response.read()))
        image_tensor = transform(image).unsqueeze(0).to(device)
    
        outputs = model(image_tensor)
        _, label = torch.max(outputs, dim=-1)
    return{"type":id2label[label.item()]}
     
def save_to_file(data, filename = "result.json"):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)     
     
def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)
    # 动物识别命令
    animal_parser = subparsers.add_parser('animal', help='动物识别分析')
    animal_parser.add_argument('--image-url', required=True, help='图片URL')
    # 情绪分析命令
    emotion_parser = subparsers.add_parser('emotion', help='情绪分析')
    emotion_parser.add_argument('--text', required=True, help='待分析文本')
    args = parser.parse_args()
    try:
        if args.command == 'animal':
            result = predict_cv(args.image_url)
        elif args.command == 'emotion':
            result = predict_text(args.text)
        save_to_file(result)
    except Exception as e:
        result = {"error": str(e)}
        save_to_file(result)
        exit(1)

if __name__ == '__main__':
    main()

   
# "https://pic1.zhimg.com/v2-a58fa2ab84be291418da2652805f8270_b.jpg"