import json
import base64
import requests
from PIL import Image
import io

class GemmaLocalClient:
    def __init__(self, base_url="http://localhost:1234"):
        self.base_url = base_url + "/v1/chat/completions"
    
    def analyze_image(self, image_path, user_prompt):
        """处理本地图片并调用Gemma模型"""
        # 1. 图像预处理
        base64_image = self._image_to_base64(image_path)
        
        system_prompt = """
            你是一位图像动物识别专家，专门识别图像中的动物类别和位置信息。

            请严格按以下流程分析：
            1. 分析图像内容，识别可能的动物区域
            2. 确定动物类别和精确位置
            3. 计算位置框坐标[起始x,起始y,宽,高]

            输出格式必须严格遵循以下模板：

            输入图像分析结果：

            识别依据：

            特征识别：
            位置分析：
            最终结果：
            {
            "类别": "动物类别",
            "位置框": [起始x, 起始y, 宽, 高]
            }
            """
        
        # 2. 构建多模态消息
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                     "image_url":{
                        "url": f"data:image/jpeg;base64,{base64_image}"
                        } 
                    },
                    {"text": user_prompt}
                ]
            }
        ]
        
        # 3. 调用模型
        response = self._call_model(
            model="27b",  # 根据实际部署模型名称调整
            messages=messages,
            temperature=0.3  # 降低随机性保证输出格式稳定
        )
        
        return self._parse_response(response)

    def _image_to_base64(self, file_path, max_size=1024):
        """图像压缩和编码"""
        img = Image.open(file_path)
        img.thumbnail((max_size, max_size))
        
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _call_model(self, model, messages, **params):
        """调用本地模型API"""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": params.get("temperature", 0.7),
            "max_tokens": params.get("max_tokens", -1),
            "stream": False
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=60
            )
            return response.json()
        except Exception as e:
            raise RuntimeError(f"模型调用失败: {str(e)}")

    def _parse_response(self, response):
        """解析模型输出"""
        try:
            content = response['choices'][0]['message']['content']
            # 尝试提取JSON（如果模型返回结构化数据）
            if "{" in content and "}" in content:
                start = content.index("{")
                end = content.rindex("}") + 1
                return json.loads(content[start:end])
            return content
        except (KeyError, json.JSONDecodeError):
            return response

# 使用示例
if __name__ == "__main__":
    client = GemmaLocalClient()
    
    while True:
        try:
            image_path = input("请输入图片路径: ").strip()
            if image_path == "q":
                break
            if not image_path:
                continue
                
            result = client.analyze_image(
                image_path=image_path,
                user_prompt="请分析图中的动物，用JSON格式回答，包含动物类别和位置信息"
            )
            
            print("\n分析结果:")
            if isinstance(result, dict):
                print(f"动物类别: {result.get('类别', '未知')}")
                print(f"位置信息: {result.get('位置', '未识别')}")
            else:
                print(result)
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"错误: {str(e)}")
