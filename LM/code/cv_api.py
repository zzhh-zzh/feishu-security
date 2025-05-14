import os
from openai import OpenAI

# 初始化客户端
client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key="6e9cdd8a-eb42-471f-a49c-9e53100a018e"
)

def analyze_animal(image_url):
    """
    调用大模型API进行动物识别和定位
    :param image_url: 需要分析的图片URL
    :return: 结构化分析结果
    """

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
    
    response = client.chat.completions.create(
        model="ep-20250513173018-dm7j2",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    },
                    {"type": "text", "text": "请分析这张图片中的动物"}
                ]
            }
        ],
        temperature=0.1  # 使用更低的随机性确保坐标准确
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    while True:
        user_input = input("\n请输入图片URL(输入q退出): ")
        if user_input.lower() == 'q':
            break
            
        if not user_input.startswith(('http://', 'https://')):
            print("请输入有效的图片URL(以http://或https://开头)")
            continue
            
        try:
            result = analyze_animal(user_input)
            print("\n分析结果:")
            print(result)
        except Exception as e:
            print(f"分析出错: {str(e)}")
