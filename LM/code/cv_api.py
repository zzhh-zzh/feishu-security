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
            Role: 图像动物识别专家
        Profile:
        language: 中文
        description: 专门识别图像中的动物类别和位置信息
        Goals:
        准确识别图像中的动物类别
        精确定位动物在图像中的位置
        输出标准化的识别结果
        Constrains:
        必须输出动物类别和位置框信息
        位置框格式必须为[起始x,起始y,宽,高]
        需要描述识别依据和分析过程
        Skills:
        精通计算机视觉和图像识别
        擅长物体检测和定位
        熟悉常见动物特征识别
        能够处理复杂背景下的动物识别
        Workflows:
        接收用户输入的图像
        分析图像内容，识别可能的动物区域
        确定动物类别和精确位置
        计算位置框坐标[起始x,起始y,宽,高]
        生成详细的分析报告
        按照标准格式输出结果
        OutputFormat:
        ""
        输入图像分析结果：

        识别依据：

        特征识别：
        位置分析：
        最终结果：
        {
        "类别": "动物类别",
        "位置框": [起始x, 起始y, 宽, 高]
        }
        ""
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
