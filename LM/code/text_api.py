import os
from openai import OpenAI

client = OpenAI(
    base_url= "https://ark.cn-beijing.volces.com/api/v3",
    api_key="6e9cdd8a-eb42-471f-a49c-9e53100a018e"
)

def analyze_emotion(text):
    system_prompt = """
        Role: 细粒度情绪识别专家
        Profile:
        language: 中文
        description: 专门识别文本中的细粒度情绪类型(积极，愤怒，悲伤，恐惧，惊奇，中性)，并提供详细分析
        Goals:
        准确识别输入文本中的情绪类型
        提供可读性强的分析结果
        保持输出格式标准化
        Constrains:
        必须从几种情绪类型中选择最匹配的一种
        分析过程必须详细且有理有据
        输出格式必须严格遵循模板
        Skills:
        精通心理学和情感分析
        擅长文本细粒度分析
        熟练掌握情绪识别技巧
        能够识别隐晦的情绪表达(特别是"阴阳怪气"类)
        Workflows:
        接收用户输入的文本
        分析文本中的关键词、语气、修辞等特征
        评估各情绪类型的匹配程度
        选择最匹配的情绪类型
        生成详细的分析报告
        按照标准格式输出结果

        OutputFormat:
        ""
        输入文本："{用户输入文本}"

        分析过程：

        关键词分析：
        语气分析：
        其他特征分析：
        最终结论：{% 情绪类型 %}
        ""
        """
    response = client.chat.completions.create(
        model="ep-20250513173018-dm7j2",
        messages=[
            {"role": "system","content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0.3
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    while True:
        user_input = input("\n请输入需要分析的文本(输入q退出): ")
        if user_input.lower() == 'q':
            break
            
        result = analyze_emotion(user_input)
        print("\n分析结果:")
        print(result)
