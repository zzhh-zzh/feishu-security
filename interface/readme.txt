1、输入格式
文本输入要求：python 指定文件 emotion --text "内容"
图像输入要求：python 指定文件 animal --image-url "内容"

2、输出格式
生成一个result.json
自备模型只会返回一个type的键值对
大模型会生成更为细致的结果

2、文件
大模型运行文件为llm_api.py
自备模型运行文件为predict.py
自备模型可以进行初步判断，而大模型可以提供更为细致的判断（取消了只有3分类的限制）

3、安装依赖库
pip install -r requirements.txt