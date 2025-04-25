# Python 核心概念

Python 是一种易学易用的高级编程语言，广泛应用于数据科学、人工智能、Web 开发等领域。下面将介绍 Python 的核心概念，配合简洁的代码示例帮助你理解。

## 1. 基本数据类型

Python 有几种内置的基本数据类型：

### 1.1 数值类型
```python
# 整数
age = 25

# 浮点数
price = 19.99

# 复数
complex_num = 3 + 4j

# 布尔值
is_valid = True
```

### 1.2 字符串
```python
# 基本字符串
name = "Python"

# 多行字符串
description = """Python 是一种高级编程语言，
它的设计哲学强调代码的可读性和简洁的语法"""

# 字符串操作
print(name.upper())  # 转大写: PYTHON
print(name.lower())  # 转小写: python
print(len(name))     # 长度: 6
print(name[0])       # 索引: P
print(name[0:2])     # 切片: Py

# 格式化字符串
language = "Python"
version = 3.9
print(f"{language} {version} 是当前的稳定版本")  # Python 3.9 是当前的稳定版本
```

### 1.3 列表（可变序列）
```python
# 创建列表
fruits = ["苹果", "香蕉", "橙子"]

# 访问元素
print(fruits[0])       # 苹果
print(fruits[-1])      # 橙子 (负索引从末尾开始)

# 修改列表
fruits.append("草莓")   # 添加元素
fruits[1] = "蓝莓"     # 修改元素
fruits.remove("橙子")   # 删除元素
fruits.sort()          # 排序

# 列表推导式
squares = [x**2 for x in range(5)]  # [0, 1, 4, 9, 16]
```

### 1.4 元组（不可变序列）
```python
# 创建元组
coordinates = (10, 20)

# 访问元素
x, y = coordinates  # 解包
print(x)            # 10

# 元组不可修改，但可以拼接
new_coord = coordinates + (30,)  # (10, 20, 30)
```

### 1.5 字典（键值对）
```python
# 创建字典
person = {
    "name": "张三",
    "age": 30,
    "city": "北京"
}

# 访问和修改
print(person["name"])   # 张三
person["age"] = 31      # 修改年龄
person["job"] = "工程师" # 添加新键值对

# 字典方法
keys = person.keys()     # 获取所有键
values = person.values() # 获取所有值
items = person.items()   # 获取所有键值对

# 字典推导式
squares_dict = {x: x**2 for x in range(5)}  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

### 1.6 集合（无序不重复元素集）
```python
# 创建集合
numbers = {1, 2, 3, 4, 5}
duplicate_numbers = {1, 2, 2, 3, 4, 4, 5}  # 自动去重: {1, 2, 3, 4, 5}

# 集合操作
numbers.add(6)           # 添加元素
numbers.remove(1)        # 删除元素

# 集合运算
set_a = {1, 2, 3, 4}
set_b = {3, 4, 5, 6}
print(set_a.union(set_b))        # 并集: {1, 2, 3, 4, 5, 6}
print(set_a.intersection(set_b)) # 交集: {3, 4}
print(set_a.difference(set_b))   # 差集: {1, 2}
```

## 2. 控制流

### 2.1 条件语句
```python
age = 20

if age < 18:
    print("未成年")
elif age < 65:
    print("成年人")
else:
    print("老年人")

# 三元表达式
status = "成年" if age >= 18 else "未成年"
```

### 2.2 循环
```python
# for 循环
for i in range(5):
    print(i)  # 打印 0, 1, 2, 3, 4

# 遍历列表
fruits = ["苹果", "香蕉", "橙子"]
for fruit in fruits:
    print(fruit)

# 遍历字典
person = {"name": "张三", "age": 30}
for key, value in person.items():
    print(f"{key}: {value}")

# while 循环
count = 0
while count < 5:
    print(count)
    count += 1

# break 和 continue
for i in range(10):
    if i == 3:
        continue  # 跳过当前迭代
    if i == 7:
        break     # 结束循环
    print(i)
```

## 3. 函数

### 3.1 定义和调用函数
```python
# 基本函数
def greet(name):
    return f"你好，{name}！"

message = greet("王五")
print(message)  # 你好，王五！

# 默认参数
def greet_with_time(name, time="早上"):
    return f"{time}好，{name}！"

print(greet_with_time("李四"))        # 早上好，李四！
print(greet_with_time("张三", "晚上")) # 晚上好，张三！

# 可变参数
def sum_all(*numbers):
    return sum(numbers)

print(sum_all(1, 2, 3, 4))  # 10

# 关键字参数
def build_profile(**user_info):
    return user_info

profile = build_profile(name="张三", age=30, city="上海")
print(profile)  # {'name': '张三', 'age': 30, 'city': '上海'}
```

### 3.2 Lambda 函数（匿名函数）
```python
# 普通函数
def double(x):
    return x * 2

# 等价的 lambda 函数
double_lambda = lambda x: x * 2

print(double(5))        # 10
print(double_lambda(5)) # 10

# 结合使用 lambda 和内置函数
numbers = [1, 2, 3, 4, 5]
squares = list(map(lambda x: x**2, numbers))  # [1, 4, 9, 16, 25]
evens = list(filter(lambda x: x % 2 == 0, numbers))  # [2, 4]
```

## 4. 类和对象（面向对象编程）

```python
class Person:
    # 类属性
    species = "人类"
    
    # 初始化方法（构造函数）
    def __init__(self, name, age):
        # 实例属性
        self.name = name
        self.age = age
        
    # 实例方法
    def introduce(self):
        return f"我叫{self.name}，今年{self.age}岁"
    
    # 类方法
    @classmethod
    def create_anonymous(cls):
        return cls("匿名", 0)
    
    # 静态方法
    @staticmethod
    def is_adult(age):
        return age >= 18

# 创建对象
person1 = Person("张三", 30)
print(person1.introduce())  # 我叫张三，今年30岁
print(Person.species)       # 人类

# 调用类方法
anonymous = Person.create_anonymous()
print(anonymous.name)  # 匿名

# 调用静态方法
print(Person.is_adult(20))  # True

# 继承
class Student(Person):
    def __init__(self, name, age, student_id):
        # 调用父类的初始化方法
        super().__init__(name, age)
        self.student_id = student_id
    
    # 重写父类方法
    def introduce(self):
        return f"{super().introduce()}，学号是{self.student_id}"

student = Student("李四", 18, "S12345")
print(student.introduce())  # 我叫李四，今年18岁，学号是S12345
```

## 5. 模块和包

### 5.1 使用内置模块
```python
# 导入整个模块
import math
print(math.sqrt(16))  # 4.0

# 导入特定函数
from random import randint
print(randint(1, 10))  # 1到10的随机整数

# 导入并重命名
import datetime as dt
now = dt.datetime.now()
print(now)
```

### 5.2 创建自己的模块
假设我们有一个文件 `mymath.py`：

```python
# mymath.py
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
```

然后在另一个文件中使用：

```python
# 导入自定义模块
import mymath
print(mymath.add(5, 3))      # 8
print(mymath.multiply(4, 2)) # 8

# 或者选择性导入
from mymath import add
print(add(5, 3))  # 8
```

### 5.3 包
包是一种组织相关模块的方式，它是一个包含 `__init__.py` 文件的目录：

```
my_package/
    __init__.py
    module1.py
    module2.py
```

使用方式：
```python
# 导入包中的模块
from my_package import module1
module1.function()

# 直接导入函数
from my_package.module2 import some_function
some_function()
```

## 6. 异常处理

```python
# 基本异常处理
try:
    number = int(input("请输入一个数字: "))
    result = 10 / number
    print(f"结果是 {result}")
except ValueError:
    print("输入无效，请输入一个数字")
except ZeroDivisionError:
    print("不能除以零")
except Exception as e:
    print(f"发生错误: {e}")
else:
    print("没有发生异常")
finally:
    print("无论是否发生异常，这里都会执行")

# 抛出异常
def validate_age(age):
    if age < 0:
        raise ValueError("年龄不能为负数")
    if age > 120:
        raise ValueError("年龄不太可能超过120")
    return age

# 自定义异常
class CustomError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

try:
    raise CustomError("这是一个自定义错误")
except CustomError as e:
    print(e)
```

## 7. 文件操作

```python
# 读取文件
try:
    with open("example.txt", "r", encoding="utf-8") as file:
        content = file.read()
        print(content)
except FileNotFoundError:
    print("文件不存在")

# 写入文件
with open("new_file.txt", "w", encoding="utf-8") as file:
    file.write("这是第一行\n")
    file.write("这是第二行\n")

# 追加到文件
with open("new_file.txt", "a", encoding="utf-8") as file:
    file.write("这是追加的一行\n")

# 逐行读取大文件
with open("large_file.txt", "r", encoding="utf-8") as file:
    for line in file:
        print(line.strip())
```

## 8. 列表、字典和集合的高级操作

### 8.1 列表高级操作
```python
# 切片操作
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(numbers[2:7])     # [2, 3, 4, 5, 6]
print(numbers[::2])     # [0, 2, 4, 6, 8] - 步长为2
print(numbers[::-1])    # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0] - 反转列表

# 列表解析
numbers = [1, 2, 3, 4, 5]
squares = [x**2 for x in numbers]  # [1, 4, 9, 16, 25]
even_squares = [x**2 for x in numbers if x % 2 == 0]  # [4, 16]

# 合并列表
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined = list1 + list2  # [1, 2, 3, 4, 5, 6]

# 展平嵌套列表
nested = [[1, 2], [3, 4], [5, 6]]
flat = [item for sublist in nested for item in sublist]  # [1, 2, 3, 4, 5, 6]
```

### 8.2 字典高级操作
```python
# 合并字典
dict1 = {"a": 1, "b": 2}
dict2 = {"b": 3, "c": 4}
# Python 3.9+
merged = dict1 | dict2  # {'a': 1, 'b': 3, 'c': 4}
# 其他版本
merged = {**dict1, **dict2}  # {'a': 1, 'b': 3, 'c': 4}

# 字典推导式
users = ["Alice", "Bob", "Charlie"]
user_scores = {user: len(user) for user in users}  # {'Alice': 5, 'Bob': 3, 'Charlie': 7}

# 默认字典
from collections import defaultdict
fruit_counts = defaultdict(int)  # 默认值为0
fruits = ["苹果", "香蕉", "苹果", "橙子", "香蕉", "苹果"]
for fruit in fruits:
    fruit_counts[fruit] += 1
print(dict(fruit_counts))  # {'苹果': 3, '香蕉': 2, '橙子': 1}
```

## 9. 生成器和迭代器

```python
# 生成器函数
def countdown(n):
    while n > 0:
        yield n
        n -= 1

# 使用生成器
for i in countdown(5):
    print(i)  # 打印 5, 4, 3, 2, 1

# 生成器表达式 (类似列表推导式但使用圆括号)
squares_gen = (x**2 for x in range(1, 6))
for square in squares_gen:
    print(square)  # 打印 1, 4, 9, 16, 25

# 无限序列生成器
def infinite_sequence():
    num = 0
    while True:
        yield num
        num += 1

# 取前10个数
gen = infinite_sequence()
for _ in range(10):
    print(next(gen))  # 打印 0 到 9
```

## 10. 实用的标准库

```python
# 日期和时间
import datetime
now = datetime.datetime.now()
print(now)                      # 2023-04-23 14:30:45.123456
print(now.strftime("%Y-%m-%d")) # 2023-04-23

# JSON 处理
import json
data = {
    "name": "张三",
    "age": 30,
    "skills": ["Python", "数据分析"]
}
# 转为 JSON 字符串
json_str = json.dumps(data, ensure_ascii=False)
print(json_str)
# 解析 JSON 字符串
parsed_data = json.loads(json_str)
print(parsed_data["name"])  # 张三

# 正则表达式
import re
text = "我的电话号码是 13912345678 和 020-87654321"
phones = re.findall(r'\b1[3-9]\d{9}\b|\d{3,4}-\d{7,8}', text)
print(phones)  # ['13912345678', '020-87654321']

# 随机数
import random
print(random.randint(1, 10))           # 1到10的随机整数
print(random.choice(["红", "绿", "蓝"])) # 随机选择列表中的一个元素
random.shuffle(numbers)                # 打乱列表顺序

# 集合运算
import itertools
# 排列
perms = list(itertools.permutations([1, 2, 3], 2))  # [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]
# 组合
combs = list(itertools.combinations([1, 2, 3, 4], 2))  # [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
```

## 11. 导入常用第三方库

```python
# NumPy - 科学计算
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(arr.mean())  # 3.0
print(arr * 2)     # [2 4 6 8 10]

# Pandas - 数据分析
import pandas as pd
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['Beijing', 'Shanghai', 'Guangzhou']
})
print(df.head())
print(df.describe())  # 统计摘要

# Matplotlib - 数据可视化
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.xlabel('x 轴')
plt.ylabel('y 轴')
plt.title('简单线图')
plt.show()

# Requests - HTTP请求
import requests
response = requests.get('https://api.example.com/data')
if response.status_code == 200:
    data = response.json()
    print(data)
```

## 12. Python 编程最佳实践

1. **命名约定**：
   - 变量和函数名使用小写字母和下划线（snake_case）
   - 类名使用驼峰命名法（CamelCase）
   - 常量使用大写字母和下划线

2. **编码风格**：
   - 遵循 PEP 8 编码规范
   - 使用 4 个空格进行缩进（不要使用制表符）
   - 行长度不超过 79 个字符

3. **文档注释**：
```python
def calculate_area(radius):
    """
    计算圆的面积。
    
    参数:
        radius (float): 圆的半径
        
    返回:
        float: 圆的面积
    """
    import math
    return math.pi * radius ** 2
```

4. **使用上下文管理器**：
```python
# 好的做法
with open('file.txt', 'r') as file:
    content = file.read()
    
# 而不是
file = open('file.txt', 'r')
content = file.read()
file.close()  # 容易忘记关闭
```

5. **使用列表推导式而不是 for 循环（当合适时）**：
```python
# 好的做法
squares = [x**2 for x in range(10)]

# 而不是
squares = []
for x in range(10):
    squares.append(x**2)
```

---

以上就是 Python 核心概念的简要介绍，包含了基础语法、数据结构、面向对象编程、异常处理等内容。这些基础知识是进行数据科学、机器学习、Web 开发等更高级应用的基石。