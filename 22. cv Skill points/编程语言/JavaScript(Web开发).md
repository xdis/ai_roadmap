# JavaScript基础（Web开发）

JavaScript是Web开发中不可或缺的编程语言，它允许你在网页上创建动态交互效果。以下内容将帮助你理解JavaScript的基础知识和在Web开发中的应用。

## 1. JavaScript基础语法

### 变量和数据类型

```javascript
// 变量声明
let name = "张三";     // 字符串类型
const age = 25;       // 数字类型
var isStudent = true; // 布尔类型

// 现代JavaScript推荐使用let和const而不是var
// let: 可以重新赋值的变量
// const: 一旦赋值不能改变的常量

// 数据类型
let stringExample = "这是一个字符串";
let numberExample = 42;
let booleanExample = false;
let arrayExample = [1, 2, 3, 4];
let objectExample = { name: "李四", age: 30 };
let undefinedExample;  // 未定义
let nullExample = null;  // 空值
```

### 基本运算符

```javascript
// 算术运算符
let sum = 10 + 5;      // 15
let difference = 10 - 5; // 5
let product = 10 * 5;   // 50
let quotient = 10 / 5;  // 2

// 比较运算符
let isEqual = 10 == "10";  // true (值相等)
let isStrictEqual = 10 === "10"; // false (值和类型都要相等)
let isGreater = 10 > 5;    // true

// 逻辑运算符
let andResult = true && false; // false
let orResult = true || false;  // true
let notResult = !true;        // false
```

## 2. 函数和控制流

### 函数定义与调用

```javascript
// 函数声明
function greet(name) {
  return "你好，" + name + "!";
}

// 函数调用
let greeting = greet("王五");  // "你好，王五!"

// 箭头函数（ES6特性）
const multiply = (a, b) => a * b;
console.log(multiply(3, 4));  // 12
```

### 条件语句

```javascript
// if-else语句
let hour = 14;

if (hour < 12) {
  console.log("上午好!");
} else if (hour < 18) {
  console.log("下午好!");
} else {
  console.log("晚上好!");
}

// switch语句
let day = "周一";

switch (day) {
  case "周一":
    console.log("开始新的一周!");
    break;
  case "周五":
    console.log("周末快到了!");
    break;
  default:
    console.log("普通工作日");
}
```

### 循环

```javascript
// for循环
for (let i = 0; i < 5; i++) {
  console.log("循环次数: " + i);
}

// while循环
let count = 0;
while (count < 3) {
  console.log("while计数: " + count);
  count++;
}

// forEach (数组方法)
const fruits = ["苹果", "香蕉", "橙子"];
fruits.forEach(function(fruit) {
  console.log("水果: " + fruit);
});
```

## 3. DOM操作（网页交互）

DOM (Document Object Model) 是JavaScript操作网页元素的接口。

### 选择元素

```javascript
// 通过ID选择元素
const mainTitle = document.getElementById("main-title");

// 通过类名选择元素（返回集合）
const paragraphs = document.getElementsByClassName("paragraph");

// 通过CSS选择器选择元素
const button = document.querySelector("#submit-button");
const allLinks = document.querySelectorAll("a.external-link");
```

### 修改元素

```javascript
// 修改内容
document.getElementById("result").textContent = "计算结果: 42";
document.getElementById("welcome").innerHTML = "<strong>欢迎光临!</strong>";

// 修改样式
const errorMessage = document.getElementById("error");
errorMessage.style.color = "red";
errorMessage.style.fontSize = "14px";

// 添加/删除类
const infoBox = document.querySelector(".info-box");
infoBox.classList.add("highlighted");
infoBox.classList.remove("hidden");
infoBox.classList.toggle("expanded");  // 切换类（有则删，无则加）
```

### 事件处理

```javascript
// 点击事件
const clickButton = document.getElementById("click-me");
clickButton.addEventListener("click", function() {
  alert("按钮被点击了!");
});

// 表单提交事件
const form = document.getElementById("signup-form");
form.addEventListener("submit", function(event) {
  event.preventDefault();  // 阻止表单默认提交行为
  
  // 获取表单数据
  const username = document.getElementById("username").value;
  
  if (username.length < 3) {
    alert("用户名太短!");
  } else {
    alert("表单提交成功!");
    form.submit();
  }
});
```

## 4. 异步JavaScript

### Promise和fetch API

```javascript
// 使用fetch获取数据（现代AJAX）
fetch("https://api.example.com/data")
  .then(response => {
    if (!response.ok) {
      throw new Error("网络错误");
    }
    return response.json();  // 解析JSON响应
  })
  .then(data => {
    console.log("获取的数据:", data);
    displayData(data);  // 处理数据
  })
  .catch(error => {
    console.error("出错了:", error);
    showErrorMessage();
  });
```

### async/await (更现代的异步处理)

```javascript
// 使用async/await简化异步代码
async function fetchUserData(userId) {
  try {
    // await暂停执行，直到Promise解决
    const response = await fetch(`https://api.example.com/users/${userId}`);
    
    if (!response.ok) {
      throw new Error("获取用户数据失败");
    }
    
    const userData = await response.json();
    return userData;
  } catch (error) {
    console.error("发生错误:", error);
    return null;
  }
}

// 调用异步函数
async function displayUserProfile() {
  const userData = await fetchUserData(123);
  
  if (userData) {
    document.getElementById("user-name").textContent = userData.name;
    document.getElementById("user-email").textContent = userData.email;
  }
}
```

## 5. 实用例子：简单计数器应用

### HTML部分
```html
<div class="counter">
  <h2>简单计数器</h2>
  <p>当前计数: <span id="count">0</span></p>
  <button id="decrease">-</button>
  <button id="reset">重置</button>
  <button id="increase">+</button>
</div>
```

### JavaScript部分
```javascript
// 获取DOM元素
const countElement = document.getElementById("count");
const decreaseBtn = document.getElementById("decrease");
const resetBtn = document.getElementById("reset");
const increaseBtn = document.getElementById("increase");

// 初始计数值
let count = 0;

// 更新显示函数
function updateDisplay() {
  countElement.textContent = count;
  
  // 根据数值设置颜色
  if (count < 0) {
    countElement.style.color = "red";
  } else if (count > 0) {
    countElement.style.color = "green";
  } else {
    countElement.style.color = "black";
  }
}

// 添加事件监听器
decreaseBtn.addEventListener("click", function() {
  count--;
  updateDisplay();
});

resetBtn.addEventListener("click", function() {
  count = 0;
  updateDisplay();
});

increaseBtn.addEventListener("click", function() {
  count++;
  updateDisplay();
});

// 初始化显示
updateDisplay();
```

## 6. 现代JavaScript特性

### 模板字符串

```javascript
const name = "小明";
const score = 95;

// 使用反引号和${...}进行字符串插值
const message = `${name}的考试成绩是${score}分，${score >= 90 ? '优秀' : '良好'}!`;
console.log(message);  // "小明的考试成绩是95分，优秀!"
```

### 解构赋值

```javascript
// 数组解构
const coordinates = [10, 20, 30];
const [x, y, z] = coordinates;
console.log(x, y, z);  // 10 20 30

// 对象解构
const person = { 
  fullName: "张三丰", 
  age: 30,
  city: "北京"
};
const { fullName, city } = person;
console.log(fullName, city);  // "张三丰" "北京"
```

### 展开运算符

```javascript
// 数组合并
const fruits = ["苹果", "香蕉"];
const moreFruits = ["橙子", "葡萄"];
const allFruits = [...fruits, ...moreFruits];
console.log(allFruits);  // ["苹果", "香蕉", "橙子", "葡萄"]

// 对象合并
const baseConfig = { theme: "dark", fontSize: 16 };
const userConfig = { fontSize: 18, showSidebar: true };
const finalConfig = { ...baseConfig, ...userConfig };
console.log(finalConfig);  
// { theme: "dark", fontSize: 18, showSidebar: true }
```

## 总结

JavaScript是Web开发中的核心语言，通过它你可以：
1. 操作网页元素(DOM)
2. 处理用户交互(事件)
3. 动态更新内容
4. 与服务器通信(AJAX/fetch)
5. 构建交互式Web应用

掌握这些基础知识后，你可以进一步学习JavaScript框架(如React, Vue, Angular)，它们提供了更高效的Web应用开发方式。