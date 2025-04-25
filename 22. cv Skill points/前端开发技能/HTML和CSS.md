# HTML 和 CSS 基础

## HTML 简介

HTML（超文本标记语言）是网页的基础，定义了网页的结构和内容。

### HTML 文档基本结构

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>我的第一个网页</title>
</head>
<body>
    <h1>这是一个标题</h1>
    <p>这是一个段落。</p>
</body>
</html>
```

### 常用 HTML 标签

1. **标题标签**：`<h1>` 到 `<h6>`，表示不同级别的标题

```html
<h1>一级标题</h1>
<h2>二级标题</h2>
<h3>三级标题</h3>
```

2. **段落标签**：`<p>` 用于创建段落

```html
<p>这是第一个段落</p>
<p>这是第二个段落</p>
```

3. **链接标签**：`<a>` 用于创建超链接

```html
<a href="https://www.example.com">访问示例网站</a>
```

4. **图像标签**：`<img>` 用于插入图片

```html
<img src="图片.jpg" alt="图片描述" width="200" height="150">
```

5. **列表标签**：
   - 无序列表：`<ul>` 和 `<li>`
   - 有序列表：`<ol>` 和 `<li>`

```html
<!-- 无序列表 -->
<ul>
    <li>苹果</li>
    <li>香蕉</li>
    <li>橙子</li>
</ul>

<!-- 有序列表 -->
<ol>
    <li>第一步</li>
    <li>第二步</li>
    <li>第三步</li>
</ol>
```

6. **表格标签**：`<table>`, `<tr>`, `<th>`, `<td>`

```html
<table border="1">
    <tr>
        <th>姓名</th>
        <th>年龄</th>
    </tr>
    <tr>
        <td>张三</td>
        <td>25</td>
    </tr>
    <tr>
        <td>李四</td>
        <td>30</td>
    </tr>
</table>
```

7. **表单标签**：`<form>`, `<input>`, `<button>`

```html
<form action="/submit" method="post">
    <label for="username">用户名：</label>
    <input type="text" id="username" name="username"><br><br>
    
    <label for="password">密码：</label>
    <input type="password" id="password" name="password"><br><br>
    
    <input type="radio" id="male" name="gender" value="male">
    <label for="male">男</label>
    
    <input type="radio" id="female" name="gender" value="female">
    <label for="female">女</label><br><br>
    
    <input type="checkbox" id="bike" name="vehicle" value="bike">
    <label for="bike">我有一辆自行车</label><br>
    
    <button type="submit">提交</button>
</form>
```

8. **区块标签**：`<div>` 和 `<span>`
   - `<div>` 是块级元素，通常用于组织内容
   - `<span>` 是内联元素，通常用于修饰文本

```html
<div>这是一个div区块</div>
<p>这是一个段落，包含<span style="color:red;">红色</span>文本</p>
```

## CSS 简介

CSS（层叠样式表）用于控制网页的外观和布局。

### CSS 使用方式

1. **内联样式**：直接在 HTML 标签内使用 `style` 属性

```html
<p style="color: blue; font-size: 18px;">这是蓝色的段落文本</p>
```

2. **内部样式表**：在 HTML 头部使用 `<style>` 标签

```html
<head>
    <style>
        p {
            color: blue;
            font-size: 18px;
        }
    </style>
</head>
```

3. **外部样式表**：单独创建 CSS 文件，通过 `<link>` 标签引入

```html
<head>
    <link rel="stylesheet" href="styles.css">
</head>
```

### CSS 选择器

1. **元素选择器**：选择所有特定类型的元素

```css
p {
    color: blue;
}
```

2. **类选择器**：选择特定类的元素

```css
.highlight {
    background-color: yellow;
}
```

HTML 使用：`<p class="highlight">高亮显示的文本</p>`

3. **ID 选择器**：选择特定 ID 的元素

```css
#header {
    background-color: black;
    color: white;
}
```

HTML 使用：`<div id="header">页眉内容</div>`

4. **组合选择器**：

```css
/* 选择 div 中的所有 p 元素 */
div p {
    margin-left: 20px;
}

/* 选择所有类为 .box 的元素中的 p 元素 */
.box p {
    font-weight: bold;
}
```

### CSS 常用属性

1. **文本样式**

```css
p {
    color: #333;              /* 文本颜色 */
    font-family: Arial, sans-serif; /* 字体 */
    font-size: 16px;          /* 字体大小 */
    font-weight: bold;        /* 字体粗细 */
    text-align: center;       /* 文本对齐 */
    line-height: 1.5;         /* 行高 */
    text-decoration: underline; /* 文本装饰 */
}
```

2. **背景样式**

```css
div {
    background-color: #f0f0f0;    /* 背景颜色 */
    background-image: url('bg.jpg'); /* 背景图片 */
    background-repeat: no-repeat;  /* 背景不重复 */
    background-position: center;   /* 背景位置 */
}
```

3. **盒模型**：每个元素都被视为一个盒子，包含内容、内边距、边框和外边距

```css
div {
    width: 300px;             /* 宽度 */
    height: 200px;            /* 高度 */
    padding: 20px;            /* 内边距 */
    border: 1px solid black;  /* 边框 */
    margin: 10px;             /* 外边距 */
}
```

4. **边框样式**

```css
div {
    border-width: 2px;        /* 边框宽度 */
    border-style: solid;      /* 边框样式 */
    border-color: red;        /* 边框颜色 */
    border-radius: 10px;      /* 圆角 */
}
```

5. **显示与定位**

```css
div {
    display: block;           /* 显示方式 */
    position: relative;       /* 定位方式 */
    top: 10px;                /* 上偏移 */
    left: 20px;               /* 左偏移 */
    z-index: 1;               /* 层级 */
}
```

## 响应式设计

响应式设计让网页能够适应不同设备的屏幕大小。

### 媒体查询

```css
/* 大屏幕（桌面电脑） */
@media (min-width: 1200px) {
    .container {
        width: 1170px;
    }
}

/* 中等屏幕（平板电脑） */
@media (min-width: 768px) and (max-width: 1199px) {
    .container {
        width: 750px;
    }
}

/* 小屏幕（手机） */
@media (max-width: 767px) {
    .container {
        width: 100%;
    }
}
```

## 实际例子：简单的导航栏

### HTML 部分
```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>导航栏示例</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <nav class="navbar">
        <div class="logo">网站名称</div>
        <ul class="nav-links">
            <li><a href="#">首页</a></li>
            <li><a href="#">关于</a></li>
            <li><a href="#">服务</a></li>
            <li><a href="#">联系我们</a></li>
        </ul>
    </nav>
    
    <div class="content">
        <h1>欢迎访问我们的网站</h1>
        <p>这是一个简单的网页示例，展示了导航栏的实现。</p>
    </div>
</body>
</html>
```

### CSS 部分
```css
/* 重置默认样式 */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: Arial, sans-serif;
    line-height: 1.6;
}

/* 导航栏样式 */
.navbar {
    background-color: #333;
    color: white;
    display: flex;
    justify-content: space-between;
    padding: 15px 20px;
    align-items: center;
}

.logo {
    font-size: 1.5rem;
    font-weight: bold;
}

.nav-links {
    display: flex;
    list-style: none;
}

.nav-links li {
    margin-left: 20px;
}

.nav-links a {
    color: white;
    text-decoration: none;
    transition: color 0.3s;
}

.nav-links a:hover {
    color: #ffd700;
}

/* 内容样式 */
.content {
    padding: 20px;
    max-width: 1200px;
    margin: 0 auto;
}

/* 响应式设计 */
@media (max-width: 768px) {
    .navbar {
        flex-direction: column;
    }
    
    .nav-links {
        margin-top: 10px;
        width: 100%;
        justify-content: space-between;
    }
    
    .nav-links li {
        margin-left: 0;
    }
}
```

## 学习建议

1. 先掌握 HTML 基础结构和常用标签
2. 学习 CSS 选择器和常用属性
3. 理解盒模型和布局方式
4. 学习响应式设计
5. 实践：尝试制作简单的页面，如个人简历或博客

通过实践和不断学习，你将能够创建更加复杂和精美的网页。