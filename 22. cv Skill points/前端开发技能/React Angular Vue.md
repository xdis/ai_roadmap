# React, Angular, Vue 前端框架介绍

这三个框架是当前最流行的前端JavaScript框架，它们各有特点，适合不同的项目需求和开发风格。

## React

React是由Facebook开发的JavaScript库，用于构建用户界面。它以组件化开发为核心理念，使用虚拟DOM提高性能。

### React的特点

- **组件化**: 将UI拆分为独立、可复用的组件
- **虚拟DOM**: 提高渲染性能
- **单向数据流**: 数据从父组件流向子组件，使应用状态可预测
- **JSX语法**: 允许在JavaScript中编写类似HTML的代码

### React基础代码示例

#### 简单组件

```jsx
// 函数式组件
function Welcome(props) {
  return <h1>你好, {props.name}</h1>;
}

// 使用组件
const element = <Welcome name="小明" />;
```

#### 带状态的组件

```jsx
import React, { useState } from 'react';

function Counter() {
  // 声明一个名为"count"的state变量，初始值为0
  const [count, setCount] = useState(0);
  
  return (
    <div>
      <p>你点击了 {count} 次</p>
      <button onClick={() => setCount(count + 1)}>
        点我增加
      </button>
    </div>
  );
}
```

#### 简单的React应用

```jsx
import React from 'react';
import ReactDOM from 'react-dom';

function App() {
  return (
    <div>
      <h1>我的待办事项</h1>
      <ul>
        <li>学习React</li>
        <li>创建组件</li>
        <li>学习状态管理</li>
      </ul>
    </div>
  );
}

ReactDOM.render(<App />, document.getElementById('root'));
```

## Angular

Angular是Google维护的完整前端框架，提供了从开发到测试的完整解决方案。它使用TypeScript，提供了强类型体验。

### Angular的特点

- **完整框架**: 包含路由、表单、HTTP客户端等
- **TypeScript**: 提供类型安全和更好的开发体验
- **双向数据绑定**: 模型和视图自动同步
- **依赖注入**: 提高可测试性和模块化
- **模块化架构**: 通过NgModule组织代码

### Angular基础代码示例

#### 组件定义

```typescript
import { Component } from '@angular/core';

@Component({
  selector: 'app-hello',
  template: `<h1>你好，{{name}}!</h1>`,
  styles: [`h1 { color: blue; }`]
})
export class HelloComponent {
  name = '小明';
}
```

#### 带用户交互的组件

```typescript
import { Component } from '@angular/core';

@Component({
  selector: 'app-counter',
  template: `
    <p>你点击了 {{count}} 次</p>
    <button (click)="increment()">点我增加</button>
  `
})
export class CounterComponent {
  count = 0;
  
  increment() {
    this.count++;
  }
}
```

#### 简单的Angular应用

```typescript
// app.module.ts
import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { AppComponent } from './app.component';

@NgModule({
  imports: [BrowserModule],
  declarations: [AppComponent],
  bootstrap: [AppComponent]
})
export class AppModule { }

// app.component.ts
import { Component } from '@angular/core';

@Component({
  selector: 'app-root',
  template: `
    <h1>我的待办事项</h1>
    <ul>
      <li *ngFor="let task of tasks">{{task}}</li>
    </ul>
  `
})
export class AppComponent {
  tasks = ['学习Angular', '创建组件', '学习服务'];
}
```

## Vue

Vue是一个渐进式JavaScript框架，可以逐步采用。它结合了React和Angular的一些最佳特性，易学易用。

### Vue的特点

- **渐进式框架**: 可以只使用核心库，也可以添加各种插件
- **响应式系统**: 自动追踪依赖关系并更新视图
- **模板语法**: 类似HTML的直观模板
- **组件化**: 构建可复用的UI组件
- **轻量级**: 核心库非常小，易于学习

### Vue基础代码示例

#### 基础组件

```html
<div id="app">
  <h1>{{ message }}</h1>
</div>

<script>
const app = new Vue({
  el: '#app',
  data: {
    message: '你好，Vue!'
  }
})
</script>
```

#### 带用户交互的组件

```html
<div id="counter">
  <p>你点击了 {{ count }} 次</p>
  <button @click="increment">点我增加</button>
</div>

<script>
new Vue({
  el: '#counter',
  data: {
    count: 0
  },
  methods: {
    increment() {
      this.count++;
    }
  }
})
</script>
```

#### 使用Vue 3 Composition API

```html
<div id="app"></div>

<script>
import { createApp, ref } from 'vue';

const App = {
  setup() {
    const count = ref(0);
    
    function increment() {
      count.value++;
    }
    
    return {
      count,
      increment
    };
  },
  template: `
    <p>你点击了 {{ count }} 次</p>
    <button @click="increment">点我增加</button>
  `
};

createApp(App).mount('#app');
</script>
```

#### 简单的Vue应用

```html
<div id="todo-app">
  <h1>我的待办事项</h1>
  <ul>
    <li v-for="task in tasks" :key="task">{{ task }}</li>
  </ul>
</div>

<script>
new Vue({
  el: '#todo-app',
  data: {
    tasks: ['学习Vue', '创建组件', '学习状态管理']
  }
})
</script>
```

## 三大框架比较

| 特性 | React | Angular | Vue |
|------|-------|---------|-----|
| 学习曲线 | 中等 | 较陡 | 平缓 |
| 性能 | 很好 | 好 | 很好 |
| 生态系统 | 非常丰富 | 丰富 | 丰富 |
| 灵活性 | 高 | 中 | 高 |
| 状态管理 | Redux, MobX等 | NGRX, Services | Vuex, Pinia |
| 公司支持 | Facebook | Google | 独立团队 |
| 适用项目 | 各种规模 | 中大型企业应用 | 各种规模 |

## 选择哪个框架？

- **React**: 如果你喜欢灵活性，想要构建大型应用，并且喜欢JavaScript的编程方式
- **Angular**: 如果你喜欢完整的框架体验，喜欢TypeScript，适合大型企业应用
- **Vue**: 如果你想要简单易学的框架，喜欢模板和清晰的项目结构

无论选择哪个框架，它们都能帮助你构建现代、响应式的前端应用。初学者可以从Vue开始，因为它的学习曲线最平缓。