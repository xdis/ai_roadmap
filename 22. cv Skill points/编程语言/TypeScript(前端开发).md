# TypeScript 前端开发基础

TypeScript 是 JavaScript 的超集，由微软开发，它为 JavaScript 添加了静态类型和其他高级功能，使得代码更可靠、更易于维护。本文档将介绍 TypeScript 在前端开发中的基础知识和实际应用。

## 1. 为什么使用 TypeScript？

TypeScript 相比纯 JavaScript 有以下优势：

- **静态类型检查**：可以在编译时发现错误，而不是运行时
- **更好的开发工具支持**：智能提示、代码导航、重构等
- **更好的代码组织**：接口、枚举、命名空间等
- **更好的大型项目维护性**：类型定义使代码更清晰
- **与现代框架的良好集成**：如React、Vue、Angular等

## 2. TypeScript 基础类型

### 基本类型注解

```typescript
// 基本类型
let isDone: boolean = false;
let decimal: number = 6;
let color: string = "blue";
let list: number[] = [1, 2, 3];
let x: [string, number] = ["hello", 10]; // 元组

// 枚举
enum Color {Red, Green, Blue}
let c: Color = Color.Green;

// Any 类型（当你不确定类型时使用）
let notSure: any = 4;
notSure = "maybe a string";
notSure = false;

// Void 类型（通常用于没有返回值的函数）
function warnUser(): void {
    console.log("This is a warning message");
}

// Null 和 Undefined
let u: undefined = undefined;
let n: null = null;

// Never 类型（表示永远不会发生的值）
function error(message: string): never {
    throw new Error(message);
}
```

## 3. 接口和类型

### 接口 (Interface)

接口可以描述对象的形状：

```typescript
// 定义接口
interface User {
    id: number;
    name: string;
    email?: string; // 可选属性
    readonly createdAt: Date; // 只读属性
}

// 使用接口
function createUser(user: User): void {
    console.log(`创建用户: ${user.name}`);
}

createUser({
    id: 1,
    name: "张三",
    createdAt: new Date()
});
```

### 类型别名 (Type Alias)

与接口类似，但更灵活：

```typescript
// 类型别名
type Point = {
    x: number;
    y: number;
};

// 联合类型
type ID = number | string;

// 字面量类型
type Direction = 'north' | 'south' | 'east' | 'west';
```

## 4. 函数

### 函数类型注解

```typescript
// 函数参数和返回值类型
function add(x: number, y: number): number {
    return x + y;
}

// 可选参数
function buildName(firstName: string, lastName?: string): string {
    return lastName ? firstName + " " + lastName : firstName;
}

// 默认参数
function greeting(name: string = "Guest"): string {
    return `Hello, ${name}!`;
}

// 剩余参数
function sum(...numbers: number[]): number {
    return numbers.reduce((total, n) => total + n, 0);
}

// 函数类型
let myAdd: (x: number, y: number) => number = add;
```

## 5. 类

### 基本类用法

```typescript
class Person {
    // 属性
    name: string;
    private age: number;
    protected readonly id: number;
    
    // 构造函数
    constructor(name: string, age: number, id: number) {
        this.name = name;
        this.age = age;
        this.id = id;
    }
    
    // 方法
    greet(): string {
        return `Hello, my name is ${this.name} and I am ${this.age} years old.`;
    }
    
    // Getter
    get personAge(): number {
        return this.age;
    }
    
    // Setter
    set personAge(newAge: number) {
        if (newAge >= 0 && newAge < 120) {
            this.age = newAge;
        } else {
            throw new Error("Invalid age value");
        }
    }
}

// 类的实例化
const person = new Person("李四", 25, 101);
console.log(person.greet());
```

### 继承

```typescript
class Employee extends Person {
    department: string;
    
    constructor(name: string, age: number, id: number, department: string) {
        super(name, age, id); // 调用父类构造函数
        this.department = department;
    }
    
    // 覆盖父类方法
    greet(): string {
        return `${super.greet()} I work in ${this.department}.`;
    }
}

const employee = new Employee("王五", 30, 102, "工程部");
console.log(employee.greet());
```

## 6. 泛型

泛型允许我们创建可重用的组件：

```typescript
// 泛型函数
function identity<T>(arg: T): T {
    return arg;
}

let output1 = identity<string>("myString");
let output2 = identity(42); // 类型推断为 number

// 泛型接口
interface GenericIdentityFn<T> {
    (arg: T): T;
}

// 泛型类
class GenericBox<T> {
    private content: T;
    
    constructor(value: T) {
        this.content = value;
    }
    
    getContent(): T {
        return this.content;
    }
}

const numberBox = new GenericBox<number>(123);
const stringBox = new GenericBox("Hello TypeScript");
```

## 7. 装饰器

装饰器是一种特殊类型的声明，可以附加到类、方法、访问器、属性或参数上：

```typescript
// 启用装饰器（需要在tsconfig.json中设置）
// 类装饰器
function Logger(target: Function) {
    console.log(`Class ${target.name} is decorated`);
}

@Logger
class Example {
    // 属性装饰器
    @format
    title: string = "TypeScript Example";
    
    // 方法装饰器
    @log
    greet() {
        return "Hello from decorated method";
    }
}

function format(target: any, propertyKey: string) {
    let value = target[propertyKey];
    
    const getter = function() {
        return value;
    };
    
    const setter = function(newVal: string) {
        value = newVal.toUpperCase();
    };
    
    Object.defineProperty(target, propertyKey, {
        get: getter,
        set: setter
    });
}

function log(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    const originalMethod = descriptor.value;
    
    descriptor.value = function(...args: any[]) {
        console.log(`调用方法: ${propertyKey}`);
        return originalMethod.apply(this, args);
    };
    
    return descriptor;
}
```

## 8. 模块和命名空间

### 模块

```typescript
// math.ts
export function add(x: number, y: number): number {
    return x + y;
}

export function subtract(x: number, y: number): number {
    return x - y;
}

// app.ts
import { add, subtract } from './math';
console.log(add(5, 3));
```

### 命名空间

```typescript
// Validation.ts
namespace Validation {
    export interface StringValidator {
        isValid(s: string): boolean;
    }
    
    export class EmailValidator implements StringValidator {
        isValid(s: string): boolean {
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            return emailRegex.test(s);
        }
    }
}

// 使用命名空间
let emailValidator = new Validation.EmailValidator();
console.log(emailValidator.isValid("test@example.com"));
```

## 9. 实际应用实例：创建一个简单的 Todo 应用

下面是一个结合 TypeScript 和 React 的简单 Todo 应用示例：

```typescript
// Todo.tsx
import React, { useState } from 'react';

// 定义Todo项的接口
interface TodoItem {
    id: number;
    text: string;
    completed: boolean;
}

const TodoApp: React.FC = () => {
    // 状态管理
    const [todos, setTodos] = useState<TodoItem[]>([]);
    const [input, setInput] = useState<string>('');
    
    // 添加Todo
    const handleAddTodo = (): void => {
        if (input.trim() !== '') {
            const newTodo: TodoItem = {
                id: Date.now(),
                text: input,
                completed: false
            };
            setTodos([...todos, newTodo]);
            setInput('');
        }
    };
    
    // 切换完成状态
    const toggleTodo = (id: number): void => {
        setTodos(todos.map(todo => 
            todo.id === id ? { ...todo, completed: !todo.completed } : todo
        ));
    };
    
    // 删除Todo
    const removeTodo = (id: number): void => {
        setTodos(todos.filter(todo => todo.id !== id));
    };
    
    return (
        <div>
            <h1>TypeScript Todo App</h1>
            <div>
                <input 
                    type="text" 
                    value={input} 
                    onChange={(e) => setInput(e.target.value)} 
                    placeholder="Add a task" 
                />
                <button onClick={handleAddTodo}>Add</button>
            </div>
            <ul>
                {todos.map(todo => (
                    <li key={todo.id} style={{ textDecoration: todo.completed ? 'line-through' : 'none' }}>
                        <span onClick={() => toggleTodo(todo.id)}>{todo.text}</span>
                        <button onClick={() => removeTodo(todo.id)}>Delete</button>
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default TodoApp;
```

## 10. TypeScript 配置

TypeScript 项目通常需要一个 `tsconfig.json` 文件来配置编译选项：

```json
{
  "compilerOptions": {
    "target": "es6",           // 指定ECMAScript目标版本
    "module": "commonjs",      // 指定模块代码生成
    "lib": ["dom", "es6"],     // 指定要包含的库文件
    "jsx": "react",            // 支持JSX
    "sourceMap": true,         // 生成相应的 .map 文件
    "outDir": "./dist",        // 输出目录
    "strict": true,            // 启用所有严格类型检查选项
    "noImplicitAny": true,     // 禁止隐含的 any 类型
    "esModuleInterop": true    // 支持CommonJS和ES模块之间的互操作性
  },
  "include": [
    "src/**/*"                 // 需要编译的文件
  ],
  "exclude": [
    "node_modules",            // 排除的文件
    "**/*.test.ts"
  ]
}
```

## 总结

TypeScript 为前端开发提供了很多好处，特别是在大型项目中。通过静态类型检查，它可以帮助我们捕获潜在的错误，提高代码质量和可维护性。本文档介绍了 TypeScript 的基础知识和实际应用，希望能够帮助你开始使用 TypeScript 进行前端开发。