# Next.js 入门指南

Next.js 是一个基于 React 的轻量级框架，它让你能够构建服务器端渲染（SSR）和静态网站生成（SSG）的 React 应用。它由 Vercel 开发和维护，特别适合创建高性能、SEO 友好的网站和应用程序。

## 核心特性

1. **服务器端渲染 (SSR)** - 在服务器上预渲染页面，提高首屏加载速度
2. **静态网站生成 (SSG)** - 构建时预渲染页面，极快的加载速度
3. **增量静态再生 (ISR)** - 结合静态生成和服务器渲染的优点
4. **文件系统路由** - 基于文件结构自动生成路由
5. **API 路由** - 轻松创建 API 端点
6. **内置 CSS 和 Sass 支持** - 无需额外配置
7. **零配置** - 开箱即用

## 快速上手

### 安装与创建项目

```bash
# 创建新项目
npx create-next-app my-next-app
# 或者使用 yarn
yarn create next-app my-next-app

# 进入项目目录
cd my-next-app

# 启动开发服务器
npm run dev
# 或者
yarn dev
```

### 文件结构

```
my-next-app/
├── pages/             # 页面目录，文件名即路由
│   ├── index.js       # 首页 (/)
│   ├── about.js       # 关于页 (/about)
│   └── api/           # API 路由目录
│       └── hello.js   # API 端点 (/api/hello)
├── public/            # 静态资源目录
├── styles/            # 样式文件
├── components/        # React 组件
├── next.config.js     # Next.js 配置文件
└── package.json       # 项目依赖
```

## 基本页面示例

### 基础页面组件

```jsx
// pages/index.js - 首页
import Head from 'next/head'

export default function Home() {
  return (
    <div>
      <Head>
        <title>我的 Next.js 网站</title>
        <meta name="description" content="使用 Next.js 构建的网站" />
      </Head>

      <main>
        <h1>欢迎来到我的 Next.js 网站!</h1>
        <p>这是一个使用 Next.js 构建的简单示例。</p>
      </main>
    </div>
  )
}
```

### 页面导航

```jsx
// pages/about.js - 关于页面
import Link from 'next/link'

export default function About() {
  return (
    <div>
      <h1>关于我们</h1>
      <p>这是关于页面的内容。</p>
      
      {/* 使用 Link 组件进行客户端导航 */}
      <Link href="/">
        <a>返回首页</a>
      </Link>
    </div>
  )
}
```

## 数据获取方法

Next.js 提供了三种获取数据的方法：

### 1. getStaticProps (静态生成)

```jsx
// pages/posts.js
export default function Posts({ posts }) {
  return (
    <div>
      <h1>博客文章列表</h1>
      <ul>
        {posts.map((post) => (
          <li key={post.id}>{post.title}</li>
        ))}
      </ul>
    </div>
  )
}

// 在构建时获取数据
export async function getStaticProps() {
  // 这里可以是从 API 或数据库获取数据
  const res = await fetch('https://jsonplaceholder.typicode.com/posts')
  const posts = await res.json()

  return {
    props: {
      posts: posts.slice(0, 5), // 只取前5篇文章
    },
  }
}
```

### 2. getServerSideProps (服务器端渲染)

```jsx
// pages/dashboard.js
export default function Dashboard({ data }) {
  return (
    <div>
      <h1>仪表盘</h1>
      <p>当前时间: {data.time}</p>
    </div>
  )
}

// 每次请求时在服务器端执行
export async function getServerSideProps() {
  return {
    props: {
      data: {
        time: new Date().toISOString(),
      },
    },
  }
}
```

### 3. 客户端数据获取 (使用 SWR)

```jsx
// pages/profile.js
import useSWR from 'swr'

// 数据获取函数
const fetcher = url => fetch(url).then(res => res.json())

export default function Profile() {
  const { data, error } = useSWR('/api/user', fetcher)

  if (error) return <div>加载失败</div>
  if (!data) return <div>加载中...</div>

  return (
    <div>
      <h1>用户资料</h1>
      <p>用户名: {data.name}</p>
    </div>
  )
}
```

## API 路由

Next.js 允许你在 `pages/api` 目录下创建 API 端点：

```jsx
// pages/api/hello.js
export default function handler(req, res) {
  // 处理不同的 HTTP 方法
  if (req.method === 'POST') {
    // 处理 POST 请求
    res.status(200).json({ message: '数据已接收' })
  } else {
    // 处理 GET 请求等
    res.status(200).json({ name: '张三', age: 25 })
  }
}
```

## 动态路由

Next.js 支持动态路由，让你可以创建参数化的页面：

```jsx
// pages/posts/[id].js
import { useRouter } from 'next/router'

export default function Post({ post }) {
  const router = useRouter()
  
  // 如果页面还在预渲染，显示一个加载状态
  if (router.isFallback) {
    return <div>加载中...</div>
  }

  return (
    <div>
      <h1>{post.title}</h1>
      <p>{post.body}</p>
    </div>
  )
}

// 指定哪些路径需要被静态生成
export async function getStaticPaths() {
  const res = await fetch('https://jsonplaceholder.typicode.com/posts')
  const posts = await res.json()
  
  // 为前5篇文章生成路径
  const paths = posts.slice(0, 5).map((post) => ({
    params: { id: post.id.toString() },
  }))
  
  return { paths, fallback: true }
}

// 为每个路径获取数据
export async function getStaticProps({ params }) {
  const res = await fetch(`https://jsonplaceholder.typicode.com/posts/${params.id}`)
  const post = await res.json()
  
  return {
    props: { post },
    revalidate: 60, // 每60秒重新验证数据
  }
}
```

## CSS 样式

Next.js 支持多种样式方案：

### 1. 全局 CSS

```jsx
// pages/_app.js
import '../styles/globals.css'

function MyApp({ Component, pageProps }) {
  return <Component {...pageProps} />
}

export default MyApp
```

### 2. CSS 模块 (推荐)

```jsx
// components/Button.js
import styles from './Button.module.css'

export default function Button({ children }) {
  return (
    <button className={styles.button}>
      {children}
    </button>
  )
}

// Button.module.css
.button {
  padding: 10px 15px;
  background-color: blue;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}
```

### 3. CSS-in-JS (使用 styled-jsx)

```jsx
export default function Card() {
  return (
    <div className="card">
      <h2>标题</h2>
      <p>内容描述</p>
      <style jsx>{`
        .card {
          border: 1px solid #eaeaea;
          border-radius: 8px;
          padding: 16px;
          margin: 16px 0;
        }
        h2 {
          margin-top: 0;
          color: #333;
        }
      `}</style>
    </div>
  )
}
```

## 部署

Next.js 应用可以部署到任何支持 Node.js 的环境，但最简单的方法是使用 Vercel 平台：

```bash
# 安装 Vercel CLI
npm install -g vercel

# 部署
vercel
```

也可以构建静态版本并部署到任何静态网站托管服务：

```bash
# 构建静态版本
npm run build
npm run export

# 静态文件将生成在 out 目录中
```

## 总结

Next.js 通过其强大的功能和简单的 API，使得构建高性能的 React 应用变得更加容易。它适合各种类型的项目，从简单的静态网站到复杂的应用程序。通过它的混合渲染能力，你可以为每个页面选择最合适的渲染方式，获得最佳的性能和用户体验。