# SQL数据库基础指南 (MySQL, PostgreSQL等)

SQL数据库是最常用的关系型数据库管理系统(RDBMS)，它们使用结构化查询语言(SQL)来管理数据。下面我将介绍SQL数据库的基础知识及常见操作，主要以MySQL和PostgreSQL为例。

## 1. 关系型数据库基本概念

关系型数据库将数据组织成表、行和列的结构：
- **表(Table)**: 存储特定类型数据的集合
- **行(Row)**: 表中的一条记录
- **列(Column)**: 表中的一个字段
- **主键(Primary Key)**: 唯一标识表中每条记录的字段
- **外键(Foreign Key)**: 用于建立表之间关系的字段

## 2. 数据库操作基础

### 2.1 连接数据库

**MySQL连接:**
```bash
mysql -u username -p
# 然后输入密码
```

**PostgreSQL连接:**
```bash
psql -U username -d database_name
# 然后输入密码
```

### 2.2 创建数据库

```sql
-- 创建新数据库
CREATE DATABASE my_database;

-- 使用数据库
USE my_database;  -- MySQL
\c my_database    -- PostgreSQL命令行
```

## 3. 表操作

### 3.1 创建表

```sql
CREATE TABLE employees (
    employee_id INT PRIMARY KEY AUTO_INCREMENT,  -- MySQL自增
    -- 在PostgreSQL中使用: employee_id SERIAL PRIMARY KEY
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE,
    hire_date DATE,
    salary DECIMAL(10, 2),
    department_id INT,
    FOREIGN KEY (department_id) REFERENCES departments(department_id)
);
```

### 3.2 修改表结构

```sql
-- 添加新列
ALTER TABLE employees ADD COLUMN phone VARCHAR(15);

-- 修改列
ALTER TABLE employees MODIFY COLUMN phone VARCHAR(20);  -- MySQL
ALTER TABLE employees ALTER COLUMN phone TYPE VARCHAR(20);  -- PostgreSQL

-- 删除列
ALTER TABLE employees DROP COLUMN phone;
```

### 3.3 删除表

```sql
DROP TABLE IF EXISTS employees;
```

## 4. 数据操作

### 4.1 插入数据

```sql
-- 单行插入
INSERT INTO employees (first_name, last_name, email, hire_date, salary, department_id)
VALUES ('John', 'Doe', 'john.doe@example.com', '2023-01-15', 50000.00, 1);

-- 多行插入
INSERT INTO employees (first_name, last_name, email, hire_date, salary, department_id)
VALUES 
    ('Jane', 'Smith', 'jane.smith@example.com', '2023-02-20', 55000.00, 1),
    ('Mike', 'Johnson', 'mike.johnson@example.com', '2023-03-10', 52000.00, 2);
```

### 4.2 查询数据

```sql
-- 基本查询
SELECT * FROM employees;

-- 选择特定列
SELECT first_name, last_name, salary FROM employees;

-- 条件查询
SELECT * FROM employees WHERE department_id = 1;

-- 排序
SELECT * FROM employees ORDER BY salary DESC;

-- 限制结果数量
SELECT * FROM employees LIMIT 5;  -- MySQL和PostgreSQL
```

### 4.3 更新数据

```sql
-- 更新单个字段
UPDATE employees SET salary = 58000.00 WHERE employee_id = 1;

-- 更新多个字段
UPDATE employees 
SET salary = 60000.00, email = 'john.new@example.com' 
WHERE employee_id = 1;
```

### 4.4 删除数据

```sql
-- 删除特定记录
DELETE FROM employees WHERE employee_id = 3;

-- 删除所有记录
DELETE FROM employees;
-- 或者更高效的方式(不保留自增序列):
TRUNCATE TABLE employees;
```

## 5. 高级查询

### 5.1 JOIN操作

```sql
-- 内连接(INNER JOIN)：获取两个表中满足连接条件的记录
SELECT e.first_name, e.last_name, d.department_name
FROM employees e
INNER JOIN departments d ON e.department_id = d.department_id;

-- 左连接(LEFT JOIN)：获取左表所有记录，右表匹配的记录
SELECT e.first_name, e.last_name, d.department_name
FROM employees e
LEFT JOIN departments d ON e.department_id = d.department_id;

-- 右连接(RIGHT JOIN)：获取右表所有记录，左表匹配的记录
SELECT e.first_name, e.last_name, d.department_name
FROM employees e
RIGHT JOIN departments d ON e.department_id = d.department_id;
```

### 5.2 聚合函数

```sql
-- 计数
SELECT COUNT(*) FROM employees;

-- 求和
SELECT SUM(salary) FROM employees;

-- 平均值
SELECT AVG(salary) FROM employees;

-- 最大值和最小值
SELECT MAX(salary), MIN(salary) FROM employees;

-- 分组统计
SELECT department_id, AVG(salary) as avg_salary
FROM employees
GROUP BY department_id;

-- 筛选分组
SELECT department_id, AVG(salary) as avg_salary
FROM employees
GROUP BY department_id
HAVING AVG(salary) > 50000;
```

### 5.3 子查询

```sql
-- 子查询作为条件
SELECT * FROM employees 
WHERE salary > (SELECT AVG(salary) FROM employees);

-- 子查询作为计算字段
SELECT 
    first_name,
    last_name,
    salary,
    (SELECT AVG(salary) FROM employees) as avg_company_salary
FROM employees;
```

## 6. MySQL和PostgreSQL的主要区别

| 特性 | MySQL | PostgreSQL |
|------|-------|------------|
| 自增字段 | `AUTO_INCREMENT` | `SERIAL` |
| 类型 | `INT`, `VARCHAR` | 支持更多高级类型，如数组、JSON |
| 事务支持 | InnoDB引擎支持 | 全面支持 |
| 存储过程 | 有限支持 | 强大支持(PL/pgSQL) |
| 全文搜索 | 基本支持 | 更强大(集成全文搜索) |
| 扩展 | 插件系统 | 丰富的扩展系统 |

## 7. 实际应用案例

### 7.1 创建完整的博客数据库

```sql
-- 创建用户表
CREATE TABLE users (
    user_id INT PRIMARY KEY AUTO_INCREMENT,  -- MySQL
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建文章表
CREATE TABLE posts (
    post_id INT PRIMARY KEY AUTO_INCREMENT,
    title VARCHAR(200) NOT NULL,
    content TEXT,
    user_id INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- 创建评论表
CREATE TABLE comments (
    comment_id INT PRIMARY KEY AUTO_INCREMENT,
    post_id INT NOT NULL,
    user_id INT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (post_id) REFERENCES posts(post_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- 插入测试数据
INSERT INTO users (username, email, password_hash) VALUES
('john_doe', 'john@example.com', 'hashed_password_1'),
('jane_smith', 'jane@example.com', 'hashed_password_2');

INSERT INTO posts (title, content, user_id) VALUES
('First Post', 'This is my first post content', 1),
('Learning SQL', 'SQL is a powerful language for databases', 2);

INSERT INTO comments (post_id, user_id, content) VALUES
(1, 2, 'Great first post!'),
(2, 1, 'I agree, SQL is amazing!');

-- 查询文章及其作者
SELECT p.title, p.content, u.username 
FROM posts p
JOIN users u ON p.user_id = u.user_id;

-- 查询文章、作者和评论数
SELECT 
    p.title, 
    u.username as author,
    COUNT(c.comment_id) as comment_count
FROM posts p
JOIN users u ON p.user_id = u.user_id
LEFT JOIN comments c ON p.post_id = c.post_id
GROUP BY p.post_id, p.title, u.username;
```

## 8. 数据库优化基础

### 8.1 索引优化

```sql
-- 创建索引
CREATE INDEX idx_employee_department ON employees(department_id);

-- 创建复合索引
CREATE INDEX idx_name ON employees(last_name, first_name);

-- 查看索引
SHOW INDEX FROM employees;  -- MySQL
\d employees                -- PostgreSQL

-- 删除索引
DROP INDEX idx_employee_department ON employees;
```

### 8.2 EXPLAIN查询分析

```sql
-- 分析查询执行计划
EXPLAIN SELECT * FROM employees WHERE department_id = 1;
```

## 9. 连接代码示例

### 9.1 使用Python连接MySQL

```python
import mysql.connector

# 建立连接
conn = mysql.connector.connect(
    host="localhost",
    user="username",
    password="password",
    database="my_database"
)

# 创建游标
cursor = conn.cursor()

# 执行查询
cursor.execute("SELECT first_name, last_name FROM employees WHERE department_id = %s", (1,))

# 获取结果
for (first_name, last_name) in cursor:
    print(f"{first_name} {last_name}")

# 插入数据
cursor.execute(
    "INSERT INTO employees (first_name, last_name, email, hire_date, salary, department_id) "
    "VALUES (%s, %s, %s, %s, %s, %s)",
    ("Alice", "Williams", "alice@example.com", "2023-04-15", 53000.00, 2)
)

# 提交事务
conn.commit()

# 关闭连接
cursor.close()
conn.close()
```

### 9.2 使用Python连接PostgreSQL

```python
import psycopg2

# 建立连接
conn = psycopg2.connect(
    host="localhost",
    database="my_database",
    user="username",
    password="password"
)

# 创建游标
cursor = conn.cursor()

# 执行查询
cursor.execute("SELECT first_name, last_name FROM employees WHERE department_id = %s", (1,))

# 获取结果
rows = cursor.fetchall()
for row in rows:
    print(f"{row[0]} {row[1]}")

# 关闭连接
cursor.close()
conn.close()
```

## 10. 总结

SQL数据库是企业应用中的核心组件。MySQL和PostgreSQL都是功能强大的关系型数据库系统：

- **MySQL** 易于使用，性能好，适合大多数Web应用
- **PostgreSQL** 更注重标准遵循，提供更高级的功能，适合复杂应用场景

无论选择哪种数据库，掌握基本的SQL操作都是至关重要的。通过本指南中的示例，你应该已经了解了如何创建表、执行基本的CRUD操作、进行高级查询以及如何从应用程序连接数据库。

随着你的进步，可以进一步学习事务处理、存储过程、触发器和更高级的优化技术。