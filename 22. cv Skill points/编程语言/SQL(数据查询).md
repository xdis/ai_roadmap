# SQL 数据查询基础

SQL（Structured Query Language，结构化查询语言）是一种专门用于管理和查询关系型数据库的标准化语言。无论是数据分析、机器学习还是人工智能项目，掌握SQL都是操作和提取数据的基础技能。

## 1. SQL基础概念

SQL主要用于：
- **查询数据**：从数据库获取数据
- **插入数据**：向数据库添加新数据
- **更新数据**：修改数据库中现有的数据
- **删除数据**：从数据库中删除不需要的数据
- **创建/修改/删除**：管理数据库结构

## 2. 基本查询语法

### SELECT语句
最基本的SQL查询命令，用于从数据库中检索数据。

```sql
-- 查询表中所有列
SELECT * FROM 表名;

-- 查询特定列
SELECT 列名1, 列名2 FROM 表名;

-- 示例：查询学生表中所有学生的姓名和年龄
SELECT 姓名, 年龄 FROM 学生表;
```

### WHERE子句
用于过滤记录，只返回满足特定条件的记录。

```sql
SELECT 列名1, 列名2 FROM 表名 WHERE 条件;

-- 示例：查询年龄大于20岁的学生
SELECT * FROM 学生表 WHERE 年龄 > 20;

-- 多条件查询（AND和OR）
SELECT * FROM 学生表 WHERE 年龄 > 20 AND 性别 = '男';
SELECT * FROM 学生表 WHERE 年龄 > 25 OR 成绩 > 90;
```

### ORDER BY子句
用于对结果集进行排序。

```sql
SELECT 列名 FROM 表名 ORDER BY 列名 [ASC|DESC];

-- 示例：按照年龄升序排列学生
SELECT * FROM 学生表 ORDER BY 年龄 ASC;

-- 多字段排序（先按年龄降序，相同年龄按成绩升序）
SELECT * FROM 学生表 ORDER BY 年龄 DESC, 成绩 ASC;
```

### LIMIT子句
限制返回结果的数量。

```sql
SELECT 列名 FROM 表名 LIMIT 数量;

-- 示例：只返回前5名学生
SELECT * FROM 学生表 LIMIT 5;

-- 分页查询（跳过前10条，返回接下来的5条）
SELECT * FROM 学生表 LIMIT 5 OFFSET 10;
```

## 3. 高级查询

### 聚合函数
用于对数据进行计算。

```sql
-- 常用聚合函数
SELECT COUNT(*) FROM 学生表;  -- 计算总记录数
SELECT AVG(成绩) FROM 学生表;  -- 计算平均成绩
SELECT MAX(成绩) FROM 学生表;  -- 查找最高成绩
SELECT MIN(成绩) FROM 学生表;  -- 查找最低成绩
SELECT SUM(成绩) FROM 学生表;  -- 计算成绩总和
```

### GROUP BY子句
对数据进行分组，通常与聚合函数一起使用。

```sql
SELECT 分组列, 聚合函数(列名) FROM 表名 GROUP BY 分组列;

-- 示例：按性别分组，计算各性别的平均成绩
SELECT 性别, AVG(成绩) as 平均成绩 FROM 学生表 GROUP BY 性别;

-- 示例：按班级分组，统计各班级人数和平均成绩
SELECT 班级, COUNT(*) as 人数, AVG(成绩) as 平均成绩 
FROM 学生表 GROUP BY 班级;
```

### HAVING子句
用于过滤分组后的结果。

```sql
SELECT 分组列, 聚合函数(列名) 
FROM 表名 
GROUP BY 分组列 
HAVING 条件;

-- 示例：查询平均成绩大于80的班级
SELECT 班级, AVG(成绩) as 平均成绩 
FROM 学生表 
GROUP BY 班级 
HAVING AVG(成绩) > 80;
```

## 4. 连接查询

在实际应用中，数据通常分布在多个相关表中，需要使用连接(JOIN)来组合数据。

### 内连接(INNER JOIN)
只返回两表中匹配的行。

```sql
SELECT 表1.列名, 表2.列名 
FROM 表1 
INNER JOIN 表2 ON 表1.关联列 = 表2.关联列;

-- 示例：连接学生表和课程表，查询每个学生选择的课程
SELECT 学生表.姓名, 课程表.课程名 
FROM 学生表 
INNER JOIN 选课表 ON 学生表.学号 = 选课表.学号
INNER JOIN 课程表 ON 选课表.课程号 = 课程表.课程号;
```

### 左连接(LEFT JOIN)
返回左表中所有行，即使右表中没有匹配。

```sql
SELECT 表1.列名, 表2.列名 
FROM 表1 
LEFT JOIN 表2 ON 表1.关联列 = 表2.关联列;

-- 示例：查询所有学生及其选课情况（包括没选课的学生）
SELECT 学生表.姓名, 课程表.课程名 
FROM 学生表 
LEFT JOIN 选课表 ON 学生表.学号 = 选课表.学号
LEFT JOIN 课程表 ON 选课表.课程号 = 课程表.课程号;
```

### 右连接(RIGHT JOIN)
返回右表中所有行，即使左表中没有匹配。

```sql
SELECT 表1.列名, 表2.列名 
FROM 表1 
RIGHT JOIN 表2 ON 表1.关联列 = 表2.关联列;
```

## 5. 子查询

在一个SQL语句内部嵌套另一个SQL语句。

```sql
SELECT 列名 FROM 表名 WHERE 列名 操作符 (SELECT 列名 FROM 表名 WHERE 条件);

-- 示例：查询成绩高于平均成绩的学生
SELECT * FROM 学生表 
WHERE 成绩 > (SELECT AVG(成绩) FROM 学生表);

-- 示例：查询选修了"数据库"课程的学生
SELECT * FROM 学生表 
WHERE 学号 IN (
    SELECT 学号 FROM 选课表 
    WHERE 课程号 = (
        SELECT 课程号 FROM 课程表 WHERE 课程名 = '数据库'
    )
);
```

## 6. 数据操作语言(DML)

### 插入数据(INSERT)

```sql
-- 插入一条记录
INSERT INTO 表名 (列1, 列2, ...) VALUES (值1, 值2, ...);

-- 示例：向学生表添加一条记录
INSERT INTO 学生表 (学号, 姓名, 年龄, 性别) 
VALUES ('2021001', '张三', 20, '男');
```

### 更新数据(UPDATE)

```sql
-- 更新记录
UPDATE 表名 SET 列1 = 值1, 列2 = 值2 WHERE 条件;

-- 示例：更新张三的年龄
UPDATE 学生表 SET 年龄 = 21 WHERE 姓名 = '张三';
```

### 删除数据(DELETE)

```sql
-- 删除记录
DELETE FROM 表名 WHERE 条件;

-- 示例：删除学号为2021001的学生记录
DELETE FROM 学生表 WHERE 学号 = '2021001';
```

## 7. 实际应用示例

### 数据分析示例

```sql
-- 计算各个年龄段学生的平均成绩
SELECT 
    CASE 
        WHEN 年龄 < 20 THEN '20岁以下'
        WHEN 年龄 BETWEEN 20 AND 22 THEN '20-22岁'
        ELSE '22岁以上'
    END AS 年龄段,
    COUNT(*) as 人数,
    AVG(成绩) as 平均成绩
FROM 学生表
GROUP BY 年龄段
ORDER BY 平均成绩 DESC;

-- 查找成绩前三名的学生
SELECT 姓名, 成绩
FROM 学生表
ORDER BY 成绩 DESC
LIMIT 3;
```

### 机器学习数据准备示例

```sql
-- 选择特定特征和目标变量
SELECT 年龄, 性别, 学习时长, 上课出勤率, 成绩 
FROM 学生表
WHERE 成绩 IS NOT NULL;

-- 创建训练集和测试集
SELECT * FROM 学生表 
ORDER BY RANDOM() 
LIMIT (SELECT COUNT(*)*0.8 FROM 学生表);  -- 80%作为训练集
```

## 8. SQL优化技巧

1. **使用索引**：对经常查询的列创建索引可以提高查询速度
2. **避免SELECT ***：只选择需要的列，减少数据传输量
3. **使用EXPLAIN**：分析SQL语句的执行计划，找出性能瓶颈
4. **限制结果集大小**：使用LIMIT控制返回的数据量
5. **优化JOIN操作**：确保连接列已建立索引，减少表连接次数

## 总结

SQL是数据查询和操作的强大工具，掌握这些基础语法和技巧将帮助你有效地进行数据提取和分析。在数据科学和机器学习项目中，熟练使用SQL可以大大提高数据准备的效率。随着经验的增长，你可以进一步探索更高级的SQL功能，如窗口函数、公共表表达式(CTE)等。