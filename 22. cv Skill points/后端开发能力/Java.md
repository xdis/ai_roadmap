# Java 后端开发基础指南

Java 是一种广泛应用于企业级后端开发的编程语言，具有跨平台、面向对象和强类型的特点。本指南将介绍 Java 后端开发的基础知识和常用技术。

## 1. Java 基础

### 1.1 基本语法

```java
// 这是一个简单的 Java 程序
public class HelloWorld {
    public static void main(String[] args) {
        // 打印输出到控制台
        System.out.println("Hello, Backend Development!");
        
        // 变量声明和赋值
        String message = "Java is powerful";
        int number = 42;
        
        // 条件判断
        if (number > 40) {
            System.out.println(message);
        }
    }
}
```

**讲解**：
- `public class HelloWorld`：定义一个公共类，类名必须与文件名相同
- `public static void main(String[] args)`：程序入口方法
- `System.out.println()`：输出文本到控制台
- Java 是强类型语言，变量必须声明类型

### 1.2 面向对象编程

```java
// 定义一个简单的类
public class User {
    // 属性（成员变量）
    private int id;
    private String name;
    private String email;
    
    // 构造方法
    public User(int id, String name, String email) {
        this.id = id;
        this.name = name;
        this.email = email;
    }
    
    // Getter 和 Setter 方法
    public int getId() {
        return id;
    }
    
    public void setName(String name) {
        this.name = name;
    }
    
    // 普通方法
    public String getInfo() {
        return "User: " + name + " (ID: " + id + ")";
    }
}

// 使用示例
public class Main {
    public static void main(String[] args) {
        User user = new User(1, "张三", "zhangsan@example.com");
        System.out.println(user.getInfo());
        
        user.setName("李四");
        System.out.println(user.getInfo());
    }
}
```

**讲解**：
- 类（Class）是 Java 面向对象编程的基本单元
- 属性（成员变量）定义对象的数据
- 方法定义对象的行为
- 访问修饰符（如 `private`、`public`）控制访问权限
- 构造方法用于创建对象并初始化

## 2. Spring Boot 框架

Spring Boot 是当前最流行的 Java 后端框架，它大大简化了 Spring 应用的创建和开发过程。

### 2.1 创建 RESTful API

```java
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.*;

// 应用入口
@SpringBootApplication
public class Application {
    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}

// 控制器类
@RestController
@RequestMapping("/api/users")
public class UserController {
    
    // GET 请求 - 获取用户信息
    @GetMapping("/{id}")
    public User getUser(@PathVariable int id) {
        // 实际项目中，这里通常会从数据库查询
        return new User(id, "测试用户", "test@example.com");
    }
    
    // POST 请求 - 创建新用户
    @PostMapping
    public String createUser(@RequestBody User user) {
        // 实际项目中，这里会保存到数据库
        System.out.println("创建用户: " + user.getInfo());
        return "用户创建成功";
    }
    
    // PUT 请求 - 更新用户
    @PutMapping("/{id}")
    public String updateUser(@PathVariable int id, @RequestBody User user) {
        System.out.println("更新用户ID: " + id);
        return "用户更新成功";
    }
    
    // DELETE 请求 - 删除用户
    @DeleteMapping("/{id}")
    public String deleteUser(@PathVariable int id) {
        System.out.println("删除用户ID: " + id);
        return "用户删除成功";
    }
}
```

**讲解**：
- `@SpringBootApplication`: 表明这是一个 Spring Boot 应用
- `@RestController`: 标记这个类处理 HTTP 请求并返回 JSON 响应
- `@RequestMapping`: 定义 API 的基础 URL 路径
- `@GetMapping`, `@PostMapping` 等: 定义不同 HTTP 方法的处理函数
- `@PathVariable`: 获取 URL 路径中的参数
- `@RequestBody`: 将请求体转换为 Java 对象

### 2.2 依赖注入

```java
// 服务接口
public interface UserService {
    User getUserById(int id);
    void saveUser(User user);
}

// 服务实现
@Service
public class UserServiceImpl implements UserService {
    
    private final UserRepository userRepository;
    
    // 构造器注入
    @Autowired
    public UserServiceImpl(UserRepository userRepository) {
        this.userRepository = userRepository;
    }
    
    @Override
    public User getUserById(int id) {
        return userRepository.findById(id).orElse(null);
    }
    
    @Override
    public void saveUser(User user) {
        userRepository.save(user);
    }
}

// 在控制器中使用服务
@RestController
@RequestMapping("/api/users")
public class UserController {
    
    private final UserService userService;
    
    @Autowired
    public UserController(UserService userService) {
        this.userService = userService;
    }
    
    @GetMapping("/{id}")
    public User getUser(@PathVariable int id) {
        return userService.getUserById(id);
    }
    
    @PostMapping
    public String createUser(@RequestBody User user) {
        userService.saveUser(user);
        return "用户创建成功";
    }
}
```

**讲解**：
- 依赖注入是 Spring 的核心特性，帮助解耦组件
- `@Service`: 标记类为服务组件
- `@Autowired`: 自动装配依赖
- 构造器注入是推荐的依赖注入方式
- 接口和实现分离促进代码可维护性

## 3. 数据库交互 (JPA)

Spring Data JPA 简化了数据库操作。

```java
// 实体类
@Entity
@Table(name = "users")
public class User {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @Column(nullable = false)
    private String name;
    
    @Column(unique = true)
    private String email;
    
    @Column(name = "created_at")
    private LocalDateTime createdAt;
    
    // 构造器、getter和setter
}

// 仓库接口
public interface UserRepository extends JpaRepository<User, Long> {
    // Spring Data JPA 会自动实现这些方法
    List<User> findByName(String name);
    
    Optional<User> findByEmail(String email);
    
    @Query("SELECT u FROM User u WHERE u.createdAt > :date")
    List<User> findNewUsers(@Param("date") LocalDateTime date);
}
```

**讲解**：
- `@Entity`: 标记类为 JPA 实体
- `@Table`: 指定对应的表名
- `@Id`: 标记主键
- `@GeneratedValue`: 定义主键生成策略
- `@Column`: 定义列属性
- `JpaRepository`: 提供基本的 CRUD 操作
- 可以通过方法名称自动生成查询
- `@Query`: 可以自定义 JPQL 查询

## 4. 异常处理

```java
// 自定义异常
public class ResourceNotFoundException extends RuntimeException {
    public ResourceNotFoundException(String message) {
        super(message);
    }
}

// 全局异常处理
@RestControllerAdvice
public class GlobalExceptionHandler {
    
    @ExceptionHandler(ResourceNotFoundException.class)
    @ResponseStatus(HttpStatus.NOT_FOUND)
    public Map<String, String> handleResourceNotFoundException(ResourceNotFoundException ex) {
        Map<String, String> error = new HashMap<>();
        error.put("message", ex.getMessage());
        return error;
    }
    
    @ExceptionHandler(Exception.class)
    @ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)
    public Map<String, String> handleGlobalException(Exception ex) {
        Map<String, String> error = new HashMap<>();
        error.put("message", "发生了内部服务器错误");
        return error;
    }
}

// 在服务中使用
@Service
public class UserServiceImpl implements UserService {
    
    @Override
    public User getUserById(Long id) {
        return userRepository.findById(id)
            .orElseThrow(() -> new ResourceNotFoundException("未找到ID为 " + id + " 的用户"));
    }
}
```

**讲解**：
- 自定义异常类便于表达业务逻辑错误
- `@RestControllerAdvice`: 创建全局异常处理器
- `@ExceptionHandler`: 指定处理的异常类型
- `@ResponseStatus`: 设置 HTTP 状态码
- 规范化的异常处理提高 API 的可用性

## 5. 单元测试

```java
// 测试 Service 层
@ExtendWith(MockitoExtension.class)
public class UserServiceTest {
    
    @Mock
    private UserRepository userRepository;
    
    @InjectMocks
    private UserServiceImpl userService;
    
    @Test
    public void testGetUserById_Success() {
        // 准备测试数据
        User mockUser = new User();
        mockUser.setId(1L);
        mockUser.setName("测试用户");
        
        // 设置 Mock 行为
        when(userRepository.findById(1L)).thenReturn(Optional.of(mockUser));
        
        // 调用测试方法
        User result = userService.getUserById(1L);
        
        // 验证结果
        assertNotNull(result);
        assertEquals("测试用户", result.getName());
        
        // 验证方法被调用
        verify(userRepository).findById(1L);
    }
    
    @Test
    public void testGetUserById_NotFound() {
        // 设置 Mock 行为
        when(userRepository.findById(99L)).thenReturn(Optional.empty());
        
        // 验证抛出异常
        assertThrows(ResourceNotFoundException.class, () -> {
            userService.getUserById(99L);
        });
    }
}
```

**讲解**：
- JUnit 5 是 Java 常用的测试框架
- Mockito 用于模拟依赖
- `@Mock`: 创建模拟对象
- `@InjectMocks`: 将模拟对象注入到被测试类
- `when().thenReturn()`: 定义模拟行为
- `verify()`: 验证方法被调用
- 单元测试确保代码质量和可维护性

## 6. 项目结构

一个典型的 Spring Boot 项目结构：

```
src/
├── main/
│   ├── java/
│   │   └── com/example/demo/
│   │       ├── DemoApplication.java       # 应用入口
│   │       ├── controller/                # 控制器层
│   │       ├── service/                   # 服务层
│   │       ├── repository/                # 数据访问层
│   │       ├── model/                     # 实体/模型
│   │       ├── dto/                       # 数据传输对象
│   │       ├── exception/                 # 自定义异常
│   │       └── config/                    # 配置类
│   └── resources/
│       ├── application.properties         # 应用配置
│       ├── static/                        # 静态资源
│       └── templates/                     # 模板文件
└── test/                                  # 测试代码
```

## 总结

Java 后端开发的优势:
- 稳定性和成熟的生态系统
- 丰富的框架和库支持
- 良好的企业级应用支持
- 强大的并发处理能力

学习建议:
1. 掌握 Java 核心语法和面向对象编程
2. 熟悉 Spring Boot 框架
3. 学习数据库交互 (JPA/JDBC)
4. 了解 RESTful API 设计
5. 掌握单元测试
6. 学习常用设计模式

通过这份指南，你已经了解了 Java 后端开发的基础知识和核心技能。随着经验的积累，你可以进一步探索更高级的主题，如微服务架构、消息队列、缓存等。