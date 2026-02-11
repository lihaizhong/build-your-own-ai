# npm 使用场景指南

本目录包含了 npm 包管理器的常见使用场景和对应的命令参考。

## 场景列表

| 场景 | 描述 | 文档 |
|------|------|------|
| [项目初始化](./01-项目初始化.md) | 创建新的 Node.js 项目 | 详情 |
| [依赖管理](./02-依赖管理.md) | 安装、更新、卸载项目依赖 | 详情 |
| [脚本执行](./03-脚本执行.md) | 运行 package.json 中定义的脚本 | 详情 |
| [包发布](./04-包发布.md) | 将自己的包发布到 npm 仓库 | 详情 |
| [包搜索与查看](./05-包搜索与查看.md) | 查找和了解 npm 包的信息 | 详情 |
| [版本管理](./06-版本管理.md) | 管理包的版本和标签 | 详情 |
| [缓存管理](./07-缓存管理.md) | 管理 npm 缓存以提高性能 | 详情 |
| [配置管理](./08-配置管理.md) | 配置 npm 的行为和设置 | 详情 |
| [调试与诊断](./09-调试与诊断.md) | 检查和解决项目问题 | 详情 |
| [团队协作](./10-团队协作.md) | 多人开发时的协作命令 | 详情 |

## 快速参考

### 常用命令速查

```bash
# 项目初始化
npm init -y                    # 快速初始化项目

# 依赖管理
npm install                    # 安装所有依赖
npm install <package>          # 安装生产依赖
npm install <package> -D       # 安装开发依赖
npm uninstall <package>        # 卸载包
npm update <package>           # 更新包

# 脚本执行
npm run <script>               # 运行自定义脚本
npm start                      # 启动项目
npm test                       # 运行测试

# 包搜索与查看
npm search <keyword>           # 搜索包
npm view <package>             # 查看包信息

# 调试与诊断
npm audit                      # 检查安全漏洞
npm doctor                     # 诊断 npm 环境

# 配置管理
npm config list                # 查看配置
npm config set <key> <value>   # 设置配置

# 缓存管理
npm cache verify               # 验证缓存
npm cache clean --force        # 清理缓存
```

## 术语说明

| 术语 | 说明 |
|------|------|
| dependencies | 生产环境依赖，项目运行必需的包 |
| devDependencies | 开发环境依赖，开发时使用的工具 |
| peerDependencies | 同伴依赖，宿主项目需要提供的包 |
| optionalDependencies | 可选依赖，安装失败不会中断安装过程 |
| bundleDependencies | 打包依赖，打包时包含的包 |
| semver | 语义化版本控制规范 (Major.Minor.Patch) |

## 相关资源

- [npm 官方文档](https://docs.npmjs.com/)
- [package.json 规范](https://docs.npmjs.com/cli/v9/configuring-npm/package-json)
- [语义化版本规范](https://semver.org/lang/zh-CN/)
- [npm Scripts 详解](https://docs.npmjs.com/cli/v9/using-npm/scripts)

---

*最后更新: 2026年2月11日*
*npm 版本: 11.9.0*