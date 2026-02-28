# Homebrew 使用场景指南

本目录包含了 Homebrew 包管理器的常见使用场景和对应的命令参考。

## 场景列表

| 场景 | 描述 | 文档 |
|------|------|------|
| [安装与配置](./01-安装与配置.md) | Homebrew 安装与环境配置 | 详情 |
| [软件包管理](./02-软件包管理.md) | 安装、更新、卸载软件包 | 详情 |
| [软件服务管理](./03-软件服务管理.md) | 管理后台服务（数据库、消息队列等） | 详情 |
| [Tap与仓库管理](./04-Tap与仓库管理.md) | 添加和管理第三方软件源 | 详情 |
| [Cask应用管理](./05-Cask应用管理.md) | 安装和管理 macOS 图形界面应用 | 详情 |
| [版本管理](./06-版本管理.md) | 管理软件包的多个版本 | 详情 |
| [缓存与清理](./07-缓存与清理.md) | 管理缓存和清理旧版本 | 详情 |
| [诊断与故障排查](./08-诊断与故障排查.md) | 检查和解决 Homebrew 问题 | 详情 |

## 快速参考

### 常用命令速查

```bash
# 安装与更新
brew install <package>          # 安装软件包
brew uninstall <package>        # 卸载软件包
brew update                     # 更新 Homebrew 本身
brew upgrade                    # 升级所有软件包
brew upgrade <package>          # 升级指定软件包

# 查询与搜索
brew search <keyword>           # 搜索软件包
brew info <package>             # 查看软件包信息
brew list                       # 列出已安装的软件包
brew outdated                   # 查看过时的软件包

# 服务管理
brew services start <service>   # 启动服务
brew services stop <service>    # 停止服务
brew services restart <service> # 重启服务
brew services list              # 列出所有服务状态

# Cask 应用
brew install --cask <app>       # 安装图形界面应用
brew uninstall --cask <app>     # 卸载图形界面应用

# 清理与维护
brew cleanup                    # 清理旧版本
brew doctor                     # 诊断问题
brew autoremove                 # 自动移除未使用的依赖
```

## 术语说明

| 术语 | 说明 |
|------|------|
| Formula | 软件包的定义文件（Ruby 脚本），描述如何编译和安装软件 |
| Cask | macOS 图形界面应用的安装定义 |
| Keg | 已安装软件包的目录 |
| Cellar | 所有 Keg 的存储目录（默认 `/opt/homebrew/Cellar` 或 `/usr/local/Cellar`） |
| Tap | 第三方软件源仓库 |
| Bottle | 预编译的二进制包，加速安装过程 |
| Prefix | Homebrew 的安装根目录 |
| opt | 软件包的符号链接目录 |
| Services | 后台服务管理（如 MySQL、Redis 等） |

## 目录结构

```
/opt/homebrew/           # Apple Silicon Mac
├── Cellar/              # 所有已安装的软件包
│   ├── git/             # git 软件包
│   ├── node/            # node 软件包
│   └── ...
├── Caskroom/            # 已安装的 Cask 应用
├── Homebrew/            # Homebrew 核心代码
├── opt/                 # 软件包的符号链接
├── bin/                 # 可执行文件链接
├── sbin/                # 系统可执行文件链接
├── etc/                 # 配置文件
├── var/                 # 变量数据
├── cache/               # 缓存目录
└── logs/                # 日志目录
```

## 相关资源

- [Homebrew 官方网站](https://brew.sh/)
- [Homebrew 官方文档](https://docs.brew.sh/)
- [Homebrew Formula 浏览器](https://formulae.brew.sh/)
- [Homebrew GitHub 仓库](https://github.com/Homebrew/brew)

---

*最后更新: 2026年2月28日*
*Homebrew 版本: 4.x*
