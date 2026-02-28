# 场景四：Tap与仓库管理

## 场景描述

管理 Homebrew 的软件源仓库（Tap），包括官方仓库和第三方仓库，以获取更多软件包。

## 涉及命令

### 1. brew tap

**用途**: 管理第三方软件源仓库

**语法**:
```bash
brew tap                       # 列出已添加的 Tap
brew tap <user/repo>           # 添加 Tap
brew tap <user/repo> <url>     # 添加指定 URL 的 Tap
brew tap --force <user/repo>   # 强制添加（覆盖）
brew untap <user/repo>         # 移除 Tap
```

**示例**:
```bash
# 列出所有 Tap
$ brew tap
homebrew/cask
homebrew/core
homebrew/services

# 添加第三方 Tap
$ brew tap mongodb/brew
==> Tapping mongodb/brew
Cloning into '/opt/homebrew/Library/Taps/mongodb/homebrew-brew'...

# 添加自定义 URL 的 Tap
$ brew tap myorg/packages https://github.com/myorg/homebrew-packages.git

# 移除 Tap
$ brew untap mongodb/brew
Untapping mongodb/brew...
Untapped 1 formula
```

---

### 2. brew untap

**用途**: 移除已添加的 Tap

**语法**:
```bash
brew untap <user/repo>         # 移除 Tap
brew untap --force <user/repo> # 强制移除（即使有包依赖）
```

**示例**:
```bash
# 移除 Tap
$ brew untap mongodb/brew
Untapping mongodb/brew...
Untapped 1 formula

# 强制移除
$ brew untap --force myorg/packages
```

---

### 3. brew tap-info

**用途**: 查看 Tap 详细信息

**语法**:
```bash
brew tap-info <user/repo>      # 查看单个 Tap 信息
brew tap-info --installed      # 查看所有已安装 Tap 信息
brew tap-info --json <tap>     # JSON 格式输出
```

**示例**:
```bash
# 查看 Tap 信息
$ brew tap-info homebrew/cask
homebrew/cask: (9037 formulae, 9037 casks)
/usr/local/Homebrew/Library/Taps/homebrew/homebrew-cask
From: https://github.com/Homebrew/homebrew-cask

# 查看所有已安装 Tap
$ brew tap-info --installed
```

---

## 官方 Tap 列表

| Tap | 说明 |
|------|------|
| `homebrew/core` | 核心软件包仓库（默认） |
| `homebrew/cask` | macOS 图形界面应用（默认） |
| `homebrew/services` | 服务管理工具（默认） |
| `homebrew/bundle` | Brewfile 支持 |
| `homebrew/test-bot` | CI 测试工具 |
| `homebrew/command-not-found` | 命令未找到提示 |

---

## 常用第三方 Tap

### 开发工具

```bash
# Google 工具
brew tap google-cloud-sdk

# AWS 工具
brew tap aws/tap

# MongoDB
brew tap mongodb/brew

# ElasticSearch
brew tap elastic/tap

# HashiCorp（Terraform、Consul 等）
brew tap hashicorp/tap
```

### 编程语言

```bash
# PHP 扩展
brew tap shivammathur/php

# Python 版本
brew tap pyenv/pyenv

# Node.js 版本
brew tap nodelayer/nodelayer
```

### 特定软件

```bash
# GUI 应用
brew tap homebrew/cask-fonts     # 字体
brew tap homebrew/cask-versions  # 版本变体
brew tap homebrew/cask-drivers   # 驱动

# 常用工具
brew tap derailed/k9s            # Kubernetes CLI
brew tap jesseduffield/lazygit   # Lazygit
brew tap jesseduffield/lazydocker # Lazydocker
```

---

## 使用场景

### 场景 1: 安装 MongoDB

```bash
# 1. 添加 MongoDB Tap
brew tap mongodb/brew

# 2. 安装 MongoDB
brew install mongodb-community

# 3. 启动服务
brew services start mongodb-community
```

### 场景 2: 安装 Google Cloud SDK

```bash
# 1. 添加 Google Cloud Tap
brew tap google-cloud-sdk

# 2. 安装
brew install google-cloud-sdk

# 3. 初始化
gcloud init
```

### 场景 3: 安装 PHP 扩展版本

```bash
# 1. 添加 PHP Tap
brew tap shivammathur/php

# 2. 安装特定版本
brew install shivammathur/php/php@8.2

# 3. 链接
brew link --overwrite --force shivammathur/php/php@8.2
```

### 场景 4: 安装字体

```bash
# 1. 添加字体 Tap
brew tap homebrew/cask-fonts

# 2. 搜索字体
brew search font-fira-code

# 3. 安装字体
brew install --cask font-fira-code
```

### 场景 5: 创建自己的 Tap

```bash
# 1. 在 GitHub 创建仓库
# 格式: homebrew-<tap-name>
# 例如: https://github.com/username/homebrew-tools

# 2. 添加 Formula
# 创建 Formula/<package>.rb 文件

# 3. 用户添加你的 Tap
brew tap username/tools

# 4. 安装
brew install <package>
```

---

## 自定义 Tap 示例

### Formula 结构

```
homebrew-tools/
├── Formula/
│   ├── mycli.rb
│   └── another-tool.rb
├── Casks/
│   └── myapp.rb
└── README.md
```

### Formula 示例

```ruby
# Formula/mycli.rb
class Mycli < Formula
  desc "My awesome CLI tool"
  homepage "https://github.com/username/mycli"
  version "1.0.0"
  license "MIT"

  on_macos do
    on_arm do
      url "https://github.com/username/mycli/releases/download/v1.0.0/mycli_1.0.0_darwin_arm64.tar.gz"
      sha256 "abc123..."
    end
    on_intel do
      url "https://github.com/username/mycli/releases/download/v1.0.0/mycli_1.0.0_darwin_amd64.tar.gz"
      sha256 "def456..."
    end
  end

  def install
    bin.install "mycli"
  end

  test do
    assert_match "version 1.0.0", shell_output("#{bin}/mycli --version")
  end
end
```

### Cask 示例

```ruby
# Casks/myapp.rb
cask "myapp" do
  version "1.0.0"
  sha256 "abc123..."

  url "https://github.com/username/myapp/releases/download/v#{version}/MyApp.dmg"
  name "My App"
  desc "My awesome application"
  homepage "https://github.com/username/myapp"

  app "MyApp.app"
end
```

---

## Tap 目录结构

```
/opt/homebrew/Library/Taps/
├── homebrew/
│   ├── homebrew-cask/
│   ├── homebrew-core/
│   └── homebrew-services/
├── mongodb/
│   └── homebrew-brew/
├── elastic/
│   └── homebrew-tap/
└── custom/
    └── homebrew-packages/
```

---

## 最佳实践

### 1. 定期更新 Tap

```bash
# 更新所有 Tap
brew update

# 更新特定 Tap
brew tap --force homebrew/cask
```

### 2. 检查 Tap 状态

```bash
# 查看所有 Tap
brew tap

# 查看详细信息
brew tap-info --installed
```

### 3. 清理不需要的 Tap

```bash
# 移除不使用的 Tap
brew untap <user/repo>

# 检查是否有包依赖
brew uses --installed <tap>/<formula>
```

### 4. 使用官方 Tap 优先

```bash
# 优先搜索官方仓库
brew search <package>

# 如果没有，再考虑第三方 Tap
```

---

## 常见问题

### Q: Tap 添加失败？

```bash
# 检查网络连接
ping github.com

# 检查 Tap 是否存在
# 访问 https://github.com/<user>/homebrew-<repo>

# 强制添加
brew tap --force <user/repo>
```

### Q: 如何查看 Tap 中的软件包？

```bash
# 列出 Tap 中的所有包
brew list --formula | grep <tap-name>

# 查看 Tap 目录
ls /opt/homebrew/Library/Taps/<user>/<repo>/Formula/
```

### Q: Tap 和 Formula 的关系？

- **Tap**: 软件包仓库（Git 仓库）
- **Formula**: 软件包定义文件（Ruby 脚本）

一个 Tap 可以包含多个 Formula。

### Q: 如何创建私有 Tap？

```bash
# 使用私有 Git 仓库
brew tap myorg/packages https://github.com/myorg/homebrew-packages.git

# 使用 SSH
brew tap myorg/packages git@github.com:myorg/homebrew-packages.git
```

---

## 相关命令

| 命令 | 说明 |
|------|------|
| `brew tap` | 管理第三方仓库 |
| `brew untap` | 移除仓库 |
| `brew tap-info` | 查看仓库信息 |
| `brew update` | 更新所有仓库 |

---

*最后更新: 2026年2月28日*
