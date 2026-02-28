# 场景五：Cask应用管理

## 场景描述

使用 Homebrew Cask 安装和管理 macOS 图形界面应用程序，如浏览器、IDE、设计工具等。

## 涉及命令

### 1. brew install --cask

**用途**: 安装 macOS 图形界面应用

**语法**:
```bash
brew install --cask <app>      # 安装应用
brew install --cask --no-quarantine <app> # 安装并跳过隔离检查
brew install --cask --skip-cask-deps <app> # 跳过 Cask 依赖
brew install --cask --force <app> # 强制重新安装
```

**示例**:
```bash
# 安装 VS Code
$ brew install --cask visual-studio-code
==> Downloading https://update.code.visualstudio.com/1.87.0/darwin/stable
==> Downloading from https://az764295.vo.msecnd.net/stable/xxx/VSCode-darwin.zip
==> Verifying checksum for Cask 'visual-studio-code'
==> Moving App 'Visual Studio Code.app' to '/Applications/Visual Studio Code.app'
🍺  visual-studio-code was successfully installed!

# 安装多个应用
brew install --cask google-chrome firefox

# 安装并跳过隔离检查（解决"无法打开，因为它来自身份不明的开发者"）
brew install --cask --no-quarantine some-app
```

---

### 2. brew uninstall --cask

**用途**: 卸载 Cask 应用

**语法**:
```bash
brew uninstall --cask <app>    # 卸载应用
brew uninstall --cask --force <app> # 强制卸载
brew uninstall --cask --zap <app> # 彻底删除（包括配置文件）
```

**示例**:
```bash
# 基本卸载
$ brew uninstall --cask visual-studio-code
==> Uninstalling Cask visual-studio-code
==> Backing App 'Visual Studio Code.app' up to '/opt/homebrew/Caskroom/visual-studio-code/1.87.0'
==> Removing App '/Applications/Visual Studio Code.app'
==> Purging files for version 1.87.0 of Cask visual-studio-code

# 彻底删除（包括配置和缓存）
$ brew uninstall --cask --zap visual-studio-code
==> Uninstalling Cask visual-studio-code with zap; ignoring quarantine settings
==> Removing App '/Applications/Visual Studio Code.app'
==> Removing files:
~/Library/Application Support/Code
~/Library/Caches/com.microsoft.VSCode
~/Library/Preferences/com.microsoft.VSCode.plist
...
```

---

### 3. brew list --cask

**用途**: 列出已安装的 Cask 应用

**语法**:
```bash
brew list --cask               # 列出所有 Cask 应用
brew list --cask --versions    # 显示版本号
brew list --cask <app>         # 列出应用的文件
```

**示例**:
```bash
# 列出所有 Cask 应用
$ brew list --cask
docker              google-chrome       visual-studio-code
firefox             slack               zoom

# 带版本号
$ brew list --cask --versions
docker 4.27.1
google-chrome 122.0.6261.112
visual-studio-code 1.87.0
```

---

### 4. brew info --cask

**用途**: 查看 Cask 应用信息

**语法**:
```bash
brew info --cask <app>         # 查看应用信息
brew info --cask --verbose <app> # 详细信息
brew info --cask --json=v2 <app> # JSON 格式
```

**示例**:
```bash
$ brew info --cask visual-studio-code
==> visual-studio-code: 1.87.0 (auto_updates)
https://code.visualstudio.com/
/opt/homebrew/Caskroom/visual-studio-code/1.87.0 (112B)
From: https://github.com/Homebrew/homebrew-cask/blob/HEAD/Casks/v/visual-studio-code.rb
==> Name
Microsoft Visual Studio Code
==> Description
Open-source code editor
==> Artifacts
Visual Studio Code.app (App)
==> Not Installed
==> Caveats
To use the CLI tool, add to your PATH:
  export PATH="$PATH:/Applications/Visual Studio Code.app/Contents/Resources/app/bin"
```

---

### 5. brew search --cask

**用途**: 搜索 Cask 应用

**语法**:
```bash
brew search --cask <keyword>   # 搜索 Cask 应用
brew search --cask /<regex>/   # 正则表达式搜索
```

**示例**:
```bash
# 搜索 Chrome 相关应用
$ brew search --cask chrome
==> Casks
chrome-devtools              google-chrome               google-chrome-beta
chromedriver                 google-chrome-canary        chrome-remote-desktop

# 搜索开发工具
$ brew search --cask code
==> Casks
codeedit                    visual-studio-code          xcodes
codecov                     visual-studio-code-insiders
```

---

### 6. brew upgrade --cask

**用途**: 升级 Cask 应用

**语法**:
```bash
brew upgrade --cask           # 升级所有 Cask 应用
brew upgrade --cask <app>     # 升级指定应用
brew upgrade --cask --greedy  # 包含自动更新的应用
brew upgrade --cask --dry-run # 预览升级
```

**示例**:
```bash
# 升级所有 Cask
$ brew upgrade --cask
==> Upgrading 2 outdated packages:
visual-studio-code 1.86.0 -> 1.87.0
docker 4.26.0 -> 4.27.1

# 升级指定应用
$ brew upgrade --cask visual-studio-code

# 强制升级自动更新的应用
$ brew upgrade --cask --greedy google-chrome
```

---

### 7. brew outdated --cask

**用途**: 查看过时的 Cask 应用

**语法**:
```bash
brew outdated --cask          # 查看过时的 Cask
brew outdated --cask --greedy # 包含自动更新的应用
brew outdated --cask --json   # JSON 格式
```

**示例**:
```bash
$ brew outdated --cask
docker (4.26.0) != 4.27.1
visual-studio-code (1.86.0) != 1.87.0
```

---

### 8. brew reinstall --cask

**用途**: 重新安装 Cask 应用

**语法**:
```bash
brew reinstall --cask <app>   # 重新安装
brew reinstall --cask --force <app> # 强制重新安装
```

---

### 9. brew audit --cask

**用途**: 审核 Cask 配置

**语法**:
```bash
brew audit --cask <app>       # 审核指定 Cask
brew audit --cask --strict    # 严格审核
```

---

## 常用 Cask 应用分类

### 开发工具

```bash
# 编辑器和 IDE
brew install --cask visual-studio-code     # VS Code
brew install --cask cursor                 # Cursor
brew install --cask sublime-text           # Sublime Text
brew install --cask intellij-idea          # IntelliJ IDEA
brew install --cask pycharm                # PyCharm
brew install --cask webstorm               # WebStorm
brew install --cask goland                 # GoLand

# 终端工具
brew install --cask iterm2                 # iTerm2
brew install --cask warp                   # Warp

# 开发环境
brew install --cask docker                 # Docker Desktop
brew install --cask rancher                # Rancher Desktop
```

### 浏览器

```bash
brew install --cask google-chrome          # Chrome
brew install --cask firefox                # Firefox
brew install --cask microsoft-edge         # Edge
brew install --cask brave-browser          # Brave
brew install --cask arc                    # Arc
```

### 效率工具

```bash
# 截图录屏
brew install --cask cleanshot              # CleanShot X
brew install --cask kap                    # Kap

# 剪贴板
brew install --cask maccy                  # Maccy

# 启动器
brew install --cask raycast                # Raycast
brew install --cask alfred                 # Alfred

# 窗口管理
brew install --cask rectangle              # Rectangle
brew install --cask magnet                 # Magnet

# 笔记
brew install --cask notion                 # Notion
brew install --cask obsidian               # Obsidian
brew install --cask typora                 # Typora
```

### 设计工具

```bash
brew install --cask figma                  # Figma
brew install --cask sketch                 # Sketch
brew install --cask adobe-creative-cloud   # Adobe CC
brew install --cask sip                    # Sip (取色器)
```

### 通讯工具

```bash
brew install --cask slack                  # Slack
brew install --cask discord                # Discord
brew install --cask telegram               # Telegram
brew install --cask zoom                   # Zoom
brew install --cask microsoft-teams        # Teams
```

### 多媒体

```bash
brew install --cask vlc                    # VLC
brew install --cask spotify                # Spotify
brew install --cask iina                   # IINA
```

### 字体

```bash
# 添加字体 Tap
brew tap homebrew/cask-fonts

# 安装字体
brew install --cask font-fira-code
brew install --cask font-hack-nerd-font
brew install --cask font-jetbrains-mono
brew install --cask font-source-code-pro
```

---

## 使用场景

### 场景 1: 搭建开发环境

```bash
# 安装核心开发工具
brew install --cask visual-studio-code
brew install --cask iterm2
brew install --cask docker

# 安装浏览器
brew install --cask google-chrome

# 安装 Git 客户端
brew install --cask github
brew install --cask fork
```

### 场景 2: 批量安装应用

```bash
# 创建应用列表
apps=(
    "visual-studio-code"
    "iterm2"
    "docker"
    "google-chrome"
    "firefox"
    "slack"
    "notion"
    "rectangle"
)

# 批量安装
for app in "${apps[@]}"; do
    brew install --cask "$app"
done

# 或使用 Brewfile
cat > Brewfile << 'EOF'
cask "visual-studio-code"
cask "iterm2"
cask "docker"
cask "google-chrome"
cask "firefox"
cask "slack"
cask "notion"
cask "rectangle"
EOF

brew bundle install
```

### 场景 3: 更新所有应用

```bash
# 查看过时的应用
brew outdated --cask

# 更新所有应用
brew upgrade --cask

# 强制更新自动更新的应用
brew upgrade --cask --greedy
```

### 场景 4: 彻底卸载应用

```bash
# 普通卸载（保留配置）
brew uninstall --cask visual-studio-code

# 彻底卸载（删除配置和缓存）
brew uninstall --cask --zap visual-studio-code
```

### 场景 5: 解决安装问题

```bash
# 如果遇到"无法打开，因为它来自身份不明的开发者"
brew install --cask --no-quarantine <app>

# 或手动移除隔离属性
xattr -cr /Applications/<App>.app
```

---

## Cask vs Formula

| 特性 | Cask | Formula |
|------|------|---------|
| 安装目标 | macOS 图形界面应用 | 命令行工具 |
| 安装位置 | `/Applications/` | `/opt/homebrew/Cellar/` |
| 更新方式 | 手动/自动更新 | `brew upgrade` |
| 配置文件 | `.rb` (Casks/) | `.rb` (Formula/) |
| 示例 | Chrome、VS Code | git、node、python |

---

## Cask 配置文件结构

```ruby
# Casks/visual-studio-code.rb
cask "visual-studio-code" do
  version "1.87.0"
  sha256 "abc123..."

  url "https://update.code.visualstudio.com/#{version}/darwin/stable"
  name "Microsoft Visual Studio Code"
  desc "Open-source code editor"
  homepage "https://code.visualstudio.com/"

  livecheck do
    url "https://code.visualstudio.com/Updates"
    strategy :sparkle
  end

  auto_updates true
  depends_on macos: ">= :high_sierra"

  app "Visual Studio Code.app"
  binary "#{appdir}/Visual Studio Code.app/Contents/Resources/app/bin/code"

  zap trash: [
    "~/Library/Application Support/Code",
    "~/Library/Preferences/com.microsoft.VSCode.plist",
    "~/Library/Caches/com.microsoft.VSCode",
  ]
end
```

---

## 最佳实践

### 1. 使用 Brewfile 管理应用

```ruby
# Brewfile
tap "homebrew/cask-fonts"

# 开发工具
cask "visual-studio-code"
cask "iterm2"
cask "docker"

# 浏览器
cask "google-chrome"
cask "firefox"

# 效率工具
cask "rectangle"
cask "maccy"

# 字体
cask "font-fira-code"
cask "font-jetbrains-mono"
```

### 2. 处理自动更新应用

```bash
# 有些应用自带更新功能，brew upgrade 不会更新
# 使用 --greedy 强制更新
brew upgrade --cask --greedy google-chrome

# 或在 Cask 中 auto_updates: true
```

### 3. 解决权限问题

```bash
# 移除隔离属性
xattr -cr /Applications/<App>.app

# 或安装时跳过隔离
brew install --cask --no-quarantine <app>
```

### 4. 定期更新

```bash
# 每周执行
brew update
brew outdated --cask
brew upgrade --cask
```

---

## 常见问题

### Q: 安装后应用无法打开？

```bash
# 方法 1: 安装时跳过隔离
brew install --cask --no-quarantine <app>

# 方法 2: 手动移除隔离属性
xattr -cr /Applications/<App>.app

# 方法 3: 在系统偏好设置中允许
# 系统偏好设置 → 安全性与隐私 → 通用 → 允许从以下位置下载的 App
```

### Q: 如何查找应用的 Cask 名称？

```bash
# 搜索
brew search --cask <keyword>

# 或在官网查找
# https://formulae.brew.sh/cask/
```

### Q: Cask 应用不更新？

```bash
# 检查是否自动更新
brew info --cask <app> | grep auto_updates

# 强制更新
brew upgrade --cask --greedy <app>
```

### Q: 如何备份已安装的应用列表？

```bash
# 导出为 Brewfile
brew bundle dump --file=~/Brewfile

# 恢复
brew bundle install --file=~/Brewfile
```

---

## 相关命令

| 命令 | 说明 |
|------|------|
| `brew search --cask` | 搜索 Cask 应用 |
| `brew info --cask` | 查看应用信息 |
| `brew install --cask` | 安装应用 |
| `brew uninstall --cask` | 卸载应用 |
| `brew upgrade --cask` | 升级应用 |
| `brew list --cask` | 列出已安装应用 |

---

*最后更新: 2026年2月28日*
