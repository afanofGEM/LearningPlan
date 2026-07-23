# 将 LearningPlan 项目上传到 GitHub

本文说明如何把整个 `learningPlan` 项目交给 Git 管理，并上传到 GitHub。

## 一、项目结构

当前项目大致如下：

```text
learningPlan/
├── common/
├── tips/
├── week1：训练loop示例/
├── week2/
└── week3/
```

因为要上传整个项目，所以应在 `learningPlan` 根目录执行 Git 命令，而不是进入某个 `week` 目录。

## 二、首次上传

### 1. 进入项目根目录

在 VS Code 中打开终端，然后执行：

```powershell
cd D:\cuhk\learningPlan
```

`cd` 是 `change directory` 的缩写，用来切换当前目录。后续 Git 命令将作用于这个目录及其子目录。

可以使用下面的命令确认当前位置：

```powershell
pwd
```

### 2. 创建 `.gitignore`

在 `learningPlan` 根目录创建一个名为 `.gitignore` 的文本文件，建议写入：

```gitignore
# Python 缓存
__pycache__/
*.py[cod]

# Python 虚拟环境
.venv/
venv/

# VS Code 本地配置
.vscode/

# Jupyter 临时文件
.ipynb_checkpoints/

# 环境变量和密钥
.env
*.env
```

`.gitignore` 用来告诉 Git 哪些文件不需要管理和上传。即使项目中没有隐私内容，也不建议上传 Python 缓存、虚拟环境和编辑器的本地配置。

### 3. 初始化 Git 仓库

```powershell
git init
```

该命令会在当前目录创建或完善隐藏的 `.git` 文件夹，让 Git 开始管理这个项目。

`.git` 中保存提交历史、分支信息和远程仓库配置，不应手动修改。执行 `git init` 只会初始化本地仓库，不会把文件上传到 GitHub。

### 4. 将文件加入暂存区

```powershell
git add .
```

其中：

- `git add` 表示选择要包含在下一次提交中的文件。
- `.` 表示当前目录及其所有子目录。

可以把暂存区理解成“下一次存档的文件清单”：

```text
工作区修改 → git add → 暂存区 → git commit → 本地提交历史
```

如果只想暂存一个文件，可以指定其路径：

```powershell
git add week3/src/train.py
```

### 5. 检查 Git 状态

```powershell
git status
```

该命令会显示：

- 当前所在分支；
- 已经暂存、等待提交的文件；
- 已修改但尚未暂存的文件；
- 尚未被 Git 跟踪的新文件。

提交前建议执行一次 `git status`，确认没有误加入不需要上传的文件。

### 6. 创建第一次本地提交

```powershell
git commit -m "Initial commit: learning plan"
```

其中：

- `git commit` 表示创建一条正式的本地版本记录。
- `-m` 表示直接在命令中填写提交说明。
- `"Initial commit: learning plan"` 是本次提交的说明。

一次提交类似一次项目存档，会记录作者、时间、说明和当时的文件状态。这一步仍然只保存在本地，尚未上传到 GitHub。

如果 Git 提示尚未配置用户名和邮箱，可以执行：

```powershell
git config --global user.name "你的GitHub用户名"
git config --global user.email "你的GitHub邮箱"
```

配置完成后，再次执行提交命令。

### 7. 将主分支命名为 `main`

```powershell
git branch -M main
```

其中：

- `git branch` 用来管理分支。
- `-M` 表示重命名当前分支。
- `main` 是新的分支名称。

GitHub 通常使用 `main` 作为默认分支名称。可以使用下面的命令查看本地分支：

```powershell
git branch
```

当前分支前会显示星号：

```text
* main
```

### 8. 在 GitHub 创建空仓库

登录 GitHub，点击右上角的 `+`，选择 **New repository**，然后：

1. 将仓库命名为 `learningPlan`。
2. 根据需要选择 Public 或 Private。
3. 不要勾选自动创建 README、`.gitignore` 或 License。
4. 点击 **Create repository**。

创建后会获得类似下面的仓库地址：

```text
https://github.com/你的用户名/learningPlan.git
```

### 9. 连接本地仓库与 GitHub

将命令中的用户名替换成自己的 GitHub 用户名：

```powershell
git remote add origin https://github.com/你的用户名/learningPlan.git
```

其中：

- `git remote` 用来管理远程仓库。
- `add` 表示添加一个远程仓库。
- `origin` 是这个远程仓库的简称。
- 最后的 URL 是 GitHub 仓库地址。

可以检查远程仓库配置：

```powershell
git remote -v
```

正常情况下会看到类似结果：

```text
origin  https://github.com/你的用户名/learningPlan.git (fetch)
origin  https://github.com/你的用户名/learningPlan.git (push)
```

### 10. 首次上传

```powershell
git push -u origin main
```

其中：

- `git push` 表示将本地提交上传到远程仓库。
- `origin` 表示上传到刚才配置的 GitHub 仓库。
- `main` 表示上传本地的 `main` 分支。
- `-u` 表示建立本地 `main` 与远程 `main` 的默认对应关系。

首次上传时，VS Code 或浏览器可能会要求登录 GitHub并进行授权。建立对应关系后，以后通常只需要执行 `git push`。

## 三、以后如何更新项目

修改代码后，可以依次执行：

```powershell
git add .
git status
git commit -m "Update week3"
git push
```

对应的含义是：

```text
git add      选择这次需要保存的修改
git status   检查将要提交的内容
git commit   在本地创建版本记录
git push     将本地提交上传到 GitHub
```

提交说明应尽量描述本次修改，例如：

```powershell
git commit -m "Add week3 model evaluation"
git commit -m "Fix data preprocessing bug"
git commit -m "Update project README"
```

## 四、常用查看命令

### 查看当前状态

```powershell
git status
```

### 查看提交历史

```powershell
git log --oneline
```

### 查看远程仓库

```powershell
git remote -v
```

### 查看当前分支

```powershell
git branch
```

### 查看尚未暂存的具体修改

```powershell
git diff
```

### 查看已暂存、即将提交的具体修改

```powershell
git diff --staged
```

## 五、常见问题

### `git` 不是可识别的命令

这通常表示尚未安装 Git，或者安装后尚未重新启动 VS Code。安装 Git 后，关闭并重新打开 VS Code。

### Git 要求配置用户名和邮箱

执行：

```powershell
git config --global user.name "你的GitHub用户名"
git config --global user.email "你的GitHub邮箱"
```

这里的用户名和邮箱会记录在提交信息中。

### `remote origin already exists`

这表示已经配置过名为 `origin` 的远程仓库。先查看当前配置：

```powershell
git remote -v
```

如果地址不正确，可以修改：

```powershell
git remote set-url origin https://github.com/你的用户名/learningPlan.git
```

### 修改后执行 `git push`，GitHub 没有变化

`git push` 只上传已经提交的内容。应先执行：

```powershell
git add .
git commit -m "描述本次修改"
git push
```

### `.gitignore` 中的文件为什么仍然被 Git 跟踪

`.gitignore` 通常只对尚未被 Git 跟踪的文件生效。如果某个文件已经提交过，即使后来将它写入 `.gitignore`，Git 仍会继续跟踪它，需要另行取消跟踪。

## 六、核心概念总结

```text
Git：在本地记录和管理项目版本
GitHub：在网络上保存和分享 Git 仓库

git init：初始化本地仓库
git add：选择下一次提交的内容
git commit：创建本地版本记录
git remote：配置远程仓库
git push：将本地提交上传到 GitHub
```
