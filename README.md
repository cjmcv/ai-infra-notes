# Reading notes on the open source code of AI infrastructure

AI Infra 相关开源代码的阅读笔记，笔记均以代码注释的方式呈现。

## 仓库管理方式：

1. 将想深入阅读代码的仓库fork下来，保留主分支用于同步代码.

2. 基于当前主分支最新代码，新增note-date1分支, date1对应开分支前的最后一次提交日期。

3. 网页端在settings中，将note-date1分支设为Default branch。

4. 开始阅读note-date1分支代码，在代码阅读过程中将自己的理解以注释方式添加到代码里。

5. 经一段时间阅读后，主分支已落后于原仓库，则通过网页端Sync fork直接将主分支更新到最新，基于更新后的主分支再新增一个note-date2。

6. 因note-date2是基于主分支从原仓库上同步过来的，上一阶段的阅读笔记还留在note-date1中，则需要使用对比软件，
   手动将note-date1中的中文注解挪到note-date2中，同时将新的阅读分支设为Default branch，进行下一阶段的阅读，以5和6步反复推进。

7. 将代码阅读的fork仓库以submodule的方式接入到本仓库中统一管理。
   如: git submodule add https://github.com/cjmcv/sglang.git sglang

## 浏览笔记方式

```bash
git clone --recursive https://github.com/cjmcv/ai-infra-reading-notes.git
# OR
git clone https://github.com/cjmcv/ai-infra-reading-notes.git
cd ai-infra-reading-notes
git submodule init
cd the-module-you-like
git pull
```