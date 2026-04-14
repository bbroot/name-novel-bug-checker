# 小说Bug检查技能

## 概述

小说Bug检查技能是一个专业的叙事质量审查工具，专门用于检测小说中的逻辑漏洞、角色不一致、节奏问题和叙事Bug。

## 核心功能

| 功能 | 说明 |
|------|------|
| 逻辑漏洞检测 | 时间线矛盾、因果断裂、能力突变、信息知晓不合理 |
| 角色一致性检查 | 性格突变、动机矛盾、对话风格偏移、成长弧线断裂 |
| 节奏结构分析 | 信息密度、高潮铺垫、场景过渡、伏笔回收 |

## 安装

```bash
pip install jieba
```

> 仅依赖 `jieba`（中文分词），轻量无需重型NLP库。

## 快速开始

在 OpenClaw 中激活技能后，直接粘贴小说文本或提供文件路径，AI将自动执行全面检查并输出分级报告。

## 文件结构

```
novel-bug-checker/
├── SKILL.md                    # 核心定义文件
├── README.md                  # 本文件
├── references/                # 参考资料库
│   ├── bug-patterns.md        # 常见Bug模式分类
│   ├── repair-strategies.md   # 修复策略库
│   ├── narrative-theory.md    # 叙事学理论基础
│   └── character-consistency.md # 角色一致性检查指南
├── scripts/                   # 分析工具脚本
│   ├── logic-analyzer.py     # 逻辑漏洞分析
│   ├── rhythm-analyzer.py    # 节奏分析
│   └── consistency-checker.py # 角色一致性检查
└── templates/                 # 输出模板
    ├── bug-report.txt        # Bug报告模板
    ├── repair-suggestions.txt # 修复建议模板
    └── summary-report.txt    # 总结报告模板
```

## Bug严重程度

- 🔴 **致命**：破坏故事可信度的核心矛盾
- 🟠 **严重**：影响重要情节合理性的问题
- 🟡 **中等**：影响阅读体验但可容忍
- 🟢 **轻微**：细节瑕疵，不影响整体

## 命令行使用（可选）

```bash
# 逻辑分析
python scripts/logic-analyzer.py novel.txt [-o report.txt]

# 节奏分析
python scripts/rhythm-analyzer.py novel.txt -g 玄幻

# 角色一致性检查
python scripts/consistency-checker.py novel.txt [-o report.txt]
```
