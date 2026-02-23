# 🛒 VOC-Analyzer: 智能电商评价洞察与 AI 商业分析引擎

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Moonshot AI](https://img.shields.io/badge/LLM-Kimi_API-purple.svg)](https://platform.moonshot.cn/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **🚀 核心价值**：将传统的“人工逐条阅读差评 -> 主观打标签 -> 耗时数天撰写报告”的低效流程，转化为“**一键上传 -> 算法降维聚类 -> LLM 秒级输出高管级商业洞察**”的自动化闭环。用技术跨越数据鸿沟，赋能产品与运营团队高效决策。

---

**🌐 在线体验地址**：[点击这里访问我的云端部署产品](https://bi7e7gi7jkznernmxwlyp5.streamlit.app/) 

---

## 🌟 核心产品特性

* **🔪 纯净降噪的 NLP 流水线**：内置专属电商停用词库，结合 `Jieba` 分词与 `TF-IDF` 算法，精准过滤情绪废话，锁定核心实体特征词。
* **🧠 无监督深度痛点挖掘**：抛弃繁琐的人工打标，引入 `Scikit-Learn NMF` (非负矩阵分解) 机器学习算法。在海量非结构化短文本中，自动收敛并精准聚类出 N 个核心产品痛点（如：漏光、模具公差、按键异响）。
* **🤖 AI 商业洞察自动生成**：无缝接入 **Kimi (月之暗面) 大模型 API**。将算法提取的干瘪特征词组装为专业 Prompt，一键输出包含“痛点定性”、“深层归因”与“迭代建议”的 Markdown 商业报告。
* **📊 全链路可视化与存档**：提供直观的词频柱状图与高亮情绪词云图。内置本地 JSON 状态持久化机制，历史分析项目秒级调阅，沉淀企业数据资产。

---

## 🏗️ 系统架构设计

本项目采用轻量级解耦架构，兼顾了算法的严谨性与前端展现的敏捷性：

1.  **Data Input (数据层)**：支持标准化 CSV 评价数据源接入。
2.  **NLP Pipeline (预处理层)**：正则清洗 -> Jieba 深度分词 -> TF-IDF 权重计算。
3.  **Machine Learning (算法层)**：NMF 矩阵分解 / Gensim LDA 主题模型。
4.  **LLM Agent (决策层)**：Moonshot API 零样本推理与长文本生成。
5.  **Application (展示层)**：基于 Streamlit 框架的响应式 Web 交互界面。

---

## 🚀 极速本地部署

### 1. 克隆项目
git clone [https://github.com/sheesy/VOC-.git](https://github.com/sheesy/VOC-.git)

cd VOC-Analyzer

### 2.安装依赖环境
pip install -r requirements.txt

### 3.配置 API 秘钥与启动
本项目已实现 API Key 前端动态注入，代码本体零硬编码，保护开发者隐私。

streamlit run VOC-DEMO.py

运行后：浏览器会自动打开 http://localhost:8501。
在左侧边栏填入你的 Kimi API Key 即可开始分析。


## 📂 目录结构说明

VOC-Analyzer/
├── VOC-DEMO.py               # Streamlit 主程序应用引擎
├── requirements.txt          # 核心依赖清单
├── simhei.ttf                # 解决云端 Linux 环境下中文词云乱码的内置字体包
├── monitor_reviews.csv       # (可选) 提供给测试用户的 2000 条显示器 Mock 评价数据
└── voc_history_projects/     # 历史项目本地存档缓存库 (运行时自动生成)
