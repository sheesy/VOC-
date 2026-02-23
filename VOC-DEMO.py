import streamlit as st
import pandas as pd
import jieba
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from openai import OpenAI
import os
import json
from datetime import datetime
import platform
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ==========================================
# 0. 初始化与环境辅助函数
# ==========================================
HISTORY_DIR = "voc_history_projects"
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

# 直接使用项目文件夹里自带的黑体文件！
def get_chinese_font():
    return "simhei.ttf"

# ==========================================
# 1. 页面配置与侧边栏
# ==========================================
st.set_page_config(page_title="VOC 全链路分析引擎", page_icon="⚙️", layout="wide")
st.title("⚙️ VOC 智能挖掘系统 (NMF 提效版)")

history_files = [f for f in os.listdir(HISTORY_DIR) if f.endswith('.json')]
history_files.sort(reverse=True)

with st.sidebar:
    st.header("🔑 引擎配置")
    kimi_api_key = st.text_input("请输入 Kimi API Key (sk-...)", type="password")
    num_topics = st.slider("期望 NMF 聚类的主题数量", min_value=2, max_value=6, value=3)

    st.markdown("---")
    options = ["🟢 当前工作区 (上传新数据)"] + history_files
    selected_mode = st.selectbox("🗂️ 历史分析项目记录", options)

    st.markdown("---")
    st.markdown("🛠️ **当前版本**: V4.0 | 底层升级为 NMF + TF-IDF")

# ==========================================
# 2. 核心逻辑路由
# ==========================================

# ---------------- 场景 A：查看历史项目 ----------------
if selected_mode != "🟢 当前工作区 (上传新数据)":
    st.subheader(f"📂 历史报告归档：{selected_mode}")
    file_path = os.path.join(HISTORY_DIR, selected_mode)
    with open(file_path, "r", encoding="utf-8") as f:
        saved_data = json.load(f)

    st.info(f"🕒 **分析时间**：{saved_data.get('analyze_time', '未知')} | 📁 **原始数据**：{saved_data.get('original_file', '未知')}")
    st.success("提取的核心痛点特征词：")
    for i, kw in enumerate(saved_data.get('topics', [])):
        st.write(f"- 主题 {i+1}: {kw}")

    st.markdown("---")
    st.header("🤖 Kimi 商业归因分析报告")
    st.markdown(saved_data.get('report', '暂无报告内容'))

# ---------------- 场景 B：当前工作区 ----------------
else:
    st.markdown("请上传原始评价数据，见证 **文本切分 -> TF-IDF 权重计算 -> NMF 矩阵分解 -> AI 报告** 的全流程！")
    uploaded_file = st.file_uploader("📂 选择 CSV 评价数据文件 (如 monitor_reviews.csv)", type=["csv"])

    if uploaded_file is not None:
        if st.button("🚀 启动全链路智能分析"):

            # --- 数据加载 ---
            df = pd.read_csv(uploaded_file)
            if 'content' not in df.columns:
                st.error("❌ 数据中必须包含名为 'content' 的评论文本列！")
                st.stop()

            if 'score' in df.columns:
                df = df[df['score'] <= 3].copy()

            df = df.dropna(subset=['content'])
            raw_docs = df['content'].tolist()

            if len(raw_docs) == 0:
                st.error("❌ 过滤后没有可分析的负面数据！")
                st.stop()
            st.success(f"✅ 成功锁定 {len(raw_docs)} 条待分析的吐槽数据。")
            st.markdown("---")

            # --- 第一阶段：Jieba ---
            st.header("🔪 第一阶段：NLP 文本切分与词云")
            # 针对显示器的专属停用词（过滤掉没营养的口水话）
            stop_words = set(['的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', 
                              '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', 
                              '买', '真的', '什么', '可以', '怎么', '感觉', '这么', '显示器', '屏幕', '这台', '实话', '一周', '总体', '来说'])

            with st.spinner("正在启动 Jieba 引擎并绘制图表..."):
                cut_docs = []
                all_words = []
                for text in raw_docs:
                    words = [w for w in jieba.cut(str(text)) if w not in stop_words and len(w) > 1]
                    cut_docs.append(words)
                    all_words.extend(words)

                st.subheader("全局情绪核心词云图")
                font_path = get_chinese_font()
                try:
                    wordcloud = WordCloud(
                        font_path=font_path, width=800, height=300, 
                        background_color='white', colormap='magma', max_words=100
                    ).generate(" ".join(all_words))

                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
                except Exception as e:
                    st.warning("词云生成失败，请检查字体。")
            st.markdown("---")

            # --- 第二阶段：NMF 矩阵分解 ---
            st.header("🧠 第二阶段：Sklearn NMF 痛点聚类 (TF-IDF 加持)")
            with st.spinner(f"正在计算 TF-IDF 权重矩阵并进行 NMF 降维分解..."):
                # NMF 需要将切好的词重新拼成字符串
                joined_docs = [" ".join(words) for words in cut_docs]

                # 1. 计算 TF-IDF
                tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)
                tfidf = tfidf_vectorizer.fit_transform(joined_docs)
                tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

                # 2. 训练 NMF 模型
                nmf_model = NMF(n_components=num_topics, random_state=42, init='nndsvd')
                nmf_model.fit(tfidf)

                # 3. 提取核心词汇
                nmf_topics_for_llm = []
                cols = st.columns(num_topics)

                for topic_idx, topic in enumerate(nmf_model.components_):
                    # 获取权重最高的前 6 个词
                    top_features_ind = topic.argsort()[:-7:-1]
                    top_features = [tfidf_feature_names[i] for i in top_features_ind]
                    keywords_str = "、".join(top_features)
                    nmf_topics_for_llm.append(keywords_str)

                    with cols[topic_idx]:
                        st.info(f"**💡 痛点集群 {topic_idx + 1}**\n\n{keywords_str}")

            st.markdown("---")

            # --- 第三阶段：Kimi LLM ---
            st.header("🤖 第三阶段：Kimi 大模型商业洞察报告")
            if not kimi_api_key:
                st.warning("⚠️ 左侧边栏未检测到 Kimi API Key，流水线在此中止。")
                st.stop()

            client = OpenAI(api_key=kimi_api_key, base_url="https://api.moonshot.cn/v1")

            with st.spinner("正在将 NMF 提取的锋利特征发送至 Kimi 进行组装..."):
                topics_prompt = "\n".join([f"痛点 {i+1}：【{kw}】" for i, kw in enumerate(nmf_topics_for_llm)])
                prompt = f"""
                你是一位资深的硬件产品经理。我们通过 NMF 矩阵分解算法从显示器差评中提取了以下痛点特征词：
                {topics_prompt}

                请撰写一份商业洞察报告：
                1. 为每个痛点起一个专业标题（如：面板漏光与坏点品控问题）。
                2. 还原用户痛点场景，并深挖原因（面板选型、模具公差、驱动板调校等）。
                3. 给出 1 条供应链或研发端的硬核改进建议。
                排版要求：严谨清晰，使用 Markdown 格式。
                """

                try:
                    response = client.chat.completions.create(
                        model="moonshot-v1-8k",
                        messages=[
                            {"role": "system", "content": "你是一个硬核的数据驱动型硬件产品策略专家。"},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.6
                    )
                    report_content = response.choices[0].message.content
                    st.success("🎉 AI 洞察分析完毕！")
                    st.markdown(report_content)

                    # 存档
                    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    save_filename = f"{current_time_str}_{uploaded_file.name}.json"
                    save_path = os.path.join(HISTORY_DIR, save_filename)
                    project_data = {
                        "analyze_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "original_file": uploaded_file.name,
                        "topics": nmf_topics_for_llm,
                        "report": report_content
                    }
                    with open(save_path, "w", encoding="utf-8") as f:
                        json.dump(project_data, f, ensure_ascii=False, indent=4)

                    st.toast('💾 分析结果已自动存档！', icon='✅')
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Kimi API 调用失败：{e}")

