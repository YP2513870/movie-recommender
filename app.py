import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
 
st.set_page_config(page_title="智能电影推荐系统", page_icon="🎬")

# ---------------------- 1. 数据加载与预处理 ----------------------
@st.cache_data  # Streamlit缓存，避免重复加载数据
def load_and_preprocess_data():
    # 读取数据集（需将数据集文件放在与app.py同级的"data"文件夹中）
    movies_df = pd.read_csv("data/movies.csv")  # 电影信息：movieId, title, genres
    ratings_df = pd.read_csv("data/ratings.csv")  # 用户评分：userId, movieId, rating, timestamp

    # 数据清洗：删除缺失值、去重
    movies_df = movies_df.dropna().drop_duplicates(subset=["movieId"])
    ratings_df = ratings_df.dropna().drop_duplicates(subset=["userId", "movieId"])

    # 电影类型独热编码（核心特征）
    genres_set = set()
    # 拆分类型并收集所有唯一类型
    movies_df["genres_list"] = movies_df["genres"].str.split("|")
    for genres in movies_df["genres_list"]:
        genres_set.update(genres)
    genres_list = list(genres_set)

    # 构建类型特征矩阵
    genre_features = pd.DataFrame(0, index=movies_df.index, columns=genres_list)
    for idx, genres in enumerate(movies_df["genres_list"]):
        for genre in genres:
            genre_features.loc[idx, genre] = 1

    # 合并电影信息与类型特征
    movies_with_features = pd.concat([movies_df, genre_features], axis=1)

    # 构建用户-电影评分矩阵（用于后续筛选用户喜欢的电影）
    user_movie_matrix = ratings_df.pivot_table(
        index="userId", columns="movieId", values="rating", fill_value=0
    )

    return movies_df, ratings_df, movies_with_features, user_movie_matrix, genres_list

# 加载数据
movies_df, ratings_df, movies_with_features, user_movie_matrix, genres_list = load_and_preprocess_data()

# ---------------------- 2. 推荐算法核心 ----------------------
def get_similar_movies(target_movie_title, top_n=5):
    """
    基于内容相似度推荐电影
    :param target_movie_title: 用户输入的目标电影名称（支持模糊匹配）
    :param top_n: 推荐电影数量
    :return: 推荐电影列表（含标题、类型、相似度得分）
    """
    # 1. 模糊匹配目标电影（解决用户输入不精确问题）
    matched_movies = movies_df[movies_df["title"].str.contains(target_movie_title, case=False, na=False)]
    if matched_movies.empty:
        return None  # 无匹配电影时返回None

    # 取匹配到的第一部电影（若有多个匹配，默认选第一个）
    target_movie = matched_movies.iloc[0]
    target_movie_id = target_movie["movieId"]

    # 2. 提取目标电影的特征向量
    target_features = movies_with_features[movies_with_features["movieId"] == target_movie_id][genres_list].values

    # 3. 计算所有电影与目标电影的余弦相似度
    all_features = movies_with_features[genres_list].values
    similarity_scores = cosine_similarity(target_features, all_features)[0]

    # 4. 构建相似度DataFrame并排序（排除目标电影本身）
    similarity_df = pd.DataFrame({
        "movieId": movies_with_features["movieId"],
        "title": movies_with_features["title"],
        "genres": movies_with_features["genres"],
        "similarity_score": similarity_scores
    })
    # 排除目标电影，按相似度降序排序
    similarity_df = similarity_df[similarity_df["movieId"] != target_movie_id].sort_values(
        by="similarity_score", ascending=False
    )

    # 5. 筛选Top-N推荐结果
    recommended_movies = similarity_df.head(top_n).reset_index(drop=True)
    # 保留2位小数，提升可读性
    recommended_movies["similarity_score"] = recommended_movies["similarity_score"].round(2)

    return recommended_movies, target_movie["title"]  # 返回推荐结果和匹配到的电影名称

# ---------------------- 3. Streamlit交互界面 ----------------------
def main():
    # 设置页面标题和图标
    st.title("🎬 智能电影推荐系统")
    st.markdown("基于电影内容相似度，为您推荐喜欢的电影！")
    st.divider()  # 分割线

    # 用户输入组件
    with st.form(key="recommendation_form"):
        # 电影名称输入框（提示示例）
        target_movie = st.text_input("请输入您喜欢的电影名称（示例：Toy Story）", placeholder="输入电影名称...")
        # 推荐数量选择器（1-10部）
        top_n = st.slider("请选择推荐电影数量", min_value=1, max_value=10, value=5)
        # 提交按钮
        submit_btn = st.form_submit_button(label="获取推荐")

    # 处理推荐请求
    if submit_btn:
        if not target_movie.strip():  # 空输入校验
            st.warning("请输入您喜欢的电影名称！")
        else:
            with st.spinner("正在为您寻找相似电影..."):
                recommended_movies, matched_title = get_similar_movies(target_movie, top_n)
                if recommended_movies is None:
                    st.error(f"未找到包含「{target_movie}」的电影，请尝试其他名称！")
                else:
                    # 显示匹配结果
                    st.success(f"已为您找到与《{matched_title}》相似的电影：")
                    # 用表格展示推荐结果（美化格式）
                    st.table(
                        recommended_movies[["title", "genres", "similarity_score"]].rename(
                            columns={
                                "title": "电影标题",
                                "genres": "电影类型",
                                "similarity_score": "相似度得分"
                            }
                        )
                    )
                    # 补充说明
                    st.caption("注：相似度得分越高，电影内容与您喜欢的电影越相似（基于类型匹配）。")

# 启动应用
if __name__ == "__main__":
    main()