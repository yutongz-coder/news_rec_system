1. 天池零基础入门推荐系统-新闻推荐： https://tianchi.aliyun.com/competition/entrance/531842<br><br>
2. datawhale的推荐系统教程是很好的入门学习资料，其中不仅有教程，还有推荐算法实习面经：https://datawhalechina.github.io/fun-rec/#/ <br><br>
3. 这是一份Markdown语法官方指南：https://markdown.com.cn/basic-syntax/headings.html<br><br>
4. 2025年腾讯广告算法大赛：https://algo.qq.com/<br><br>
5. 生成式推荐论文阅读笔记：<br><br>
[A Review of Modern Recommender Systems Using Generative Models (Gen-RecSys)](https://www.notion.so/A-Review-of-Modern-Recommender-Systems-Using-Generative-Models-Gen-RecSys-21c4ea50d98a8001801fd1fd4f9a099a?source=copy_link)<br><br>
[Recommender Systems with Generative Retrieval](https://www.notion.so/Recommender-Systems-with-Generative-Retrieval-23c4ea50d98a804c85dfd6a61630817e?source=copy_link)<br><br>
[Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://www.notion.so/Actions-Speak-Louder-than-Words-Trillion-Parameter-Sequential-Transducers-for-Generative-Recommenda-23c4ea50d98a804585bec85476aeee69?source=copy_link)<br><br>

作为只有Python基础的小白，入门推荐系统第一个项目，本文档在datawhale的推荐系统实战项目基础上说明了不同文件实现的功能，以及应该如何运行；并对该项目的流程进行详细的梳理；同时练习了使用Markdown的能力。运行环境：rec，新创建的环境，需要安装tensorflow才能安装deepctr，其他的包提示未安装的时候再一一安装。<br><br>
阅读项目代码的时候，小白看到一堆新定义的函数头都大了，可以拿草稿纸梳理一下有哪些函数，数据流是什么样的，动手写一遍可以对代码有更熟练的掌握，只看真的很难熟练掌握。<br><br>
打开切换面板（右上角设置键左边中间那个面板），点击下面的终端界面右上角的下拉箭头，不要选择PowerShell，选择Git Bash输入下面的内容就可以把修改的内容用git同步到仓库。
 `git add .` 
 `git commit -m "修改说明".` 
 `git push` 

## 赛题数据
赛题以预测用户未来点击新闻文章为任务，该数据来自某新闻APP平台的用户交互数据，包括30万用户，近300万次点击，共36万多篇不同的新闻文章，同时每篇新闻文章有对应的embedding向量表示。为了保证比赛的公平性，将会从中抽取20万用户的点击日志数据作为训练集，5万用户的点击日志数据作为测试集A，5万用户的点击日志数据作为测试集B。

## 1 基准模型实现了最基础的基于物品的协同过滤（ItemCF的召回）
**主要流程**：
    读取用户点击日志（训练集/测试集）
    构建用户-文章-点击时间的字典
    计算文章与文章之间的相似度（ItemCF，经典共现+归一化）
    对每个用户，基于历史点击文章召回与之最相似的文章
    结果整理、生成提交文件  
**特点**：
    只用了一种召回方式：ItemCF
    召回分数只考虑了用户共现和归一化，没有引入其他特征或复杂权重
    代码结构简单，适合入门和基础理解

## 2 数据分析
**主要数据变量**
1. *trn_click - 训练集用户点击日志*
* 数据来源: train_click_log.csv
* 包含内容: 用户点击行为数据，包含用户ID、文章ID、时间戳、点击环境等信息
* 数据量: 1,112,623 条记录，20万用户
2. *tst_click - 测试集用户点击日志*
* 数据来源: testA_click_log.csv
* 包含内容: 测试集用户点击行为数据
* 数据量: 5万用户（用户ID: 200000-249999）
3. *item_df - 新闻文章信息*
* 数据来源: articles.csv
* 包含内容: 文章属性信息，包含文章ID、类别ID、创建时间、字数等
* 数据量: 364,047 篇文章
4. *item_emb_df - 新闻文章embedding向量*
* 数据来源: articles_emb.csv
* 包含内容: 文章的向量表示，用于计算文章相似度
5. *user_click_merge - 合并后的用户点击数据*
* 数据来源: trn_click + tst_click 的合并
* 包含内容: 训练集和测试集的完整用户点击数据
* 用途: 用于统一特征处理和统计分析
**衍生数据变量**
6. *user_click_count - 用户重复点击统计*
* 包含内容: 每个用户对每篇文章的点击次数统计
7. *item_click_count - 文章点击次数统计*
* 包含内容: 每篇文章被点击的次数统计（3.5万篇文章被点击过）
8. *union_item - 新闻共现频次*
* 包含内容: 两篇文章连续出现的次数统计
9. *mean_diff_click_time - 用户点击时间差*
* 包含内容: 每个用户前后点击文章的时间差平均值
10. *item_emb_np - 文章embedding数组*
* 包含内容: 文章向量表示的numpy数组，用于相似度计算

## 3 多路召回.ipynb实现了多路召回（多种召回策略）并引入更多特征和复杂性
**主要流程和功能**：
    读取用户点击日志、文章属性、文章embedding等多种数据
    构建用户-文章、文章-用户等多种字典
**多路召回**：
    ItemCF召回（但引入了更多权重：点击顺序、点击时间、文章创建时间等）
    Embedding召回（基于文章内容embedding，用faiss加速向量检索）
    UserCF召回（基于用户相似性）
    预留了 YouTubeDNN、冷启动等召回方式
    召回评估（线下验证）和线上全量召回都支持
结果整理、评估、保存
**特点**：
    多种召回方式，可融合多路召回结果
    ItemCF更复杂：不仅仅是共现，还考虑了点击顺序、时间、文章属性等多种权重
    支持Embedding召回：利用内容特征，faiss加速
    支持UserCF召回：基于用户相似性
    支持冷启动、YouTubeDNN等扩展
    代码结构更复杂，功能更全，适合比赛后期或实际系统# news-recommendation-system

## 4 特征工程.ipynb
**整体代码主线思路**
1. 数据读取与划分
* 读取原始点击日志、文章信息等。
* 划分训练集、验证集、测试集（如每个用户最后一次点击为验证，其余为训练）。
2. 召回阶段
* 针对每个数据集（训练/验证/测试），用召回算法（如itemCF、内容相似等）为每个用户召回一批候选新闻，生成 recall_list_dict。
* 将召回字典转为 DataFrame（recall_list_df）。
3. 打标签
* 将召回的 user-item 对与真实点击数据对齐，打上正负样本标签（label=1/0），得到 recall_items_df。
4. 负采样
* 对训练集的负样本进行下采样，平衡正负样本比例，减少数据量。
5. 特征工程
* 基于用户历史行为、候选新闻、用户画像、新闻画像等，生成丰富的特征（如相似性、时间差、字数差等）。
形成最终用于排序模型训练/验证/测试的特征 DataFrame。
6. 模型训练与预测
* 用 LightGBM、XGBoost 等排序模型进行训练、验证和预测。
**主要数据结构梳理**
* click_trn / click_val / click_tst：训练/验证/测试集的点击日志（DataFrame）
* recall_list_dict：召回结果，字典结构，key为user_id，value为候选新闻及得分
* recall_list_df：召回结果转成的DataFrame，列有user_id、sim_item、score
* recall_items_df：召回结果与真实点击对齐并打标签后的DataFrame，多了label列
* trn_user_item_label_tuples_dict：以user_id为key，value为三元组(item, score, label)的列表，便于特征工程
* all_user_feas：二维列表，每行是一个样本的所有特征，最终转成特征DataFrame
**主要函数梳理**
* trn_val_split：划分训练集和验证集
* get_recall_list：生成召回列表（dict）
* recall_dict_2_df：召回字典转DataFrame
* get_rank_label_df：召回数据打标签
* neg_sample_recall_data：负采样
* make_tuple_func：分组后转三元组列表
* create_feature：生成排序特征的核心函数