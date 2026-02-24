import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pandas as pd
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pathlib import Path
from MetaFeature import getfeature

df = pd.read_csv('./dataset/data.csv')
target_cols = [f'target{i}' for i in range(7)]
# 标签编码成数字并记录映射关系
all_label_mapping = []
for col in target_cols:
    df[col], unique = pd.factorize(df[col])
    all_label_mapping.append(unique)

# 预处理数据
def preprocess(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return lemmatized_tokens

# 获取模型输入， 使用Word2Vec 将Target和DatasetName转换为编码
def get_input(df):
    for col_name in ['DatasetName', 'Target']:
        # 将所有数据集中的文本预处理并转换为列表的列表
        all_tokens = [preprocess(text) for text in df[col_name]]
        all_tokens = [[text] for text in df[col_name]]

        # 训练Word2Vec模型
        model = Word2Vec(all_tokens, vector_size=100, window=5, min_count=1, workers=4)
        model.train(all_tokens, total_examples=len(all_tokens), epochs=10)

        all_vectors = []
        for row in df[col_name].values:
            # 获取单词的向量表示
            word_vector = model.wv[row]
            all_vectors.append(word_vector)
        all_vectors = pd.DataFrame(all_vectors)
        all_vectors.columns = [f'{col_name}{i}' for i in all_vectors.columns]

        df = pd.concat([df, all_vectors], axis=1)
        
    # df['DatasetName'] = df['DatasetName'].astype('category').cat.codes
    # df['Target'] = df['Target'].astype('category').cat.codes

    return df.drop(['DatasetName', 'Target'], axis=1)

df = get_input(df)

from openfe import OpenFE, tree_to_formula, transform
# 设置task和metric
task = 'classification'
metric = 'multi_logloss'

for col in target_cols:
    print(f'Processing {col}')
    x_df = df.drop(target_cols, axis=1)
    y_df = df[col]
    train_idx, val_idx = pd.Series(x_df.index), pd.Series(x_df.index)
    # 训练openfe模型
    ofe = OpenFE()

    params = {"num_iterations": 1000, "seed": 42}
    params.update(
        {
            'objective':'l2',
            "colsample_bytree": 0.8,
            "colsample_bynode": 0.8,
            "learning_rate": 0.05,
            "num_leaves":31,
        }
    )
    ofe.fit(data=x_df, task=task, train_index=train_idx, val_index=val_idx, 
            metric=metric, label=y_df, seed=42,stage2_metric='permutation',
            n_data_blocks=1,
            stage2_params=params)