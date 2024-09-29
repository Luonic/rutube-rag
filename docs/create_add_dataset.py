import json
import pandas as pd
from glob import glob

if __name__ == '__main__':
    paths = glob('data/generated_qa_for_docs/*.json')
    data = []
    for path in paths:
        with open(path, 'r') as f:
            sample_data = json.load(f)
            data.extend(sample_data['questions']['questions'])
    test_df = pd.read_csv('../reranker/data/all_data.csv', sep=';')
    test_df['clf'] = test_df['classifier_1'] + ' | ' +test_df['classifier_2']
    print(test_df.columns)
    df = pd.DataFrame(data)
    df.loc[df['clf'] == 'ОТСУТСТВУЕТ', 'clf'] = 'ОТСУТСТВУЕТ | Отсутствует'
    df.loc[df['clf'] == 'ОТСУТСТУЕТ', 'clf'] = 'ОТСУТСТВУЕТ | Отсутствует'
    df.loc[df['clf'] == 'ПРЕДЛОЖЕНИЯ | Персонализация', 'clf'] = 'ПРЕДЛОЖЕНИЯ | Персонализация 0'
    df.loc[df['clf'] == 'МОДЕРАЦИЯ | Верификация', 'clf'] = 'ОТСУТСТВУЕТ | Отсутствует'
    df['clf'] = df['clf'].apply(lambda x: x.split(' | '))
    df['classifier_1'] = df['clf'].apply(lambda x: x[0])
    df['classifier_2'] = df['clf'].apply(lambda x: x[1])
    df['is_knowledge_base'] = 1

    df = pd.concat([test_df, df], axis=0, ignore_index=True)
    df.drop('clf', axis=1, inplace=True)
    df.to_csv('data/all_data_with_docs.csv', sep=';', index=False)
    print(df.isnull().sum())