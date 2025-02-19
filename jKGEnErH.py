import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Создаем примерный датасет для 25 дилеров
data = {
    'Dealer': [f'Dealer {i}' for i in range(1, 26)],
    'Rating': [4.3, 3.9, 4.7, 3.5, 4.1, 4.6, 4.0, 3.8, 4.4, 3.7,
               4.2, 4.5, 3.6, 4.8, 3.9, 4.1, 4.3, 3.8, 4.6, 4.0,
               4.9, 3.7, 4.2, 3.9, 4.4],
    'MedianPrice': [41000, 37000, 45000, 32000, 39000, 43000, 36000, 35000, 42000, 33000,
                    40000, 44000, 34000, 48000, 37000, 39000, 41000, 36000, 45000, 38000,
                    50000, 34000, 40000, 37000, 43000],
    'AnnualTurnover': [6.2, 4.8, 8.5, 3.2, 5.0, 7.4, 4.5, 3.8, 6.0, 3.5,
                       5.6, 7.0, 3.3, 9.0, 5.2, 5.4, 6.3, 3.7, 8.0, 5.0,
                       10.0, 3.6, 5.8, 5.1, 6.7],
    'Branches': [3, 2, 4, 1, 2, 3, 2, 1, 3, 1,
                 2, 3, 1, 4, 2, 2, 3, 1, 4, 2,
                 5, 1, 2, 2, 3],
    'SocialNetworks': ['Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No',
                       'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes',
                       'Yes', 'No', 'Yes', 'Yes', 'Yes']
}

df = pd.DataFrame(data)

# Убедимся, что отсутствуют пропущенные значения
df.fillna(0, inplace=True)

# Преобразуем наличие соцсетей в бинарный формат: 'Yes' -> 1, 'No' -> 0
df['SocialNetworksBinary'] = df['SocialNetworks'].apply(lambda x: 1 if x == 'Yes' else 0)

# Нормализуем числовые показатели (Rating, MedianPrice, AnnualTurnover, Branches)
df['norm_Rating'] = (df['Rating'] - df['Rating'].min()) / (df['Rating'].max() - df['Rating'].min())
df['norm_MedianPrice'] = (df['MedianPrice'] - df['MedianPrice'].min()) / (df['MedianPrice'].max() - df['MedianPrice'].min())
df['norm_Turnover'] = (df['AnnualTurnover'] - df['AnnualTurnover'].min()) / (df['AnnualTurnover'].max() - df['AnnualTurnover'].min())
df['norm_Branches'] = (df['Branches'] - df['Branches'].min()) / (df['Branches'].max() - df['Branches'].min())

# Определяем веса:
# SocialNetworks (бинарно) - 0.35, MedianPrice - 0.20, AnnualTurnover - 0.20,
# Rating - 0.15, Branches - 0.10
w_social = 0.35
w_median_price = 0.20
w_turnover = 0.20
w_rating = 0.15
w_branches = 0.10

# Вычисляем итоговый WeightedScore по нормализованным значениям
df['WeightedScore'] = (
    w_social * df['SocialNetworksBinary'] +
    w_median_price * df['norm_MedianPrice'] +
    w_turnover * df['norm_Turnover'] +
    w_rating * df['norm_Rating'] +
    w_branches * df['norm_Branches']
)

# Сортируем дилеров по итоговой оценке (от наивысшей к наименьшей)
df_sorted = df.sort_values('WeightedScore', ascending=False)

############################################
# Heatmap 1: Нормализованные показатели и итоговая оценка
############################################
norm_heatmap_data = df_sorted[['norm_MedianPrice', 'norm_Rating', 'norm_Turnover', 'norm_Branches', 'SocialNetworksBinary', 'WeightedScore']]
norm_heatmap_data.index = df_sorted['Dealer']

plt.figure(figsize=(10, 10))
sns.heatmap(norm_heatmap_data, annot=True, cmap='RdYlGn', fmt=".2f", vmin=0, vmax=1)
plt.title("Heatmap: Normalized Metrics and Final Score")
plt.ylabel("Dealer")
plt.show()

############################################
# Heatmap 2: Исходные данные с цветовой схемой нормализованных значений
############################################
# Для цветовой заливки используем нормализованные данные (чтобы цветовая схема оставалась такой же),
# а аннотации показывают исходные данные из датасета.

color_data = df_sorted[['norm_Rating', 'norm_MedianPrice', 'norm_Turnover', 'norm_Branches', 'SocialNetworksBinary', 'WeightedScore']]
color_data.index = df_sorted['Dealer']

raw_annotation = df_sorted[['Rating', 'MedianPrice', 'AnnualTurnover', 'Branches', 'SocialNetworks', 'WeightedScore']]
raw_annotation.index = df_sorted['Dealer']

# Округляем значения в столбце WeightedScore до сотых долей в аннотациях
raw_annotation['WeightedScore'] = raw_annotation['WeightedScore'].astype(float).round(2)

# Приводим все аннотации к строкам
raw_annotation = raw_annotation.astype(str)

plt.figure(figsize=(10, 10))
sns.heatmap(color_data, annot=raw_annotation, cmap='RdYlGn', fmt="", cbar=True, vmin=0, vmax=1)
plt.title("Heatmap: Raw Data with Normalized Color Mapping")
plt.ylabel("Dealer")
plt.show()