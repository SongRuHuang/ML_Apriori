import pandas as pd
from ucimlrepo import fetch_ucirepo
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 載入數據集
stock_portfolio_performance = fetch_ucirepo(id=390)
X = stock_portfolio_performance.data.features.copy()

# 強制將所有欄位轉換為數字，無法轉換的會變成 NaN ---
X = X.apply(pd.to_numeric, errors='coerce')

# 處理缺失值（qcut 不能處理 NaN） ---
# 選擇填補 0 或者刪除包含 NaN 的列
X = X.fillna(0)


# 2. 數據預處理與離散化
X_discrete = pd.DataFrame(index=X.index)

for col in X.columns:
    # --- 關鍵修正：只處理數字型態的欄位 ---
    if not pd.api.types.is_numeric_dtype(X[col]):
        print(f"跳過非數值欄位: {col}")
        continue

    full_labels = [f'{col}_Low', f'{col}_Med', f'{col}_High']

    try:
        # 測試切分
        temp_cuts, bins = pd.qcut(X[col], 3, duplicates='drop', retbins=True)
        actual_num_bins = len(bins) - 1

        if actual_num_bins == 0:
            print(f"欄位 {col} 數值太單一，無法切分。")
            continue

        current_labels = full_labels[:actual_num_bins]
        X_discrete[col] = pd.qcut(
            X[col], 3, labels=current_labels, duplicates='drop')

    except Exception as e:
        print(f"處理欄位 {col} 時發生錯誤: {e}")

# 3. 進行 One-hot Encoding
X_encoded = pd.get_dummies(X_discrete)

# 4. 應用頻繁項目集挖掘 (Apriori)
frequent_itemsets = apriori(X_encoded, min_support=0.15, use_colnames=True)

# 5. 生成關聯規則
if not frequent_itemsets.empty:
    rules = association_rules(
        frequent_itemsets, metric="lift", min_threshold=1.2)

    # 篩選出信心度較高的規則
    rules = rules[rules['confidence'] > 0.6].sort_values(
        by='lift', ascending=False)

    # 顯示結果
    print("--- 識別出的股票特徵關聯規則 ---")
    display_rules = rules[['antecedents',
                           'consequents', 'support', 'confidence', 'lift']]
    print(display_rules.head(10))

    # 6. 視覺化
if not rules.empty:
    # --- 關鍵：先設定字型，再開始畫圖 ---
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # Windows 微軟正黑體
    plt.rcParams['axes.unicode_minus'] = False              # 修正負號顯示

    plt.figure(figsize=(10, 6))

    # 繪製散點圖
    scatter = plt.scatter(rules['support'], rules['confidence'],
                          c=rules['lift'], cmap='YlOrRd',
                          s=rules['lift']*100, alpha=0.6)

    # 設定顏色條
    plt.colorbar(scatter, label='Lift (提升度)')

    # 設定標籤與標題
    plt.title("股票投資組合關聯規則分析", fontsize=15)
    plt.xlabel("Support (支持度)")
    plt.ylabel("Confidence (信心度)")

    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()
else:
    print("未找到符合條件的頻繁項目集，請嘗試降低 min_support。")
