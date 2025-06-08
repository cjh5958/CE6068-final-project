# 🧬 白血病基因表達資料增強最佳實踐指南

## 專案目標
本專案針對極高維（22,283維）小樣本（64筆）白血病基因表達資料，建立穩定可複現的資料增強與下游分析流程。核心策略為：
- 先用PCA降維（建議32維，勿超過樣本數）
- 再以條件變分自編碼器（cVAE）進行資料生成
- 所有下游流程（分類、分析）皆直接用降維後資料，不還原高維
- 全流程資料欄位、模型結構需完全一致

## 重現最佳實踐的完整步驟

### 1. 準備原始資料
- 將原始 64×22,283 維資料放於 `datasets/raw_data.csv`

### 2. PCA降維與前處理
```bash
python train_model.py --pca-dim 32
```
- 產生 `datasets/raw_data_reduced.csv`，欄位為：samples, type, PCA_0, ..., PCA_31
- **注意：PCA維度不可超過樣本數**

### 3. cVAE訓練
- 同上指令，模型自動以降維後資料訓練
- 推薦模型結構：
  - input_dim=32
  - encoder_hidden_dims=[64,32]
  - decoder_hidden_dims=[32,64]
  - latent_dim=16
- 請於config或指令明確指定，勿自動推斷

### 4. 生成資料
```bash
python generate_data.py --samples 100
```
- 產生 `datasets/gen_data.csv`，欄位與 `raw_data_reduced.csv` 完全一致
- **生成資料不需還原高維，直接用降維格式**

### 5. 下游分類/分析
- 合併 `raw_data_reduced.csv` 與 `gen_data.csv`，於 `base_model.py` 進行分類訓練與評估
- 可用 `analyze_dataset.py` 進行品質分析

## 指令範例

#### 訓練（含PCA降維）
```bash
python train_model.py --pca-dim 32 --epochs 100 --batch-size 8 --latent-dim 16 --encoder-hidden-dims 64 32 --decoder-hidden-dims 32 64
```

#### 生成資料
```bash
python generate_data.py --samples 100
```

#### 下游分析
```bash
python analyze_dataset.py datasets/gen_data.csv
```

## 資料格式說明
- **降維後資料/生成資料**：
  - 欄位：samples, type, PCA_0, ..., PCA_31
  - 兩者欄位、順序、維度完全一致
- **直接合併用於下游分類與分析**
- **不需還原到原始22,283維**

## 常見問題與排除

- **PCA維度超過樣本數**：請將 `--pca-dim` 設為小於等於樣本數
- **LabelEncoder錯誤**：請先fit全體y再分割資料
- **資料shape不符/模型結構不一致**：訓練與生成流程input_dim、hidden_dims、latent_dim需完全一致
- **生成資料欄位不符**：請確認生成資料與降維資料欄位、順序一致
- **模型檔未正確覆蓋**：請檢查 `models/` 目錄下檔案

## 技術細節
- **PCA降維**：fit於訓練集，transform於測試/生成
- **cVAE模型**：條件維度=5（五種白血病類型），推薦結構如上
- **資料分割**：訓練/驗證/測試 = 73%/12%/15%
- **下游分類**：直接用降維後資料進行

## 重要提醒
- **全流程資料格式、模型結構需一致**，避免任何自動推斷
- **生成資料直接用於下游，不需還原高維**
- **如遇bug，請先檢查資料shape、欄位、模型結構一致性**

## 📋 專案概述

本專案提供完整的條件變分自編碼器 (Conditional VAE, cVAE) 解決方案，用於白血病基因表達資料的訓練和生成。支援從零開始訓練模型，以及使用預訓練模型生成高品質合成資料。

### 🎯 主要功能
- **模型訓練**: 完整的 cVAE 訓練流程，支援自訂參數
- **資料生成**: 使用預訓練模型生成合成資料
- **多類型支援**: 支援 5 種白血病類型的資料生成
- **格式兼容**: 生成資料完全匹配原始資料格式，便於下游使用
- **靈活輸出**: 支援 CSV 和 PKL 格式
- **🆕 視覺化分析**: 自動生成資料分布檢定圖表
- **🆕 品質監控**: PCA、t-SNE、相關性分析等多維度評估
- **🆕 深度分析**: 專業的資料集分析工具，支援 PCA、t-SNE、UMAP 等多種降維方法

### 🧪 支援的白血病類型
1. **AML** - 急性骨髓性白血病
2. **Bone_Marrow** - 骨髓樣本
3. **Bone_Marrow_CD34** - CD34+ 骨髓細胞
4. **PB** - 周邊血液
5. **PBSC_CD34** - CD34+ 周邊血液幹細胞

## 📁 專案結構

```
CE6068-final-project/
├── README.md                    # 本檔案
├── train_model.py               # 模型訓練腳本
├── generate_data.py             # 資料生成腳本 (含視覺化分析)
├── analyze_dataset.py           # 🆕 專業資料集分析工具
├── model_usage_guide.py         # Python API 介面
├── models/                      # 訓練好的模型
│   ├── leukemia_cvae_model.pth  # 主要 cVAE 模型檔案
│   ├── best_cvae_model.pth      # 最佳驗證模型
│   └── final_cvae_model.pth     # 最終訓練模型
├── datasets/                    # 資料檔案
│   ├── raw_data.csv            # 原始資料集
│   ├── scaler.pkl              # 特徵正規化器
│   ├── label_encoder.pkl       # 標籤編碼器
│   ├── metadata.pkl            # 資料元資訊 (含基因名稱)
│   ├── generated_data_*.csv    # 🆕 生成的資料檔案
│   └── plots/                  # 🆕 分布檢定圖表目錄
│       ├── *_class_distribution.png     # 類別分布圖
│       ├── *_statistics_summary.png     # 統計摘要圖
│       ├── *_pca_analysis.png           # PCA 分析圖
│       ├── *_tsne_analysis.png          # t-SNE 分析圖
│       ├── *_umap_analysis.png          # 🆕 UMAP 分析圖
│       ├── *_correlation_analysis.png   # 相關性分析圖
│       └── [資料集名稱]_analysis/        # 🆕 完整分析結果目錄
├── training_logs/              # 訓練日誌 (訓練後產生)
└── [核心程式碼檔案]
```

## 🚀 快速開始

### **環境設置**
```bash
# 建立並啟動虛擬環境 (建議)
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 安裝依賴
pip install -r requirements.txt
```

### **方式 1: 模型訓練 (從零開始)**

```bash
# 基本訓練 (使用預設參數)
python train_model.py

# 快速訓練 (較少epochs，適合測試)
python train_model.py --epochs 10 --batch-size 4

# 自訂訓練參數
python train_model.py --epochs 150 --batch-size 16 --learning-rate 5e-5

# 使用GPU訓練 (如果可用)
python train_model.py --device cuda

# 使用配置檔案
python train_model.py --config my_config.json
```

### **方式 2: 資料生成 (使用預訓練模型) 🆕**

```bash
# 生成平衡資料集 (推薦) - 含完整視覺化分析
python generate_data.py --balanced --samples-per-class 50

# 生成特定類型樣本 - 含分布檢定圖表
python generate_data.py --type AML --samples 100

# 自訂輸出檔名和格式
python generate_data.py --type PB --samples 200 --format csv --output my_leukemia_data

# 跳過圖表生成 (純資料生成)
python generate_data.py --balanced --samples-per-class 30 --no-plots
```

### **方式 3: 🆕 資料集深度分析**

```bash
# 完整分析 - PCA、t-SNE、UMAP、相關性、統計特性
python analyze_dataset.py datasets/generated_data_20250602_201358.csv

# 快速分析 (限制相關性分析特徵數)
python analyze_dataset.py datasets/my_data.csv --max-corr-features 50

# 跳過計算密集的分析
python analyze_dataset.py datasets/large_data.csv --skip-tsne --skip-correlation

# 分析原始資料進行比較
python analyze_dataset.py datasets/raw_data.csv
```

### **方式 4: Python API**

```python
from model_usage_guide import LeukemiaCVAEGenerator

# 初始化生成器
generator = LeukemiaCVAEGenerator()

# 生成特定類型樣本
aml_data = generator.quick_generate_and_save(
    leukemia_type='AML', 
    n_samples=100,
    save_format='both'
)

# 生成平衡資料集
balanced_data = generator.quick_generate_balanced_and_save(
    samples_per_class=50,
    save_format='both'
)
```

## 📊 🆕 專業資料分析功能

### **analyze_dataset.py 功能概覽**

新增的專業分析工具提供多維度的資料品質檢測：

- **🔍 PCA 主成分分析**: 降維、變異解釋、類別分離度評估
- **🌀 t-SNE 非線性降維**: 探索非線性聚集結構  
- **🗺️ UMAP 統一流形逼近**: 平衡局部與全局結構的現代降維方法
- **🔗 相關性分析**: 特徵間相關性熱力圖和統計摘要
- **📈 統計特性分析**: 類別分布、特徵分布、品質評估
- **📊 降維方法比較**: 並排比較不同降維技術的效果

### **分析輸出結構**
```
datasets/plots/
└── [資料集名稱]_analysis/
    ├── pca_analysis.png              # PCA 主成分分析圖
    ├── tsne_analysis.png             # t-SNE 降維分析圖
    ├── umap_analysis.png             # UMAP 降維分析圖
    ├── correlation_analysis.png      # 特徵相關性分析圖
    ├── statistical_analysis.png      # 統計特性分析圖
    └── dimensionality_reduction_comparison.png  # 降維方法比較圖
```

### **常用分析指令**
```bash
# 1. 基本完整分析
python analyze_dataset.py datasets/generated_data.csv

# 2. 效能優化分析 (推薦用於大型資料集)
python analyze_dataset.py datasets/large_data.csv --max-corr-features 50

# 3. 比較原始 vs 生成資料
python analyze_dataset.py datasets/raw_data.csv --output-dir comparison
python analyze_dataset.py datasets/generated_data.csv --output-dir comparison

# 4. 輕量級分析 (跳過計算密集項目)
python analyze_dataset.py datasets/data.csv --skip-tsne --skip-correlation
```

## 📊 詳細資料分析工具使用指南

### **🎨 分析圖表詳解**

#### **1. PCA 主成分分析 (`pca_analysis.png`)**
- **2D 散點圖**：各類別在前兩個主成分空間的分布
- **變異解釋比例**：各主成分的貢獻度柱狀圖
- **累積變異曲線**：主成分累積解釋變異趨勢
- **品質評估**：Silhouette Score 和關鍵統計指標

#### **2. t-SNE 非線性降維 (`tsne_analysis.png`)**
- **2D 投影**：探索非線性聚集模式
- **品質評估**：聚類分離度評分
- **自動採樣**：大資料集自動採樣以提升計算效率

#### **3. UMAP 統一流形逼近 (`umap_analysis.png`)**
- **2D 投影**：保持局部和全局結構的降維結果
- **品質評估**：聚類品質評分
- **參數展示**：neighbors 參數等設定資訊

#### **4. 相關性分析 (`correlation_analysis.png`)**
- **相關性熱力圖**：高方差特徵間的相關性矩陣
- **相關係數分布**：整體相關性統計直方圖
- **高相關特徵對**：識別冗餘特徵的散點圖
- **統計摘要**：詳細的相關性統計報告

#### **5. 統計特性分析 (`statistical_analysis.png`)**
- **類別分布**：各白血病類型的樣本分布柱狀圖
- **特徵平均值分布**：基因表達平均值的分布直方圖
- **特徵變異性**：基因表達變異度的分布直方圖
- **樣本間距離**：歐幾里得距離分布分析
- **類別比較**：各類別特徵值的箱線圖比較
- **品質指標**：資料完整性和平衡性評估

#### **6. 降維方法比較 (`dimensionality_reduction_comparison.png`)**
- **並排比較**：PCA、t-SNE、UMAP 結果的視覺對比
- **聚類效果**：不同方法的類別分離效果

### **🎨 使用案例**

#### **案例 1: 快速品質檢測**
```bash
# 對新生成的資料集進行快速品質檢測
python analyze_dataset.py datasets/new_generated_data.csv --max-corr-features 30
```

#### **案例 2: 詳細研究分析**
```bash
# 完整分析，包含所有降維方法和詳細相關性分析
python analyze_dataset.py datasets/research_data.csv --max-corr-features 200
```

#### **案例 3: 輕量級分析**
```bash
# 跳過計算密集的分析，適合大型資料集
python analyze_dataset.py datasets/large_dataset.csv --skip-tsne --skip-correlation
```

#### **案例 4: 比較不同資料集**
```bash
# 分析多個資料集並比較結果
python analyze_dataset.py datasets/original_data.csv --output-dir comparison_study
python analyze_dataset.py datasets/generated_data.csv --output-dir comparison_study
python analyze_dataset.py datasets/augmented_data.csv --output-dir comparison_study
```

### **📈 結果解讀指南**

#### **PCA 分析解讀**
- **前2個主成分解釋變異 > 50%**：資料具有良好的低維表示
- **前2個主成分解釋變異 < 30%**：考慮增加主成分數量或使用非線性方法
- **高 Silhouette Score (>0.5)**：類別在降維空間中分離良好

#### **t-SNE vs UMAP 比較**
- **t-SNE 擅長**：發現局部聚集結構，突出類別內的細微差異
- **UMAP 擅長**：保持全局結構，計算效率更高，參數調整更靈活
- **建議**：兩種方法結合使用，提供互補的視角

#### **相關性分析指導**
- **高相關特徵對數量多**：考慮特徵選擇或降維以減少冗餘
- **平均相關係數接近0**：特徵獨立性良好
- **相關係數分布偏向極值**：可能存在強相關或強反相關的基因群組

#### **統計特性評估**
- **類別平衡度 > 0.8**：資料集平衡性良好
- **樣本間距離分布正常**：無明顯離群值或數據品質問題
- **特徵變異性適中**：基因表達差異合理，有利於分類

### **💡 最佳實踐建議**

#### **分析工作流程**
1. **初步檢測**：使用預設參數快速分析
2. **品質評估**：檢查 PCA 解釋變異和 Silhouette Score
3. **深入分析**：根據初步結果調整參數進行詳細分析
4. **比較研究**：對比不同資料集的分析結果

#### **效能優化**
- **大型資料集**：使用 `--max-corr-features 50` 限制相關性分析範圍
- **快速檢測**：跳過非必要分析 `--skip-tsne --skip-correlation`
- **批次處理**：編寫腳本對多個資料集批次分析

#### **結果應用**
- **模型選擇**：根據降維結果選擇合適的機器學習模型
- **特徵工程**：利用相關性分析結果進行特徵選擇
- **資料增強評估**：比較原始資料和生成資料的分布特性

## ⚙️ 系統需求

### **軟體需求**
- Python 3.8+
- PyTorch 2.0+
- 其他依賴: pandas, numpy, scikit-learn, matplotlib, seaborn, umap-learn

### **硬體需求**
- **CPU**: 現代多核心處理器
- **RAM**: 建議 8GB 以上
- **儲存空間**: 5GB 可用空間
- **GPU**: 可選，CUDA 支援的顯卡可大幅加速訓練

## 🔧 故障排除

### **訓練問題**

#### 🔴 記憶體不足
```bash
# 減少批次大小
python train_model.py --batch-size 4

# 使用較小的模型
python train_model.py --latent-dim 128
```

#### 🔴 訓練過慢
```bash
# 使用GPU (如果可用)
python train_model.py --device cuda

# 減少訓練週期
python train_model.py --epochs 50
```

### **生成問題**

#### 🔴 模型檔案遺失
```bash
# 先訓練模型
python train_model.py

# 然後生成資料
python generate_data.py --balanced
```

#### 🔴 生成失敗
確保以下檔案存在：
- `models/leukemia_cvae_model.pth`
- `datasets/scaler.pkl`
- `datasets/metadata.pkl` (含基因名稱)

#### 🔴 🆕 圖表生成失敗
```bash
# 跳過圖表生成，僅生成資料
python generate_data.py --balanced --samples-per-class 50 --no-plots

# 檢查是否安裝了視覺化依賴
pip install matplotlib seaborn umap-learn
```

### **🆕 分析問題**

#### 🔴 分析工具記憶體不足
```bash
# 減少相關性分析的特徵數量
python analyze_dataset.py dataset.csv --max-corr-features 50

# 跳過計算密集的分析
python analyze_dataset.py dataset.csv --skip-correlation --skip-tsne
```

#### 🔴 UMAP 未安裝
```bash
# 安裝 UMAP (已包含在 requirements.txt 中)
pip install umap-learn

# 或跳過 UMAP 分析
python analyze_dataset.py dataset.csv --skip-umap
```

#### 🔴 計算時間過長
```bash
# t-SNE 自動對大資料集採樣，或可手動跳過
python analyze_dataset.py dataset.csv --skip-tsne

# 減少相關性分析負擔
python analyze_dataset.py dataset.csv --max-corr-features 30
```

#### 🔴 資料格式問題
- **CSV 格式**：必須包含 `type` 欄位表示類別標籤
- **PKL 格式**：必須包含 `features`、`labels` 鍵值
- **編碼問題**：確保檔案使用 UTF-8 編碼

### **環境問題**

#### 🔴 依賴套件問題
```bash
# 安裝完整依賴
pip install -r requirements.txt

# 或使用conda
conda install pytorch pandas numpy scikit-learn matplotlib seaborn
conda install -c conda-forge umap-learn
```

## 支援與進階使用

### **命令列參數完整列表**

#### **訓練腳本 (train_model.py)**
```bash
--config          # 配置檔案路徑
--epochs          # 訓練週期數 (預設: 100)
--batch-size      # 批次大小 (預設: 8)
--learning-rate   # 學習率 (預設: 1e-4)
--latent-dim      # 潛在空間維度 (預設: 256)
--device          # 運算裝置 (預設: auto)
```

#### **生成腳本 (generate_data.py) 🆕**
```bash
--type               # 白血病類型
--balanced           # 生成平衡資料集
--samples            # 樣本數量 (單一類型)
--samples-per-class  # 每類樣本數 (平衡資料集)
--format             # 輸出格式 (csv/pkl/both)
--output             # 自訂檔名前綴 (預設: generated_data)
--output-dir         # 輸出目錄
--no-plots           # 🆕 跳過視覺化圖表生成
```

#### **🆕 分析腳本 (analyze_dataset.py)**
```bash
python analyze_dataset.py <dataset_path> [選項]

必要參數:
  dataset_path              資料集檔案路徑 (.csv 或 .pkl)

可選參數:
  --output-dir DIR          輸出目錄 (預設: datasets/plots)
  --skip-pca               跳過 PCA 分析
  --skip-tsne              跳過 t-SNE 分析
  --skip-umap              跳過 UMAP 分析
  --skip-correlation       跳過相關性分析
  --skip-statistics        跳過統計分析
  --max-corr-features N    相關性分析最大特徵數 (預設: 100)
```

### **Python API 進階功能**
```python
# 檢查格式兼容性
generator = LeukemiaCVAEGenerator()
print(f"Gene names loaded: {len(generator.gene_names)}")
print(f"First 5 genes: {generator.gene_names[:5]}")

# 批次生成多種類型
for leukemia_type in ['AML', 'PB', 'Bone_Marrow']:
    data = generator.generate_samples(leukemia_type, n_samples=50)
    # 資料格式與原始資料完全相同
```

### **🆕 專業分析進階用法**
```python
from analyze_dataset import DatasetAnalyzer

# 初始化分析器
analyzer = DatasetAnalyzer(output_base_dir='custom_analysis')

# 自訂分析參數
analyzer.tsne_params = {'n_components': 2, 'perplexity': 50}
analyzer.umap_params = {'n_neighbors': 30, 'min_dist': 0.1}

# 執行分析
output_dir = analyzer.analyze_dataset(
    'datasets/my_data.csv',
    include_umap=True,
    max_correlation_features=200
)
```