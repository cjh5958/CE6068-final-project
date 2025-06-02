# 🧬 白血病基因表達資料增強專案 (Leukemia Gene Expression Data Augmentation)

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
│       └── *_correlation_heatmap.png    # 相關性熱力圖
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
pip install torch torchvision pandas numpy scikit-learn matplotlib seaborn
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

### **方式 3: Python API**

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

## 📊 輸出格式

### **🆕 視覺化分析圖表**
每次生成資料時，系統會自動創建以下分析圖表：

1. **類別分布圓餅圖** - 顯示各白血病類型的樣本分布
2. **統計摘要圖** - 包含樣本均值、變異數、基因表達分布等
3. **PCA分析圖** - 主成分分析降維可視化，顯示類別聚集情況
4. **t-SNE分析圖** - 非線性降維可視化 (樣本數 ≤ 1000 時)
5. **相關性熱力圖** - 前50個基因的表達相關性矩陣

### **CSV 格式 (完全匹配原始資料)**
生成的資料將完全匹配原始輸入格式，包括：
```csv
samples,type,1007_s_at,1053_at,117_at,121_at,1255_g_at,1294_at,1316_at,1320_at,...
1000,AML,7.234,8.456,6.123,9.789,4.567,7.890,8.234,7.567,...
1001,AML,7.456,8.234,6.345,9.456,4.789,7.567,8.456,7.789,...
1002,Bone_Marrow,6.789,7.123,5.890,8.234,4.123,6.789,7.123,6.456,...
...
```

### **特點**
- ✅ **samples欄位**: 自動生成樣本ID (從1000開始)
- ✅ **type欄位**: 使用原始白血病類型名稱
- ✅ **基因欄位**: 使用真實基因ID (如 1007_s_at, 1053_at...)
- ✅ **欄位順序**: 完全匹配原始資料欄位順序
- ✅ **直接使用**: 可直接用於下游分析，無需額外處理
- ✅ **🆕 品質保證**: 附帶視覺化分析確保資料品質

### **檔案命名規則 🆕**
```
datasets/
├── generated_data_20250602_143022.csv                    # 平衡資料集
├── generated_data_complete_20250602_143022.pkl           # 完整PKL格式
├── generated_data_aml_20250602_143530.csv               # 特定類型資料
├── generated_data_aml_complete_20250602_143530.pkl      # 特定類型PKL
└── plots/
    ├── generated_data_class_distribution.png             # 類別分布圖
    ├── generated_data_statistics_summary.png             # 統計摘要
    ├── generated_data_pca_analysis.png                   # PCA分析
    ├── generated_data_tsne_analysis.png                  # t-SNE分析
    └── generated_data_correlation_heatmap.png            # 相關性分析
```

### **PKL 格式**
```python
{
    'features': numpy.ndarray,      # 基因表達特徵 (n_samples, 22283)
    'labels': numpy.ndarray,        # 數值標籤 
    'class_names': list,            # 類別名稱
    'generation_time': str,         # 生成時間
    'metadata': {
        'gene_names': list,         # 真實基因名稱
        'total_genes': int,         # 基因總數
        'available_classes': list   # 可用類別
    }
}
```

## ⚙️ 系統需求

### **軟體需求**
- Python 3.8+
- PyTorch 2.0+
- 其他依賴: pandas, numpy, scikit-learn, matplotlib, seaborn

### **硬體需求**
- **CPU**: 現代多核心處理器
- **RAM**: 建議 8GB 以上 (訓練時需要 16GB+)
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
pip install matplotlib seaborn
```

### **環境問題**

#### 🔴 依賴套件問題
```bash
# 安裝完整依賴
pip install torch pandas numpy scikit-learn matplotlib seaborn

# 或使用conda
conda install pytorch pandas numpy scikit-learn matplotlib seaborn
```

## 📈 使用流程建議

### **新用戶 (從零開始)**
1. **準備環境**: 建立虛擬環境並安裝依賴
2. **準備資料**: 確保 `datasets/raw_data.csv` 存在
3. **快速訓練**: `python train_model.py --epochs 10`
4. **生成並分析**: `python generate_data.py --balanced --samples-per-class 20`
5. **檢查品質**: 查看 `datasets/plots/` 中的分析圖表
6. **正式訓練**: `python train_model.py` (使用完整參數)

### **現有模型用戶**
1. **檢查模型**: 確保模型和metadata檔案存在
2. **直接生成**: `python generate_data.py --type AML --samples 100`
3. **品質分析**: 檢查自動生成的視覺化圖表
4. **下游使用**: 生成的資料可直接用於機器學習模型

### **🆕 品質評估工作流程**
1. **生成資料**: 使用 `generate_data.py` 生成合成資料
2. **檢查分布**: 查看類別分布圓餅圖確認平衡性
3. **分析聚集**: 使用 PCA/t-SNE 圖檢查類別分離度
4. **評估相關性**: 檢查基因表達相關性是否合理
5. **統計驗證**: 比較生成資料與原始資料的統計特性

### **協作開發**
1. **環境隔離**: 每個開發者建立自己的虛擬環境
2. **版本控制**: 僅追蹤程式碼，不包含模型檔案和虛擬環境
3. **資料標準**: 統一使用原始資料格式進行交換
4. **🆕 品質標準**: 使用視覺化圖表建立資料品質評估標準

## 📚 技術細節

### **模型架構**
- **類型**: 條件變分自編碼器 (Conditional VAE)
- **潛在空間維度**: 256 (可調整)
- **編碼器**: [2048, 1024, 512, 256] 隱藏層
- **解碼器**: [256, 512, 1024, 2048] 隱藏層
- **條件維度**: 5 (對應 5 種白血病類型)

### **訓練資料**
- **原始樣本**: 64 個樣本 × 22,283 個基因
- **資料來源**: Kaggle 白血病基因表達資料集
- **前處理**: 標準化 + 標籤編碼
- **資料分割**: 訓練/驗證/測試 = 73%/12%/15%

### **🆕 視覺化分析技術**
- **PCA**: 主成分分析，保留前2個主成分用於2D可視化
- **t-SNE**: 非線性降維技術，適合發現非線性聚集模式
- **相關性分析**: 使用皮爾森相關係數計算基因間相關性
- **統計摘要**: 包含均值、變異數、分布直方圖等

### **格式兼容性**
- **完全匹配**: 生成資料與原始資料格式100%相同
- **基因名稱**: 保留真實基因探針ID (如 1007_s_at)
- **樣本編號**: 自動生成不衝突的樣本ID
- **即用性**: 無需額外處理即可用於下游分析

## 📞 支援與進階使用

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

### **🆕 視覺化分析範例**
```python
# 生成資料後分析圖表
import matplotlib.pyplot as plt
import seaborn as sns

# 檢查生成的圖表
plots_dir = 'datasets/plots'
print("Generated analysis plots:")
for plot_file in os.listdir(plots_dir):
    if plot_file.endswith('.png'):
        print(f"  - {plot_file}")
```