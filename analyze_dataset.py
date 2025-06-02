#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Dataset Analysis Tool for Leukemia Gene Expression Data
支援對已生成的資料集進行全面的品質分析和視覺化檢測
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.stats import pearsonr, spearmanr, ks_2samp
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 嘗試導入 UMAP（如果沒安裝會給出友善提示）
try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️  UMAP 未安裝，將跳過 UMAP 分析。安裝指令: pip install umap-learn")

class DatasetAnalyzer:
    """
    全面的資料集分析器
    支援 PCA、t-SNE、UMAP、相關性分析等多維度檢測
    """
    
    def __init__(self, output_base_dir: str = "datasets/plots"):
        self.output_base_dir = output_base_dir
        self.analysis_results = {}
        
        # 設定字型 (移除中文字型設定以避免字體問題)
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 分析參數
        self.pca_components = 2
        self.tsne_params = {'n_components': 2, 'random_state': 42, 'perplexity': 30}
        self.umap_params = {'n_components': 2, 'random_state': 42, 'n_neighbors': 15}
        
        # 顏色配置
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', 
                      '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43']
    
    def load_dataset(self, dataset_path: str) -> tuple:
        """
        載入資料集（支援 CSV 和 PKL 格式）
        
        Returns:
            features: 基因表達特徵矩陣
            labels: 類別標籤
            label_names: 類別名稱
            dataset_name: 資料集名稱
        """
        print(f"📁 載入資料集: {dataset_path}")
        
        # 提取資料集名稱
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
            
            # 分離特徵和標籤
            if 'type' in df.columns:
                features = df.drop(['type', 'samples'], axis=1, errors='ignore').values
                labels = df['type'].values
                unique_labels = np.unique(labels)
                
                # 將類別名稱轉換為數值
                label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
                numeric_labels = np.array([label_mapping[label] for label in labels])
                
                return features, numeric_labels, unique_labels, dataset_name
            else:
                raise ValueError("CSV 檔案必須包含 'type' 欄位")
                
        elif dataset_path.endswith('.pkl'):
            with open(dataset_path, 'rb') as f:
                data = pickle.load(f)
            
            features = data['features']
            labels = data['labels']
            
            if 'class_names' in data:
                label_names = data['class_names']
            else:
                label_names = [f'Class_{i}' for i in range(len(np.unique(labels)))]
            
            return features, labels, label_names, dataset_name
        else:
            raise ValueError("僅支援 .csv 和 .pkl 格式")
    
    def create_output_directory(self, dataset_name: str) -> str:
        """
        創建以資料集名稱命名的輸出目錄
        """
        output_dir = os.path.join(self.output_base_dir, f"{dataset_name}_analysis")
        os.makedirs(output_dir, exist_ok=True)
        print(f"📂 分析結果將保存至: {output_dir}")
        return output_dir
    
    def perform_pca_analysis(self, features: np.ndarray, labels: np.ndarray, 
                           label_names: list, output_dir: str):
        """
        進行 PCA 分析並生成視覺化
        """
        print("🔍 執行 PCA 分析...")
        
        # 標準化特徵
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # PCA 降維
        pca = PCA(n_components=min(50, features.shape[1]))  # 最多計算50個主成分
        pca_features = pca.fit_transform(features_scaled)
        
        # 儲存 PCA 結果
        self.analysis_results['pca'] = {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'components': pca.components_,
            'features_2d': pca_features[:, :2]
        }
        
        # 創建 PCA 分析圖表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('PCA Principal Component Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. 2D PCA 散點圖
        ax1 = axes[0, 0]
        for i, label_name in enumerate(label_names):
            mask = labels == i
            if np.any(mask):
                ax1.scatter(pca_features[mask, 0], pca_features[mask, 1], 
                           c=self.colors[i % len(self.colors)], label=label_name, 
                           alpha=0.7, s=50)
        
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} Variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} Variance)')
        ax1.set_title('PCA 2D Projection')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 解釋變異比例
        ax2 = axes[0, 1]
        n_components = min(20, len(pca.explained_variance_ratio_))
        ax2.bar(range(1, n_components+1), pca.explained_variance_ratio_[:n_components])
        ax2.set_xlabel('Principal Component Number')
        ax2.set_ylabel('Explained Variance Ratio')
        ax2.set_title('Individual Component Variance')
        ax2.grid(True, alpha=0.3)
        
        # 3. 累積解釋變異
        ax3 = axes[1, 0]
        ax3.plot(range(1, n_components+1), 
                np.cumsum(pca.explained_variance_ratio_[:n_components]), 
                'bo-', linewidth=2, markersize=6)
        ax3.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% Variance')
        ax3.set_xlabel('Number of Components')
        ax3.set_ylabel('Cumulative Explained Variance')
        ax3.set_title('Cumulative Variance Explained')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 類別分離度評估
        ax4 = axes[1, 1]
        if len(label_names) > 1:
            silhouette_avg = silhouette_score(pca_features[:, :2], labels)
            ax4.text(0.5, 0.7, f'Silhouette Score: {silhouette_avg:.3f}', 
                    ha='center', va='center', fontsize=14, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            ax4.text(0.5, 0.5, f'First 2 PCs Variance: {np.sum(pca.explained_variance_ratio_[:2]):.2%}', 
                    ha='center', va='center', fontsize=12)
            ax4.text(0.5, 0.3, f'Sample Count: {features.shape[0]}', 
                    ha='center', va='center', fontsize=12)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.set_title('PCA Quality Assessment')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pca_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ PCA 分析完成 - 前2個主成分解釋 {np.sum(pca.explained_variance_ratio_[:2]):.2%} 的變異")
    
    def perform_tsne_analysis(self, features: np.ndarray, labels: np.ndarray, 
                            label_names: list, output_dir: str):
        """
        進行 t-SNE 分析並生成視覺化
        """
        print("🔍 執行 t-SNE 分析...")
        
        # 限制樣本數量以避免 t-SNE 計算過慢
        if features.shape[0] > 1000:
            print(f"⚠️  樣本數過多 ({features.shape[0]})，隨機採樣1000個樣本進行 t-SNE 分析")
            indices = np.random.choice(features.shape[0], 1000, replace=False)
            features_sample = features[indices]
            labels_sample = labels[indices]
        else:
            features_sample = features
            labels_sample = labels
        
        # 標準化特徵
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_sample)
        
        # t-SNE 降維
        tsne = TSNE(**self.tsne_params)
        tsne_features = tsne.fit_transform(features_scaled)
        
        # 儲存 t-SNE 結果
        self.analysis_results['tsne'] = {
            'features_2d': tsne_features,
            'labels': labels_sample
        }
        
        # 創建 t-SNE 分析圖表
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('t-SNE Nonlinear Dimensionality Reduction Results', fontsize=16, fontweight='bold')
        
        # 1. t-SNE 散點圖
        ax1 = axes[0]
        for i, label_name in enumerate(label_names):
            mask = labels_sample == i
            if np.any(mask):
                ax1.scatter(tsne_features[mask, 0], tsne_features[mask, 1], 
                           c=self.colors[i % len(self.colors)], label=label_name, 
                           alpha=0.7, s=50)
        
        ax1.set_xlabel('t-SNE 1')
        ax1.set_ylabel('t-SNE 2')
        ax1.set_title('t-SNE 2D Projection')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 品質評估
        ax2 = axes[1]
        if len(label_names) > 1:
            silhouette_avg = silhouette_score(tsne_features, labels_sample)
            ax2.text(0.5, 0.7, f'Silhouette Score: {silhouette_avg:.3f}', 
                    ha='center', va='center', fontsize=14, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            ax2.text(0.5, 0.5, f'Perplexity: {self.tsne_params["perplexity"]}', 
                    ha='center', va='center', fontsize=12)
            ax2.text(0.5, 0.3, f'Sample Count: {features_sample.shape[0]}', 
                    ha='center', va='center', fontsize=12)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_title('t-SNE Quality Assessment')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'tsne_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ t-SNE 分析完成")
    
    def perform_umap_analysis(self, features: np.ndarray, labels: np.ndarray, 
                            label_names: list, output_dir: str):
        """
        進行 UMAP 分析並生成視覺化
        """
        if not UMAP_AVAILABLE:
            print("⏭️  跳過 UMAP 分析（未安裝 umap-learn）")
            return
        
        print("🔍 執行 UMAP 分析...")
        
        # 標準化特徵
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # UMAP 降維
        reducer = umap.UMAP(**self.umap_params)
        umap_features = reducer.fit_transform(features_scaled)
        
        # 儲存 UMAP 結果
        self.analysis_results['umap'] = {
            'features_2d': umap_features,
            'labels': labels
        }
        
        # 創建 UMAP 分析圖表
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('UMAP Uniform Manifold Approximation Results', fontsize=16, fontweight='bold')
        
        # 1. UMAP 散點圖
        ax1 = axes[0]
        for i, label_name in enumerate(label_names):
            mask = labels == i
            if np.any(mask):
                ax1.scatter(umap_features[mask, 0], umap_features[mask, 1], 
                           c=self.colors[i % len(self.colors)], label=label_name, 
                           alpha=0.7, s=50)
        
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')
        ax1.set_title('UMAP 2D Projection')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 品質評估
        ax2 = axes[1]
        if len(label_names) > 1:
            silhouette_avg = silhouette_score(umap_features, labels)
            ax2.text(0.5, 0.7, f'Silhouette Score: {silhouette_avg:.3f}', 
                    ha='center', va='center', fontsize=14, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
            ax2.text(0.5, 0.5, f'Neighbors: {self.umap_params["n_neighbors"]}', 
                    ha='center', va='center', fontsize=12)
            ax2.text(0.5, 0.3, f'Sample Count: {features.shape[0]}', 
                    ha='center', va='center', fontsize=12)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_title('UMAP Quality Assessment')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'umap_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ UMAP 分析完成")
    
    def perform_correlation_analysis(self, features: np.ndarray, output_dir: str, 
                                   max_features: int = 100):
        """
        進行相關性分析並生成熱力圖
        """
        print("🔍 執行相關性分析...")
        
        # 如果特徵太多，選擇方差最大的特徵進行分析
        if features.shape[1] > max_features:
            feature_vars = np.var(features, axis=0)
            top_indices = np.argsort(feature_vars)[-max_features:]
            features_selected = features[:, top_indices]
            print(f"📊 選擇方差最大的 {max_features} 個特徵進行相關性分析")
        else:
            features_selected = features
            top_indices = np.arange(features.shape[1])
        
        # 計算皮爾森相關係數
        correlation_matrix = np.corrcoef(features_selected.T)
        
        # 儲存相關性結果
        self.analysis_results['correlation'] = {
            'correlation_matrix': correlation_matrix,
            'selected_features': top_indices
        }
        
        # 創建相關性分析圖表
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle('Feature Correlation Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. 相關性熱力圖
        ax1 = axes[0, 0]
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', 
                   center=0, square=True, ax=ax1, cbar_kws={"shrink": .8})
        ax1.set_title('Feature Correlation Heatmap')
        
        # 2. 相關性分布直方圖
        ax2 = axes[0, 1]
        corr_values = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
        ax2.hist(corr_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(corr_values), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(corr_values):.3f}')
        ax2.set_xlabel('Correlation Coefficient')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Correlation Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 高相關性特徵對
        ax3 = axes[1, 0]
        high_corr_threshold = 0.8
        high_corr_pairs = np.where((np.abs(correlation_matrix) > high_corr_threshold) & 
                                  (correlation_matrix != 1.0))
        
        if len(high_corr_pairs[0]) > 0:
            high_corr_values = correlation_matrix[high_corr_pairs]
            ax3.scatter(range(len(high_corr_values)), high_corr_values, 
                       c='red', alpha=0.7, s=50)
            ax3.axhline(y=high_corr_threshold, color='orange', linestyle='--', 
                       label=f'Threshold: {high_corr_threshold}')
            ax3.axhline(y=-high_corr_threshold, color='orange', linestyle='--')
            ax3.set_xlabel('Feature Pair Index')
            ax3.set_ylabel('Correlation Coefficient')
            ax3.set_title(f'High Correlation Pairs (|r| > {high_corr_threshold})')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 顯示統計信息
            ax3.text(0.02, 0.98, f'High Corr Pairs: {len(high_corr_values)}', 
                    transform=ax3.transAxes, va='top', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        else:
            ax3.text(0.5, 0.5, 'No High Correlation Pairs', ha='center', va='center', 
                    fontsize=14, transform=ax3.transAxes)
            ax3.set_title(f'High Correlation Pairs (|r| > {high_corr_threshold})')
        
        # 4. 統計摘要
        ax4 = axes[1, 1]
        stats_text = f"""
Correlation Statistics Summary:

• Features Analyzed: {features_selected.shape[1]}
• Mean Correlation: {np.mean(corr_values):.4f}
• Std Correlation: {np.std(corr_values):.4f}
• Max Correlation: {np.max(corr_values):.4f}
• Min Correlation: {np.min(corr_values):.4f}
• High Pairs (|r|>0.8): {len(high_corr_pairs[0])//2}
• Medium Pairs (0.5<|r|<0.8): {len(np.where((np.abs(corr_values) > 0.5) & (np.abs(corr_values) <= 0.8))[0])}
        """
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, va='top', 
                fontsize=11, fontfamily='monospace')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ 相關性分析完成")
    
    def perform_statistical_analysis(self, features: np.ndarray, labels: np.ndarray, 
                                   label_names: list, output_dir: str):
        """
        進行統計特性分析
        """
        print("🔍 執行統計特性分析...")
        
        # 創建統計分析圖表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Dataset Statistical Analysis', fontsize=16, fontweight='bold')
        
        # 1. 類別分布
        ax1 = axes[0, 0]
        unique_labels, counts = np.unique(labels, return_counts=True)
        colors_subset = [self.colors[i % len(self.colors)] for i in range(len(unique_labels))]
        
        bars = ax1.bar([label_names[i] for i in unique_labels], counts, color=colors_subset)
        ax1.set_xlabel('Leukemia Type')
        ax1.set_ylabel('Sample Count')
        ax1.set_title('Class Distribution')
        ax1.tick_params(axis='x', rotation=45)
        
        # 在柱狀圖上標註數值
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom')
        
        # 2. 特徵分布直方圖
        ax2 = axes[0, 1]
        feature_means = np.mean(features, axis=0)
        ax2.hist(feature_means, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('Feature Mean Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Feature Mean Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. 特徵變異性
        ax3 = axes[0, 2]
        feature_stds = np.std(features, axis=0)
        ax3.hist(feature_stds, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.set_xlabel('Feature Standard Deviation')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Feature Variability Distribution')
        ax3.grid(True, alpha=0.3)
        
        # 4. 樣本間距離分布
        ax4 = axes[1, 0]
        # 隨機採樣計算距離（避免計算量過大）
        if features.shape[0] > 100:
            sample_indices = np.random.choice(features.shape[0], 100, replace=False)
            features_sample = features[sample_indices]
        else:
            features_sample = features
        
        from sklearn.metrics import pairwise_distances
        distances = pairwise_distances(features_sample, metric='euclidean')
        distance_values = distances[np.triu_indices_from(distances, k=1)]
        
        ax4.hist(distance_values, bins=50, alpha=0.7, color='gold', edgecolor='black')
        ax4.set_xlabel('Euclidean Distance')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Inter-sample Distance Distribution')
        ax4.grid(True, alpha=0.3)
        
        # 5. 各類別特徵均值比較（箱線圖）
        ax5 = axes[1, 1]
        if len(label_names) <= 10:  # 避免圖表過於擁擠
            # 選擇方差最大的前10個特徵
            feature_vars = np.var(features, axis=0)
            top_feature_indices = np.argsort(feature_vars)[-10:]
            
            data_for_box = []
            labels_for_box = []
            
            for class_idx in unique_labels:
                class_mask = labels == class_idx
                class_features = features[class_mask][:, top_feature_indices]
                data_for_box.extend(class_features.flatten())
                labels_for_box.extend([label_names[class_idx]] * class_features.size)
            
            # 創建DataFrame以便使用seaborn
            import pandas as pd
            df_box = pd.DataFrame({
                'value': data_for_box,
                'class': labels_for_box
            })
            
            sns.boxplot(data=df_box, x='class', y='value', ax=ax5)
            ax5.set_xlabel('Leukemia Type')
            ax5.set_ylabel('Feature Value')
            ax5.set_title('Feature Value Distribution by Class')
            ax5.tick_params(axis='x', rotation=45)
        else:
            ax5.text(0.5, 0.5, 'Too Many Classes for Box Plot', ha='center', va='center', 
                    transform=ax5.transAxes, fontsize=12)
        
        # 6. 資料品質評估
        ax6 = axes[1, 2]
        
        # 計算品質指標
        n_samples, n_features = features.shape
        feature_completeness = 1.0  # 假設沒有缺失值
        class_balance = 1 - np.std(counts) / np.mean(counts) if len(counts) > 1 else 1.0
        
        quality_metrics = [
            ('Sample Count', n_samples),
            ('Feature Count', n_features),
            ('Class Count', len(unique_labels)),
            ('Class Balance', f'{class_balance:.3f}'),
            ('Data Completeness', f'{feature_completeness:.3f}'),
            ('Mean Feature Value', f'{np.mean(features):.3f}'),
            ('Feature Std Dev', f'{np.std(features):.3f}')
        ]
        
        y_pos = 0.9
        for metric, value in quality_metrics:
            ax6.text(0.05, y_pos, f'{metric}: {value}', transform=ax6.transAxes, 
                    fontsize=11, fontfamily='monospace')
            y_pos -= 0.12
        
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.set_title('Data Quality Metrics')
        ax6.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'statistical_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ 統計特性分析完成")
    
    def compare_dimensionality_reduction(self, output_dir: str):
        """
        比較不同降維方法的效果
        """
        if 'pca' not in self.analysis_results:
            print("⚠️  無法進行降維方法比較，缺少 PCA 結果")
            return
        
        print("🔍 比較不同降維方法...")
        
        available_methods = []
        if 'pca' in self.analysis_results:
            available_methods.append('PCA')
        if 'tsne' in self.analysis_results:
            available_methods.append('t-SNE')
        if 'umap' in self.analysis_results:
            available_methods.append('UMAP')
        
        if len(available_methods) <= 1:
            print("⚠️  可用降維方法少於2種，跳過比較分析")
            return
        
        # 創建比較圖表
        fig, axes = plt.subplots(1, len(available_methods), figsize=(6*len(available_methods), 6))
        if len(available_methods) == 1:
            axes = [axes]
        
        fig.suptitle('Dimensionality Reduction Methods Comparison', fontsize=16, fontweight='bold')
        
        # 獲取類別名稱
        label_names = self.analysis_results.get('label_names', [])
        
        for idx, method in enumerate(available_methods):
            ax = axes[idx]
            
            if method == 'PCA':
                features_2d = self.analysis_results['pca']['features_2d']
                labels = self.analysis_results.get('labels', np.array([]))
            elif method == 't-SNE':
                features_2d = self.analysis_results['tsne']['features_2d']
                labels = self.analysis_results['tsne']['labels']
            elif method == 'UMAP':
                features_2d = self.analysis_results['umap']['features_2d']
                labels = self.analysis_results['umap']['labels']
            
            # 繪製散點圖
            unique_labels = np.unique(labels)
            for i, label_idx in enumerate(unique_labels):
                mask = labels == label_idx
                if np.any(mask):
                    # 使用實際類別名稱而不是 Class 編號
                    label_name = label_names[label_idx] if label_idx < len(label_names) else f'Class_{label_idx}'
                    ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                             c=self.colors[i % len(self.colors)], 
                             alpha=0.7, s=30, label=label_name)
            
            ax.set_title(f'{method} Results')
            ax.set_xlabel(f'{method} 1')
            ax.set_ylabel(f'{method} 2')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dimensionality_reduction_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ 降維方法比較完成")
    
    def analyze_dataset(self, dataset_path: str, 
                       include_pca: bool = True,
                       include_tsne: bool = True, 
                       include_umap: bool = True,
                       include_correlation: bool = True,
                       include_statistics: bool = True,
                       max_correlation_features: int = 100):
        """
        對資料集進行全面分析
        """
        print("🚀 開始資料集全面分析...")
        print("=" * 60)
        
        # 載入資料集
        features, labels, label_names, dataset_name = self.load_dataset(dataset_path)
        
        # 創建輸出目錄
        output_dir = self.create_output_directory(dataset_name)
        
        # 儲存標籤信息供後續使用
        self.analysis_results['labels'] = labels
        self.analysis_results['label_names'] = label_names
        
        print(f"\n📊 資料集資訊:")
        print(f"   樣本數: {features.shape[0]}")
        print(f"   特徵數: {features.shape[1]}")
        print(f"   類別數: {len(label_names)}")
        print(f"   類別名稱: {label_names}")
        print("=" * 60)
        
        # 執行各項分析
        if include_pca:
            self.perform_pca_analysis(features, labels, label_names, output_dir)
        
        if include_tsne:
            self.perform_tsne_analysis(features, labels, label_names, output_dir)
        
        if include_umap:
            self.perform_umap_analysis(features, labels, label_names, output_dir)
        
        if include_correlation:
            self.perform_correlation_analysis(features, output_dir, max_correlation_features)
        
        if include_statistics:
            self.perform_statistical_analysis(features, labels, label_names, output_dir)
        
        # 比較降維方法
        self.compare_dimensionality_reduction(output_dir)
        
        print("\n" + "=" * 60)
        print("🎉 資料集分析完成！")
        print(f"📂 結果保存在: {output_dir}")
        print("=" * 60)
        
        return output_dir

def main():
    parser = argparse.ArgumentParser(description='全面的資料集分析工具')
    parser.add_argument('dataset_path', help='資料集路徑 (.csv 或 .pkl)')
    parser.add_argument('--output-dir', default='datasets/plots', help='輸出目錄')
    parser.add_argument('--skip-pca', action='store_true', help='跳過 PCA 分析')
    parser.add_argument('--skip-tsne', action='store_true', help='跳過 t-SNE 分析')
    parser.add_argument('--skip-umap', action='store_true', help='跳過 UMAP 分析')
    parser.add_argument('--skip-correlation', action='store_true', help='跳過相關性分析')
    parser.add_argument('--skip-statistics', action='store_true', help='跳過統計分析')
    parser.add_argument('--max-corr-features', type=int, default=100, 
                       help='相關性分析最大特徵數')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"❌ 錯誤: 找不到資料集檔案 {args.dataset_path}")
        return
    
    # 初始化分析器
    analyzer = DatasetAnalyzer(args.output_dir)
    
    # 執行分析
    try:
        output_dir = analyzer.analyze_dataset(
            dataset_path=args.dataset_path,
            include_pca=not args.skip_pca,
            include_tsne=not args.skip_tsne,
            include_umap=not args.skip_umap,
            include_correlation=not args.skip_correlation,
            include_statistics=not args.skip_statistics,
            max_correlation_features=args.max_corr_features
        )
        
        print(f"\n✨ 分析完成！請查看結果目錄: {output_dir}")
        
    except Exception as e:
        print(f"❌ 分析過程中發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 