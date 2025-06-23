import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')  # Sử dụng backend không tương tác để tránh lỗi tkinter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.inspection import permutation_importance
import lightgbm as lgb
import joblib
import os
import rasterio
from pathlib import Path
import time
from scipy.stats import randint, uniform
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.stats import t
import logging
from typing import Dict, List, Tuple, Optional, Callable

# Cấu hình logging với mã hóa UTF-8
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('du_bao_ngap.log', encoding='utf-8'),  # Thêm encoding='utf-8'
        logging.StreamHandler()
    ]
)

# Đặt mã hóa UTF-8 cho console
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
logger = logging.getLogger(__name__)

# Tham số dữ liệu
BASENAMES = [
    'eleStream', 'disStream', 'wSlope', 'streamSlope', 'flowLength', 'area',
    'twi', 'spi', 'tpi', 'wNdvi', 'wCN', 'wInfiRate', 'dem', 'wDem',
    'profCurvature', 'planCurvature'
]
PRECIPS = ['raster_max', 'raster_3h_max', 'raster_6h_max', 'raster_24h_max']
BASE_RASTER_FOLDER = 'rasters'
PARAMS = BASENAMES + PRECIPS
GROUP_NAMES = ['Nguy cơ rất thấp', 'Nguy cơ thấp', 'Nguy cơ trung bình', 'Nguy cơ cao']

def encrypt(plaintext: str, length: int = 10) -> str:
    """Mã hóa chuỗi bằng hàm băm SHAKE-256."""
    import hashlib
    return hashlib.shake_256(plaintext.encode('utf-8')).hexdigest(int(round(length/2)))[:length]

def robust_scaling(X: np.ndarray) -> np.ndarray:
    """Áp dụng chuẩn hóa mạnh mẽ sử dụng trung vị và IQR."""
    scaler = RobustScaler()
    return scaler.fit_transform(X)

def log_transformation(X: np.ndarray) -> np.ndarray:
    """Áp dụng biến đổi log (log(X + 1)) để xử lý độ lệch và giá trị 0."""
    if np.any(X < 0):
        raise ValueError("Dữ liệu chứa giá trị âm, không thể áp dụng biến đổi log.")
    return np.log1p(X)

def minmax_scaling(X: np.ndarray) -> np.ndarray:
    """Áp dụng chuẩn hóa MinMax để đưa dữ liệu về khoảng [0, 1]."""
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)

def zscore_scaling(X: np.ndarray) -> np.ndarray:
    """Áp dụng chuẩn hóa Z-score (mean=0, std=1)."""
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def get_scaling_func(name: str) -> List[Callable[[np.ndarray], np.ndarray]]:
    """Xác định hàm chuẩn hóa cho tham số cụ thể."""
    if name in ['eleStream', 'wInfiRate', 'dem', 'wDem']:
        return [robust_scaling, minmax_scaling]
    elif name in ['disStream', 'flowLength', 'area', 'spi'] + PRECIPS:
        return [log_transformation, minmax_scaling]
    elif name in ['tpi', 'profCurvature', 'planCurvature']:
        return [zscore_scaling, minmax_scaling]
    return [minmax_scaling]

def load_base_data(events_dir: str = 'Events/2023') -> Tuple[Dict[str, np.ndarray], Dict]:
    """Tải và tiền xử lý dữ liệu raster."""
    data = {}
    profile = None
    try:
        files = [f for f in os.listdir(BASE_RASTER_FOLDER) if f.endswith('.tif')]
        for file in files:
            file_path = os.path.join(BASE_RASTER_FOLDER, file)
            name = file.replace('.tif', '').split('_')[1]
            with rasterio.open(file_path, 'r') as src:
                d = src.read(1)
                for func in get_scaling_func(name):
                    d = func(d)
                data[name] = d

        for precip in PRECIPS:
            file_path = os.path.join(events_dir, f'{precip}.tif')
            with rasterio.open(file_path) as src:
                d = src.read(1)
                for func in get_scaling_func(precip):
                    d = func(d)
                data[precip] = d

        with rasterio.open('hazard_points.tif', 'r') as src:
            profile = src.profile

        logger.info(f"Tải và tiền xử lý dữ liệu raster từ {events_dir} thành công")
        return data, profile

    except Exception as e:
        logger.error(f"Lỗi khi tải dữ liệu cơ sở từ {events_dir}: {str(e)}")
        raise

def create_dataframes(data: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Tạo DataFrame cho dữ liệu đào tạo và nhãn đã biến đổi."""
    try:
        with rasterio.open('hazard_points.tif', 'r') as src:
            hazard = src.read(1)

        indices = np.where(hazard > 0)
        df = pd.DataFrame({
            'r': indices[0],
            'c': indices[1]
        })
        for param in PARAMS:
            df[param] = data[param][indices]
        df['label'] = hazard[indices]

        df2 = df.copy()
        label_map = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4}
        df2['label'] = df2['label'].map(label_map)
        logger.info(f"Tạo DataFrame với nhãn gốc và nhãn biến đổi ({', '.join(GROUP_NAMES)})")
        return df, df2

    except Exception as e:
        logger.error(f"Lỗi khi tạo DataFrame: {str(e)}")
        raise

def save_train_test_tif(df: pd.DataFrame, profile: Dict, output_dir: str, train_indices: pd.Index, test_indices: pd.Index) -> None:
    """Lưu dữ liệu đào tạo và kiểm tra thành file train.tif và test.tif."""
    height, width = profile['height'], profile['width']
    train_data = np.full((height, width), profile.get('nodata', -9999), dtype=np.float32)
    test_data = np.full((height, width), profile.get('nodata', -9999), dtype=np.float32)
    
    for _, row in df.loc[train_indices].iterrows():
        r, c = int(row['r']), int(row['c'])
        if 0 <= r < height and 0 <= c < width:
            train_data[r, c] = row['label']
    
    for _, row in df.loc[test_indices].iterrows():
        r, c = int(row['r']), int(row['c'])
        if 0 <= r < height and 0 <= c < width:
            test_data[r, c] = row['label']
    
    profile.update(dtype=rasterio.float32, count=1, compress='lzw')
    
    train_path = os.path.join(output_dir, 'train.tif')
    with rasterio.open(train_path, 'w', **profile) as dst:
        dst.write(train_data, 1)
    logger.info(f"Lưu dữ liệu đào tạo vào: {train_path}")
    
    test_path = os.path.join(output_dir, 'test.tif')
    with rasterio.open(test_path, 'w', **profile) as dst:
        dst.write(test_data, 1)
    logger.info(f"Lưu dữ liệu kiểm tra vào: {test_path}")

class FloodPredictionModel:
    """Lớp để huấn luyện và đánh giá các mô hình dự đoán ngập lụt."""
    
    def __init__(self, model_type: str = 'rf', output_dir: str = 'models', n_features: Optional[int] = 18):
        self.model_type = model_type.lower()
        self.output_dir = output_dir
        self.n_features = n_features
        self.model = None
        self.feature_importances = None
        self.feature_selector = None
        self.selected_features = None
        self.scaler = StandardScaler()
        
        os.makedirs(output_dir, exist_ok=True)
        
        model_map = {
            'rf': RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
            'svm': SVC(random_state=42, probability=True, class_weight='balanced'),
            'lr': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced', n_jobs=-1),
            'lgbm': lgb.LGBMClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
            'ensemble': VotingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(random_state=42, n_jobs=-1)),
                    ('lgbm', lgb.LGBMClassifier(random_state=42, n_jobs=-1))
                ], voting='soft', n_jobs=-1
            )
        }
        if self.model_type not in model_map:
            raise ValueError("Loại mô hình phải là 'rf', 'svm', 'lr', 'lgbm', hoặc 'ensemble'.")
        self.model = model_map[self.model_type]
    
    def save_correlation_matrix(self, corr_matrix: pd.DataFrame, output_dir: str) -> None:
        """Lưu ma trận tương quan dưới dạng file .csv và hình ảnh .png."""
        corr_csv_path = os.path.join(output_dir, f"{self.model_type}_correlation_matrix.csv")
        corr_matrix.to_csv(corr_csv_path)
        logger.info(f"Lưu ma trận tương quan vào: {corr_csv_path}")
        
        plt.figure(figsize=(10, 8))
        sns.set_context("paper", font_scale=0.8)
        sns.heatmap(
            corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            xticklabels=corr_matrix.columns,
            yticklabels=corr_matrix.columns,
            annot_kws={"size": 7},
            cbar=True
        )
        plt.title(f'Ma trận tương quan các đặc trưng ({self.model_type.upper()})', fontsize=10)
        plt.tight_layout(pad=0.5)
        corr_png_path = os.path.join(output_dir, f"{self.model_type}_correlation_matrix.png")
        plt.savefig(corr_png_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Lưu hình ảnh ma trận tương quan vào: {corr_png_path}")
    
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X = df.drop(['r', 'c', 'label'], axis=1)
        y = df['label']
        original_columns = X.columns.tolist()
        
        corr_matrix = X.corr().abs()
        self.save_correlation_matrix(corr_matrix, self.output_dir)
        
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
        X = X.drop(to_drop, axis=1)
        logger.info(f"Loại bỏ các đặc trưng có tương quan cao: {to_drop}")
        original_columns = X.columns.tolist()
        
        if self.n_features is not None and len(original_columns) > self.n_features:
            self.feature_selector = SelectKBest(f_classif, k=self.n_features)
            X_transformed = self.feature_selector.fit_transform(X, y)
            self.selected_features = [original_columns[i] for i in np.where(self.feature_selector.get_support())[0]]
            logger.info(f"Chọn {self.n_features} đặc trưng: {self.selected_features}")
            X = pd.DataFrame(X_transformed, columns=self.selected_features, index=X.index)
        else:
            self.selected_features = original_columns
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        if self.model_type in ['svm', 'lr']:
            train_index = X_train.index
            test_index = X_test.index
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            X_train = pd.DataFrame(X_train, columns=self.selected_features, index=train_index)
            X_test = pd.DataFrame(X_test, columns=self.selected_features, index=test_index)
        
        smote = SMOTE(random_state=42, sampling_strategy={1: 8000, 2: 8000, 3: 8000, 4: 8000})
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logger.info(f"Kích thước tập huấn luyện sau SMOTE: {len(X_train)} mẫu")
        
        return X_train, X_test, y_train, y_test
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series, n_iter: int = 3) -> object:
        param_distributions = {
            'rf': {
                'n_estimators': randint(100, 200),
                'max_depth': [15, 20, 25],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 3]
            },
            'svm': {
                'C': uniform(0.1, 20),
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            },
            'lr': {
                'C': uniform(0.1, 20),
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            },
            'lgbm': {
                'num_leaves': randint(30, 70),
                'learning_rate': uniform(0.05, 0.1),
                'n_estimators': randint(100, 300),
                'max_depth': [8, 10, 12]
            },
            'ensemble': {
                'rf__n_estimators': randint(100, 200),
                'rf__max_depth': [15, 20],
                'lgbm__num_leaves': randint(30, 50),
                'lgbm__learning_rate': uniform(0.05, 0.1),
                'lgbm__max_depth': [8, 10]
            }
        }
        
        n_iter_adjusted = 2 if self.model_type == 'svm' else n_iter
        search = RandomizedSearchCV(
            self.model, param_distributions[self.model_type], n_iter=n_iter_adjusted,
            cv=5, n_jobs=-1, verbose=1, random_state=42, scoring='f1_weighted'
        )
        
        logger.info(f"Đang điều chỉnh siêu tham số cho {self.model_type}")
        start_time = time.time()
        search.fit(X_train, y_train)
        logger.info(f"Hoàn thành điều chỉnh trong {time.time() - start_time:.2f} giây. Tham số tốt nhất: {search.best_params_}")
        
        self.model = search.best_estimator_
        return self.model
    
    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        profile: Dict,
        tune_hyperparams: bool = True,
        n_iter: int = 3
    ) -> Tuple[pd.DataFrame, pd.Series, np.ndarray, Dict, np.ndarray]:
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        save_train_test_tif(df, profile, self.output_dir, X_train.index, X_test.index)
        
        logger.info(f"Huấn luyện mô hình {self.model_type}")
        if tune_hyperparams:
            self.hyperparameter_tuning(X_train, y_train, n_iter)
        
        start_time = time.time()
        self.model.fit(X_train, y_train)
        logger.info(f"Hoàn thành huấn luyện trong {time.time() - start_time:.2f} giây")
        
        y_pred_train = cross_val_predict(self.model, X_train, y_train, cv=5, n_jobs=-1)
        y_pred_test = self.model.predict(X_test)
        
        train_indices = X_train.index[:len(y_pred_train)]
        test_indices = X_test.index
        df_train = df.loc[train_indices].copy()
        df_test = df.loc[test_indices].copy()
        df_train['pred'] = y_pred_train
        df_train['type'] = 'train'
        df_test['pred'] = y_pred_test
        df_test['type'] = 'test'
        result_df = pd.concat([df_train, df_test], axis=0)
        
        metrics, cm = self._evaluate_model(y_test, y_pred_test, 'Kiểm tra', X_test)
        self._evaluate_model(y_train[:len(y_pred_train)], y_pred_train, 'Đào tạo', X_train)
        
        self.save_model()
        
        return result_df, y_test, y_pred_test, metrics, cm
    
    def _evaluate_model(self, y_true: pd.Series, y_pred: np.ndarray, dataset_name: str, X: pd.DataFrame) -> Tuple[Dict, np.ndarray]:
        metrics = {
            'Độ chính xác': accuracy_score(y_true, y_pred),
            'Độ chính xác (Precision)': precision_score(y_true, y_pred, average='weighted'),
            'Độ nhạy (Recall)': recall_score(y_true, y_pred, average='weighted'),
            'F1 Score': f1_score(y_true, y_pred, average='weighted')
        }
        
        logger.info(f"\nHiệu suất {dataset_name} ({self.model_type.upper()}):")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        logger.info("\nBáo cáo phân loại:")
        logger.info("\n%s", classification_report(y_true, y_pred, target_names=GROUP_NAMES))
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(4, 3))
        sns.set_context("paper", font_scale=0.8)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=GROUP_NAMES,
            yticklabels=GROUP_NAMES,
            annot_kws={"size": 7},
            cbar=False
        )
        plt.title(f'Ma trận nhầm lẫn - {dataset_name} ({self.model_type.upper()})', fontsize=9)
        plt.xlabel('Dự đoán', fontsize=8)
        plt.ylabel('Thực tế', fontsize=8)
        plt.tick_params(axis='both', labelsize=7)
        plt.tight_layout(pad=0.5)
        plt.savefig(os.path.join(self.output_dir, f"{self.model_type}_cm_{dataset_name.lower().replace(' ', '_')}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        if hasattr(self.model, 'predict_proba'):
            y_score = self.model.predict_proba(X)
            n_classes = len(GROUP_NAMES)
            fpr, tpr, roc_auc = {}, {}, {}
            
            for i, group_name in enumerate(GROUP_NAMES, 1):
                y_true_bin = (y_true == i).astype(int)
                fpr[i], tpr[i], _ = roc_curve(y_true_bin, y_score[:, i-1])
                roc_auc[i] = auc(fpr[i], tpr[i])
                
                plt.figure(figsize=(4, 3))
                sns.set_context("paper", font_scale=0.8)
                plt.plot(fpr[i], tpr[i], color='blue', label=f'{group_name} (AUC = {roc_auc[i]:.2f})')
                plt.plot([0, 1], [0, 1], 'k:', linewidth=1)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('Tỷ lệ dương tính giả', fontsize=8)
                plt.ylabel('Tỷ lệ dương tính thật', fontsize=8)
                plt.title(f'Đường cong ROC - {group_name} ({self.model_type.upper()})', fontsize=9)
                plt.legend(loc='lower right', fontsize=7)
                plt.tick_params(axis='both', labelsize=7)
                plt.tight_layout(pad=0.5)
                plt.savefig(os.path.join(self.output_dir, f"{self.model_type}_roc_{group_name.lower().replace(' ', '_')}_{dataset_name.lower().replace(' ', '_')}.png"), dpi=300, bbox_inches='tight')
                plt.close()
            
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(1, n_classes + 1)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(1, n_classes + 1):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= n_classes
            roc_auc['macro'] = auc(all_fpr, mean_tpr)
            
            plt.figure(figsize=(4, 3))
            sns.set_context("paper", font_scale=0.8)
            colors = sns.color_palette("husl", n_classes)
            for i, (group_name, color) in enumerate(zip(GROUP_NAMES, colors), 1):
                plt.plot(fpr[i], tpr[i], color=color, label=f'{group_name} (AUC = {roc_auc[i]:.2f})')
            plt.plot(all_fpr, mean_tpr, 'k--', label=f'Trung bình macro (AUC = {roc_auc["macro"]:.2f})', linewidth=1.5)
            plt.plot([0, 1], [0, 1], 'k:', linewidth=1)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Tỷ lệ dương tính giả', fontsize=8)
            plt.ylabel('Tỷ lệ dương tính thật', fontsize=8)
            plt.title(f'Đường cong ROC - {dataset_name} ({self.model_type.upper()})', fontsize=9)
            plt.legend(loc='lower right', fontsize=7)
            plt.tick_params(axis='both', labelsize=7)
            plt.tight_layout(pad=0.5)
            plt.savefig(os.path.join(self.output_dir, f"{self.model_type}_roc_{dataset_name.lower().replace(' ', '_')}.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        if dataset_name == 'Kiểm tra':
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importances = pd.DataFrame({
                    'feature': self.selected_features,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
            elif self.model_type == 'lr':
                self.feature_importances = pd.DataFrame({
                    'feature': self.selected_features,
                    'importance': np.abs(self.model.coef_).mean(axis=0)
                }).sort_values('importance', ascending=False)
            elif self.model_type == 'svm':
                logger.info(f"Tính tầm quan trọng hoán vị cho {self.model_type}")
                perm_importance = permutation_importance(
                    self.model, X, y_true, n_repeats=10, random_state=42, n_jobs=-1, scoring='f1_weighted'
                )
                self.feature_importances = pd.DataFrame({
                    'feature': self.selected_features,
                    'importance': perm_importance.importances_mean
                }).sort_values('importance', ascending=False)
            elif self.model_type == 'ensemble':
                logger.info(f"Tính tầm quan trọng đặc trưng tổng hợp cho {self.model_type}")
                rf_model = self.model.named_estimators_['rf']
                lgbm_model = self.model.named_estimators_['lgbm']
                
                rf_importance = rf_model.feature_importances_ / rf_model.feature_importances_.sum()
                lgbm_importance = lgbm_model.feature_importances_ / lgbm_model.feature_importances_.sum()
                
                weights = {'rf': 0.5, 'lgbm': 0.5}
                combined_importance = (
                    weights['rf'] * rf_importance + weights['lgbm'] * lgbm_importance
                )
                
                self.feature_importances = pd.DataFrame({
                    'feature': self.selected_features,
                    'importance': combined_importance
                }).sort_values('importance', ascending=False)
        
            if self.feature_importances is not None:
                plt.figure(figsize=(4, 3))
                sns.set_context("paper", font_scale=0.8)
                sns.barplot(x='importance', y='feature', data=self.feature_importances.head(15))
                plt.title(f'Top 15 Đặc trưng Quan trọng ({self.model_type.upper()})', fontsize=9)
                plt.xlabel('Tầm quan trọng', fontsize=8)
                plt.ylabel('Đặc trưng', fontsize=8)
                plt.tick_params(axis='both', labelsize=7)
                plt.tight_layout(pad=0.5)
                plt.savefig(os.path.join(self.output_dir, f"{self.model_type}_feature_importance.png"), dpi=300, bbox_inches='tight')
                plt.close()
        
        return metrics, cm
    
    def save_model(self) -> None:
        model_path = os.path.join(self.output_dir, f"{self.model_type}_model.joblib")
        joblib.dump(self.model, model_path)
        logger.info(f"Lưu mô hình vào: {model_path}")
        
        if self.feature_importances is not None:
            fi_path = os.path.join(self.output_dir, f"{self.model_type}_feature_importances.csv")
            self.feature_importances.to_csv(fi_path, index=False)
            logger.info(f"Lưu tầm quan trọng đặc trưng vào: {fi_path}")
        
        if self.selected_features is not None:
            features_path = os.path.join(self.output_dir, f"{self.model_type}_selected_features.csv")
            pd.DataFrame(self.selected_features, columns=['feature']).to_csv(features_path, index=False)
            logger.info(f"Lưu danh sách đặc trưng đã chọn vào: {features_path}")
        
        if self.model_type in ['svm', 'lr']:
            scaler_path = os.path.join(self.output_dir, f"{self.model_type}_scaler.joblib")
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Lưu bộ chuẩn hóa vào: {scaler_path}")

class FloodPredictor:
    """Lớp để dự đoán nguy cơ ngập lụt với các mô hình đã huấn luyện."""
    
    def __init__(
        self,
        model_path: str,
        output_dir: str = 'predictions',
        selected_features_path: Optional[str] = None,
        scaler_path: Optional[str] = None
    ):
        self.model_path = model_path
        self.output_dir = output_dir
        self.selected_features_path = selected_features_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.selected_features = None
        os.makedirs(output_dir, exist_ok=True)
        self.load_model()
    
    def load_model(self) -> None:
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Tải mô hình từ: {self.model_path}")
            if self.scaler_path:
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"Tải bộ chuẩn hóa từ: {self.scaler_path}")
            if self.selected_features_path:
                self.selected_features = pd.read_csv(self.selected_features_path)['feature'].tolist()
                logger.info(f"Tải danh sách đặc trưng đã chọn từ: {self.selected_features_path}")
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình, bộ chuẩn hóa hoặc đặc trưng: {str(e)}")
            raise
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        r = df['r'].values if 'r' in df.columns else None
        c = df['c'].values if 'c' in df.columns else None
        features = df.drop(['r', 'c'], axis=1, errors='ignore')
        
        if self.selected_features is not None:
            features = features[self.selected_features]
        
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features)
            features = pd.DataFrame(features_scaled, columns=features.columns, index=features.index)
        
        logger.info(f"Đang dự đoán với {self.model_path}")
        start_time = time.time()
        predictions = self.model.predict(features)
        logger.info(f"Hoàn thành dự đoán trong {time.time() - start_time:.2f} giây")
        
        return pd.DataFrame({
            'r': r,
            'c': c,
            'prediction': predictions
        })
    
    def save_predictions(self, predictions_df: pd.DataFrame, output_file: str = 'predictions.csv') -> None:
        output_path = os.path.join(self.output_dir, output_file)
        predictions_df.to_csv(output_path, index=False)
        logger.info(f"Lưu dự đoán vào: {output_path}")
    
    def save_as_tif(self, predictions_df: pd.DataFrame, profile: Dict, output_file: str = 'predictions.tif') -> None:
        required_cols = ['r', 'c', 'prediction']
        if not all(col in predictions_df.columns for col in required_cols):
            raise ValueError("DataFrame dự đoán phải chứa các cột 'r', 'c', và 'prediction'.")
        
        height, width = profile['height'], profile['width']
        raster_data = np.full((height, width), profile.get('nodata', -9999), dtype=np.float32)
        
        for _, row in predictions_df.iterrows():
            r, c = int(row['r']), int(row['c'])
            if 0 <= r < height and 0 <= c < width:
                raster_data[r, c] = row['prediction']
        
        profile.update(dtype=rasterio.float32, count=1, compress='lzw')
        output_path = os.path.join(self.output_dir, output_file)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(raster_data, 1)
        logger.info(f"Lưu GeoTIFF vào: {output_path}")

def plot_combined_confusion_matrices(
    cms: Dict[str, np.ndarray],
    model_types: List[str],
    output_dir: str = 'models'
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.27, 5.83))
    axes = axes.ravel()
    sns.set_context("paper", font_scale=0.9)
    for i, model_type in enumerate(model_types[:4]):
        sns.heatmap(
            cms[model_type],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=GROUP_NAMES,
            yticklabels=GROUP_NAMES,
            ax=axes[i],
            annot_kws={"size": 8},
            cbar=False
        )
        axes[i].set_title(f'Ma trận nhầm lẫn - {model_type.upper()}', fontsize=10)
        axes[i].set_xlabel('Dự đoán', fontsize=9)
        axes[i].set_ylabel('Thực tế', fontsize=9)
        axes[i].tick_params(axis='both', labelsize=8)
    
    plt.tight_layout(pad=1.0)
    output_path = os.path.join(output_dir, 'ma_tran_nham_lan_tong_hop.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Lưu ma trận nhầm lẫn tổng hợp vào {output_path}")

def plot_model_comparison(
    metrics_dict: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: str = 'models'
) -> None:
    metrics = ['Độ chính xác', 'Độ chính xác (Precision)', 'Độ nhạy (Recall)', 'F1 Score']
    sns.set_context("paper", font_scale=0.9)
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8.27, 4))
        models = list(metrics_dict.keys())
        values = [metrics_dict[model][metric]['mean'] for model in models]
        errors = [metrics_dict[model][metric]['ci'] for model in models]
        
        ax.bar(models, values, yerr=errors, capsize=5, color=sns.color_palette("husl", len(models)))
        ax.set_ylim(0, 1.1)
        ax.set_title(f'So sánh mô hình - {metric} (95% CI)', fontsize=10)
        ax.set_ylabel(metric, fontsize=9)
        ax.set_xlabel('Mô hình', fontsize=9)
        ax.tick_params(axis='both', labelsize=8)
        
        plt.tight_layout(pad=1.0)
        output_path = os.path.join(output_dir, f"so_sanh_mo_hinh_{metric.lower().replace(' ', '_')}_ci.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Lưu biểu đồ so sánh mô hình cho {metric} vào {output_path}")

def plot_per_class_metrics(
    metrics_dict: Dict[str, Dict[str, np.ndarray]],
    output_dir: str = 'models'
) -> None:
    metrics = ['Độ chính xác (Precision)', 'Độ nhạy (Recall)', 'F1 Score']
    sns.set_context("paper", font_scale=0.9)
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8.27, 4))
        models = list(metrics_dict.keys())
        for i, group_name in enumerate(GROUP_NAMES):
            values = [metrics_dict[model][metric][i] for model in models]
            ax.bar([x + i*0.15 for x in range(len(models))], values, width=0.15, label=group_name)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models)
        ax.set_ylim(0, 1.1)
        ax.set_title(f'So sánh {metric} theo lớp', fontsize=10)
        ax.set_ylabel(metric, fontsize=9)
        ax.set_xlabel('Mô hình', fontsize=9)
        ax.legend(fontsize=7)
        ax.tick_params(axis='both', labelsize=8)
        plt.tight_layout(pad=1.0)
        output_path = os.path.join(output_dir, f"theo_lop_{metric.lower().replace(' ', '_')}_so_sanh.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Lưu biểu đồ so sánh {metric} theo lớp vào {output_path}")

def compute_confidence_interval(scores: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    mean = np.mean(scores)
    std = np.std(scores, ddof=1)
    n = len(scores)
    if n < 2 or std == 0:
        return mean, 0.0
    ci = t.ppf((1 + confidence) / 2, n - 1) * std / np.sqrt(n)
    return mean, ci

def compare_models(
    df: pd.DataFrame,
    profile: Dict,
    tune_params: bool = True,
    n_iter: int = 3,
    n_features: int = 18,
    output_dir: str = 'models'
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, Dict[str, float]]], Dict[str, List[str]]]:
    try:
        os.makedirs(output_dir, exist_ok=True)
        models = ['rf', 'svm', 'lr', 'lgbm', 'ensemble']
        results = {}
        metrics_dict = {}
        per_class_metrics = {}
        selected_features_dict = {}
        cms = {}
        
        for model_type in models:
            logger.info(f"\n{'='*40}\nĐánh giá {model_type.upper()}\n{'='*40}")
            model = FloodPredictionModel(model_type=model_type, output_dir=output_dir, n_features=n_features)
            result_df, y_test, y_pred_test, metrics, cm = model.train_and_evaluate(df, profile, tune_params, n_iter)
            results[model_type] = result_df
            selected_features_dict[model_type] = model.selected_features
            cms[model_type] = cm
            
            test_data = result_df[result_df['type'] == 'test']
            y_true, y_pred = test_data['label'], test_data['pred']
            
            X_selected = df.drop(['r', 'c', 'label'], axis=1)[model.selected_features]
            cv_scores = cross_val_predict(
                model.model,
                X_selected,
                df['label'],
                cv=5,
                n_jobs=-1,
                method='predict'
            )
            acc_scores = [
                accuracy_score(y_true, cv_scores[df.index.isin(test_data.index)])
                for _ in range(5)
            ]
            prec_scores = [
                precision_score(y_true, cv_scores[df.index.isin(test_data.index)], average='weighted')
                for _ in range(5)
            ]
            rec_scores = [
                recall_score(y_true, cv_scores[df.index.isin(test_data.index)], average='weighted')
                for _ in range(5)
            ]
            f1_scores = [
                f1_score(y_true, cv_scores[df.index.isin(test_data.index)], average='weighted')
                for _ in range(5)
            ]
            
            metrics_dict[model_type] = {
                'Độ chính xác': {
                    'mean': accuracy_score(y_true, y_pred),
                    'ci': compute_confidence_interval(acc_scores)[1]
                },
                'Độ chính xác (Precision)': {
                    'mean': precision_score(y_true, y_pred, average='weighted'),
                    'ci': compute_confidence_interval(prec_scores)[1]
                },
                'Độ nhạy (Recall)': {
                    'mean': recall_score(y_true, y_pred, average='weighted'),
                    'ci': compute_confidence_interval(rec_scores)[1]
                },
                'F1 Score': {
                    'mean': f1_score(y_true, y_pred, average='weighted'),
                    'ci': compute_confidence_interval(f1_scores)[1]
                }
            }
            
            per_class_metrics[model_type] = {
                'Độ chính xác (Precision)': precision_score(y_true, y_pred, average=None),
                'Độ nhạy (Recall)': recall_score(y_true, y_pred, average=None),
                'F1 Score': f1_score(y_true, y_pred, average=None)
            }
        
        plot_combined_confusion_matrices(cms, models, output_dir)
        plot_model_comparison(metrics_dict, output_dir)
        plot_per_class_metrics(per_class_metrics, output_dir)
        
        metrics_df = pd.DataFrame({
            model: {
                metric: f"{m['mean']:.4f} ± {m['ci']:.4f}"
                for metric, m in metrics_dict[model].items()
            }
            for model in models
        }).T
        logger.info("\nTóm tắt hiệu suất mô hình (Tập kiểm tra, 95%% CI):\n%s", metrics_df)
        metrics_df.to_csv(os.path.join(output_dir, 'so_sanh_mo_hinh_metrics_ci.csv'), index=True)
        logger.info(f"Lưu tóm tắt chỉ số vào {os.path.join(output_dir, 'so_sanh_mo_hinh_metrics_ci.csv')}")
        
        return results, metrics_dict, selected_features_dict
    
    except Exception as e:
        logger.error(f"Lỗi trong so_sanh_mo_hinh: {str(e)}")
        raise

def predict_full_raster(
    events_dir: str,
    model_dir: str = 'models',
    batch_size: int = 100000
) -> None:
    try:
        output_dir = os.path.join(model_dir, 'predictions')
        os.makedirs(output_dir, exist_ok=True)
        models = ['rf', 'svm', 'lr', 'lgbm', 'ensemble']
        
        data, profile = load_base_data(events_dir=events_dir)
        height, width = next(iter(data.values())).shape
        
        coords = [(r, c) for r in range(height) for c in range(width)]
        df_full = pd.DataFrame(coords, columns=['r', 'c'])
        
        for param in PARAMS:
            df_full[param] = [data[param][r, c] for r, c in coords]
        
        for model_type in models:
            try:
                logger.info(f"Dự đoán với {model_type.upper()} sử dụng lượng mưa từ {events_dir}")
                model_path = os.path.join(model_dir, f"{model_type}_model.joblib")
                scaler_path = os.path.join(model_dir, f"{model_type}_scaler.joblib") if model_type in ['svm', 'lr'] else None
                features_path = os.path.join(model_dir, f"{model_type}_selected_features.csv")
                
                predictor = FloodPredictor(
                    model_path=model_path,
                    output_dir=output_dir,
                    selected_features_path=features_path,
                    scaler_path=scaler_path
                )
                
                predictions_df = pd.DataFrame()
                for i in range(0, len(df_full), batch_size):
                    batch_df = df_full.iloc[i:i+batch_size]
                    batch_predictions = predictor.predict(batch_df)
                    predictions_df = pd.concat([predictions_df, batch_predictions], axis=0)
                    logger.info(f"Xử lý lô {i//batch_size + 1} cho {model_type}")
                
                output_file = f"{model_type}_du_bao_{events_dir.replace('/', '_')}.tif"
                predictor.save_as_tif(predictions_df, profile, output_file)
                predictor.save_predictions(predictions_df, f"{model_type}_du_bao_{events_dir.replace('/', '_')}.csv")
            except Exception as e:
                logger.error(f"Lỗi khi dự đoán với {model_type}: {str(e)}")
                continue
        
        logger.info("Hoàn thành dự đoán toàn bộ raster cho tất cả mô hình")
    
    except Exception as e:
        logger.error(f"Lỗi trong predict_full_raster: {str(e)}")
        raise

def main(events_dir: str = 'Events/2023', has_build_models: bool = True):
    output_dir = 'ML_Model_final'
    try:
        if has_build_models:
            logger.info("Bắt đầu huấn luyện mô hình mới")
            data, profile = load_base_data(events_dir=events_dir)
            df, df2 = create_dataframes(data)
            results, metrics, selected_features_dict = compare_models(
                df2,
                profile,
                tune_params=True,
                n_iter=3,
                n_features=18,
                output_dir=output_dir
            )
            logger.info("Hoàn thành huấn luyện mô hình")
        
        logger.info(f"Bắt đầu dự đoán với dữ liệu lượng mưa từ {events_dir}")
        predict_full_raster(events_dir=events_dir, model_dir=output_dir)
        logger.info("Quy trình dự đoán ngập lụt hoàn tất thành công")
    
    except Exception as e:
        logger.error(f"Quy trình thất bại: {str(e)}")
        raise

if __name__ == "__main__":
    main()