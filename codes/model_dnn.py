import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import rasterio
import os
import logging
import joblib
import time
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('flood_prediction_dnn.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Data parameters
BASENAMES = [
    'eleStream', 'disStream', 'wSlope', 'streamSlope', 'flowLength', 'area',
    'twi', 'spi', 'tpi', 'wNdvi', 'wCN', 'wInfiRate', 'dem', 'wDem',
    'profCurvature', 'planCurvature'
]
PRECIPS = ['raster_max', 'raster_3h_max', 'raster_6h_max', 'raster_24h_max']
BASE_RASTER_FOLDER = 'rasters'
PARAMS = BASENAMES + PRECIPS
GROUP_NAMES = ['Không có lũ', 'Rất nhỏ', 'Nhỏ', 'Trung bình', 'Lớn']
NEIGHBORHOOD_SIZE = 7
BATCH_SIZE = 512
N_FEATURES = 18

# Scaling functions
def robust_scaling(X: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def log_transformation(X: np.ndarray) -> np.ndarray:
    if np.any(X < 0):
        raise ValueError("Dữ liệu chứa giá trị âm, không thể áp dụng biến đổi log.")
    return np.log1p(X)

def minmax_scaling(X: np.ndarray) -> np.ndarray:
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)

def zscore_scaling(X: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def get_scaling_func(name: str) -> list[callable]:
    if name in ['eleStream', 'wInfiRate', 'dem', 'wDem']:
        return [robust_scaling, minmax_scaling]
    elif name in ['disStream', 'flowLength', 'area', 'spi'] + PRECIPS:
        return [log_transformation, minmax_scaling]
    elif name in ['tpi', 'profCurvature', 'planCurvature']:
        return [zscore_scaling, minmax_scaling]
    return [minmax_scaling]

# Load data
def load_base_data(precip_folder: str) -> tuple[dict[str, np.ndarray], dict]:
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
            file_path = os.path.join(precip_folder, f'{precip}.tif')
            with rasterio.open(file_path) as src:
                d = src.read(1)
                for func in get_scaling_func(precip):
                    d = func(d)
                data[precip] = d

        with rasterio.open('hazard_points2.tif', 'r') as src:
            profile = src.profile

        logger.info(f"Tải và tiền xử lý dữ liệu raster thành công từ {precip_folder}")
        return data, profile

    except Exception as e:
        logger.error(f"Lỗi khi tải dữ liệu cơ bản từ {precip_folder}: {str(e)}")
        raise

# Create DataFrame
def create_dataframes(data: dict[str, np.ndarray]) -> pd.DataFrame:
    try:
        with rasterio.open('hazard_points2.tif', 'r') as src:
            hazard = src.read(1)

        indices = np.where(hazard >= 0)
        df = pd.DataFrame({
            'r': indices[0],
            'c': indices[1]
        })
        for param in PARAMS:
            df[param] = data[param][indices]
        df['label'] = hazard[indices]
        logger.info(f"Tạo DataFrame với nhãn gốc ({', '.join(GROUP_NAMES)})")
        return df

    except Exception as e:
        logger.error(f"Lỗi khi tạo DataFrame: {str(e)}")
        raise

# Extract neighborhood features
def extract_neighborhood_features(data: dict[str, np.ndarray], df: pd.DataFrame, window_size: int = NEIGHBORHOOD_SIZE) -> pd.DataFrame:
    half_window = window_size // 2
    height, width = next(iter(data.values())).shape
    new_features = {}
    
    for param in PARAMS:
        values = []
        for idx, row in df.iterrows():
            r, c = int(row['r']), int(row['c'])
            if (r - half_window >= 0 and r + half_window < height and
                c - half_window >= 0 and c + half_window < width):
                window = data[param][r-half_window:r+half_window+1, c-half_window:c+half_window+1]
                values.append([
                    np.mean(window), np.min(window), np.max(window), np.std(window)
                ])
            else:
                values.append([np.nan, np.nan, np.nan, np.nan])
        new_features.update({
            f'{param}_mean': [v[0] for v in values],
            f'{param}_min': [v[1] for v in values],
            f'{param}_max': [v[2] for v in values],
            f'{param}_std': [v[3] for v in values]
        })
    
    df_new = df.copy()
    for key, values in new_features.items():
        df_new[key] = values
    df_new = df_new.dropna()
    return df_new

# Create dataset
def create_dataset(X, y, batch_size):
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)  # y is already one-hot encoded
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.cache().shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Focal loss
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(loss, axis=-1)
    return focal_loss_fixed

# DNN Model
class DeepLearningFloodModel:
    def __init__(self, output_dir: str, n_features: int = N_FEATURES):
        self.model_type = 'dnn'
        self.output_dir = output_dir
        self.n_features = n_features
        self.model = None
        self.feature_selector = None
        self.selected_features = None
        self.scaler = StandardScaler()
        os.makedirs(output_dir, exist_ok=True)
    
    def build_dnn(self, input_dim: int) -> Sequential:
        model = Sequential([
            Input(shape=(input_dim,)),
            Dense(256, kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            Dropout(0.3),
            Dense(128, kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            Dropout(0.3),
            Dense(64, kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            Dropout(0.2),
            Dense(5, activation='softmax', dtype='float32')
        ])
        return model
    
    def load_model(self) -> None:
        try:
            model_path = os.path.join(self.output_dir, f"{self.model_type}_model.h5")
            self.model = tf.keras.models.load_model(model_path, custom_objects={'focal_loss_fixed': focal_loss()})
            logger.info(f"Mô hình {self.model_type} được tải từ: {model_path}")
            self.selected_features = joblib.load(os.path.join(self.output_dir, f"{self.model_type}_features.joblib"))
            self.scaler = joblib.load(os.path.join(self.output_dir, f"{self.model_type}_scaler.joblib"))
            logger.info(f"Đã tải đặc trưng và scaler cho {self.model_type}")
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình {self.model_type}: {str(e)}")
            raise
    
    def prepare_data(self, df: pd.DataFrame, data: dict, test_size: float = 0.2):
        df = extract_neighborhood_features(data, df, window_size=NEIGHBORHOOD_SIZE)
        X = df.drop(['r', 'c', 'label'], axis=1)
        y = df['label'].values
        
        if X.empty or len(y) == 0:
            raise ValueError("Dữ liệu X hoặc y rỗng sau khi trích xuất đặc trưng.")
        
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
        X = X.drop(to_drop, axis=1)
        original_columns = X.columns.tolist()
        
        if self.n_features and len(original_columns) > self.n_features:
            self.feature_selector = SelectKBest(f_classif, k=self.n_features)
            X_transformed = self.feature_selector.fit_transform(X, y)
            self.selected_features = [original_columns[i] for i in np.where(self.feature_selector.get_support())[0]]
            X = pd.DataFrame(X_transformed, columns=self.selected_features, index=X.index)
        else:
            self.selected_features = original_columns
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=5)
        y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=5)
        
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights_dict = {i: w for i, w in enumerate(class_weights)}
        
        return X_train, X_test, y_train, y_test, y_train_cat, y_test_cat, self.selected_features, class_weights_dict
    
    def train_and_evaluate(self, df: pd.DataFrame, data: dict, batch_size: int = BATCH_SIZE, epochs: int = 100):
        X_train, X_test, y_train, y_test, y_train_cat, y_test_cat, selected_features, class_weights = self.prepare_data(df, data)
        
        self.model = self.build_dnn(X_train.shape[1])
        
        self.model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss=focal_loss(gamma=2.0, alpha=0.25),
            metrics=['accuracy']
        )
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        logger.info(f"Bắt đầu huấn luyện mô hình {self.model_type}")
        start_time = time.time()
        
        train_dataset = create_dataset(X_train, y_train_cat, batch_size)
        val_dataset = create_dataset(X_test, y_test_cat, batch_size)
        
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        training_time = time.time() - start_time
        logger.info(f"Hoàn thành huấn luyện trong {training_time:.2f} giây")
        
        self._plot_training_history(history)
        
        y_pred_test = self.model.predict(X_test)
        y_pred_test = np.argmax(y_pred_test, axis=1)
        y_test_labels = y_test  # Use 1D labels for evaluation
        
        test_indices = df.index[-len(y_test):]
        result_df = df.loc[test_indices].copy()
        result_df['pred'] = y_pred_test
        result_df['type'] = 'test'
        
        metrics, cm = self._evaluate_model(y_test_labels, y_pred_test, 'Test', X_test)
        
        self.save_model()
        
        return result_df, y_test_labels, metrics, cm
    
    def _plot_training_history(self, history):
        plt.figure(figsize=(6, 3))
        plt.plot(history.history['loss'], label='Tổn thất huấn luyện')
        plt.plot(history.history['val_loss'], label='Tổn thất kiểm tra')
        plt.title(f'Tổn thất huấn luyện {self.model_type.upper()}')
        plt.xlabel('Epoch')
        plt.ylabel('Tổn thất')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f"{self.model_type}_loss_curve.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str, X: np.ndarray) -> tuple[dict, np.ndarray]:
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='weighted'),
            'Recall': recall_score(y_true, y_pred, average='weighted'),
            'F1 Score': f1_score(y_true, y_pred, average='weighted')
        }
        
        logger.info(f"\nHiệu suất trên tập {dataset_name} ({self.model_type.upper()}):")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        logger.info("\nBáo cáo phân loại:")
        logger.info("\n%s", classification_report(y_true, y_pred, target_names=GROUP_NAMES))
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(4, 3))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=GROUP_NAMES,
            yticklabels=GROUP_NAMES,
            cbar=False
        )
        plt.title(f'Ma trận nhầm lẫn - {dataset_name} ({self.model_type.upper()})')
        plt.xlabel('Dự đoán')
        plt.ylabel('Thực tế')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{self.model_type}_cm_{dataset_name.lower()}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        y_score = self.model.predict(X)
        self._plot_roc_curve(y_true, y_score, dataset_name)
        
        return metrics, cm
    
    def _plot_roc_curve(self, y_true, y_score, dataset_name):
        fpr, tpr, roc_auc = {}, {}, {}
        for i, group_name in enumerate(GROUP_NAMES):
            y_true_bin = (y_true == i).astype(int)
            fpr[i], tpr[i], _ = roc_curve(y_true_bin, y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(5)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(5):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= 5
        roc_auc['macro'] = auc(all_fpr, mean_tpr)
        
        plt.figure(figsize=(5, 4))
        colors = sns.color_palette("husl", 5)
        for i, (group_name, color) in enumerate(zip(GROUP_NAMES, colors)):
            plt.plot(fpr[i], tpr[i], color=color, label=f'{group_name} (AUC = {roc_auc[i]:.2f})')
        plt.plot(all_fpr, mean_tpr, 'k--', label=f'Trung bình (AUC = {roc_auc["macro"]:.2f})')
        plt.plot([0, 1], [0, 1], 'k:', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tỷ lệ dương tính giả')
        plt.ylabel('Tỷ lệ dương tính thật')
        plt.title(f'Đường cong ROC - {dataset_name} ({self.model_type.upper()})')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{self.model_type}_roc_{dataset_name.lower()}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_model(self) -> None:
        model_path = os.path.join(self.output_dir, f"{self.model_type}_model.h5")
        self.model.save(model_path)
        logger.info(f"Mô hình được lưu tại: {model_path}")
        if self.selected_features is not None:
            joblib.dump(self.selected_features, os.path.join(self.output_dir, f"{self.model_type}_features.joblib"))
            joblib.dump(self.scaler, os.path.join(self.output_dir, f"{self.model_type}_scaler.joblib"))
            logger.info(f"Đã lưu danh sách đặc trưng và scaler cho {self.model_type}")

# Prediction class
class FloodPredictorDL:
    def __init__(self, model_path: str, output_dir: str, selected_features=None, scaler_path=None):
        self.model_path = model_path
        self.output_dir = output_dir
        self.selected_features = selected_features
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        os.makedirs(output_dir, exist_ok=True)
        self.load_model()
    
    def load_model(self) -> None:
        try:
            self.model = tf.keras.models.load_model(self.model_path, custom_objects={'focal_loss_fixed': focal_loss()})
            logger.info(f"Mô hình được tải từ: {self.model_path}")
            if self.scaler_path:
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"Scaler được tải từ: {self.scaler_path}")
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình hoặc scaler: {str(e)}")
            raise
    
    def predict(self, data: dict[str, np.ndarray], df: pd.DataFrame) -> pd.DataFrame:
        df = extract_neighborhood_features(data, df, window_size=NEIGHBORHOOD_SIZE)
        X = df.drop(['r', 'c'], axis=1, errors='ignore')
        if self.selected_features is not None:
            X = X[self.selected_features]
        if self.scaler is not None:
            X = self.scaler.transform(X)
        predictions = self.model.predict(X)
        predictions = np.argmax(predictions, axis=1)
        return pd.DataFrame({
            'r': df['r'].values,
            'c': df['c'].values,
            'prediction': predictions
        })
    
    def save_predictions(self, predictions_df: pd.DataFrame, output_file: str = 'predictions.csv') -> None:
        output_path = os.path.join(self.output_dir, output_file)
        predictions_df.to_csv(output_path, index=False)
        logger.info(f"Dự đoán được lưu tại: {output_path}")
    
    def save_as_tif(self, predictions_df: pd.DataFrame, profile: dict, output_file: str = 'predictions.tif') -> None:
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
        logger.info(f"GeoTIFF được lưu tại: {output_path}")

# Training function
def train_model(precip_folder: str = 'Events/2023', model_dir: str = 'models_dnn', batch_size: int = BATCH_SIZE, epochs: int = 100):
    logger.info(f"Running training with precip_folder: {precip_folder}, model_dir: {model_dir}")
    try:
        data, profile = load_base_data(precip_folder=precip_folder)
        df = create_dataframes(data)
        model = DeepLearningFloodModel(output_dir=model_dir, n_features=N_FEATURES)
        result_df, y_test_labels, metrics, cm = model.train_and_evaluate(df, data, batch_size, epochs)
        
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [metrics['Accuracy'], metrics['Precision'], metrics['Recall'], metrics['F1 Score']]
        })
        metrics_df.to_csv(os.path.join(model_dir, 'dnn_metrics.csv'), index=False)
        logger.info(f"Đã lưu số liệu hiệu suất tại {os.path.join(model_dir, 'dnn_metrics.csv')}")
        
        return model.selected_features, profile
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

# Prediction function
def predict_data(model_dir: str = 'models_dnn', precip_folder: str = 'Events/2023', batch_size: int = 100000):
    logger.info(f"Running prediction with model_dir: {model_dir}, precip_folder: {precip_folder}")
    try:
        data, profile = load_base_data(precip_folder=precip_folder)
        height, width = next(iter(data.values())).shape
        coords = [(r, c) for r in range(height) for c in range(width)]
        df_full = pd.DataFrame(coords, columns=['r', 'c'])
        
        for param in PARAMS:
            if param in data:
                df_full[param] = [data[param][r, c] for r, c in coords]
        
        model_path = os.path.join(model_dir, 'dnn_model.h5')
        scaler_path = os.path.join(model_dir, 'dnn_scaler.joblib')
        selected_features_path = os.path.join(model_dir, 'dnn_features.joblib')
        
        selected_features = joblib.load(selected_features_path) if os.path.exists(selected_features_path) else None
        
        predictor = FloodPredictorDL(
            model_path=model_path,
            output_dir=os.path.join(model_dir, 'predictions', precip_folder),
            selected_features=selected_features,
            scaler_path=scaler_path
        )
        
        predictions_df = pd.DataFrame()
        for i in range(0, len(df_full), batch_size):
            batch_df = df_full.iloc[i:i+batch_size]
            batch_predictions = predictor.predict(data, batch_df)
            predictions_df = pd.concat([predictions_df, batch_predictions], axis=0)
            logger.info(f"Đã xử lý lô {i//batch_size + 1}")
            gc.collect()
        
        predictor.save_as_tif(predictions_df, profile, 'dnn_predictions.tif')
        predictor.save_predictions(predictions_df, 'dnn_predictions.csv')
        
        logger.info("Prediction completed successfully")
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise

# Clear GPU memory
def clear_gpu_memory():
    try:
        tf.keras.backend.clear_session()
        gc.collect()
        logger.info("Cleared GPU memory")
    except Exception as e:
        logger.error(f"Error clearing memory: {str(e)}")

if __name__ == "__main__":
    # Train the model
    train_model(precip_folder='Events/2023', model_dir='models_dnn')
    
    # Predict with different precipitation folders
    predict_data(model_dir='models_dnn', precip_folder='Events/2023')