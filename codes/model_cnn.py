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
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization,
    Input, LeakyReLU, Add, GlobalAveragePooling2D, SpatialDropout2D
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.experimental import CosineDecayRestarts
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import rasterio
import os
import logging
import joblib
from scipy.stats import t
import time
import gc
from scipy import ndimage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('flood_prediction_cnn.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def configure_gpu_memory():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logger.info("Enabled GPU memory growth")
        else:
            logger.info("No GPU found, running on CPU")
    except Exception as e:
        logger.error(f"Error configuring GPU: {str(e)}")

os.environ["PATH"] += os.pathsep + r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin"
configure_gpu_memory()

set_global_policy('mixed_float16')

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
PATCH_SIZE = 5
BATCH_SIZE = 512

# Data augmentation functions
@tf.function
def rotate_image(image: tf.Tensor, angle: tf.Tensor) -> tf.Tensor:
    logger.debug(f"Rotating image with angle: {angle}")
    if tf.equal(angle, 90):
        return tf.image.rot90(image, k=1)
    elif tf.equal(angle, 180):
        return tf.image.rot90(image, k=2)
    elif tf.equal(angle, 270):
        return tf.image.rot90(image, k=3)
    else:
        return image

@tf.function
def flip_image(image: tf.Tensor, flip_horizontal: tf.Tensor, flip_vertical: tf.Tensor) -> tf.Tensor:
    if flip_horizontal:
        image = tf.image.flip_left_right(image)
    if flip_vertical:
        image = tf.image.flip_up_down(image)
    return image

@tf.function
def adjust_brightness(image: tf.Tensor, brightness_factor: tf.Tensor) -> tf.Tensor:
    image = image * brightness_factor
    return tf.clip_by_value(image, 0.0, 1.0)

@tf.function
def add_gaussian_noise(image: tf.Tensor, mean: tf.Tensor, std: tf.Tensor) -> tf.Tensor:
    noise = tf.random.normal(tf.shape(image), mean=mean, stddev=std)
    return tf.clip_by_value(image + noise, 0.0, 1.0)

@tf.function
def adjust_contrast(image: tf.Tensor, contrast_factor: tf.Tensor) -> tf.Tensor:
    mean = tf.reduce_mean(image, axis=[0, 1], keepdims=True)
    image = mean + contrast_factor * (image - mean)
    return tf.clip_by_value(image, 0.0, 1.0)

@tf.function
def custom_augmentation(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    
    if tf.random.uniform(()) < 0.5:
        angle = tf.random.uniform((), minval=0, maxval=3, dtype=tf.int32) * 90
        image = rotate_image(image, tf.cast(angle, tf.float32))
    
    if tf.random.uniform(()) < 0.5:
        image = flip_image(image, tf.constant(True), tf.constant(False))
    if tf.random.uniform(()) < 0.5:
        image = flip_image(image, tf.constant(False), tf.constant(True))
    
    if tf.random.uniform(()) < 0.4:
        brightness_factor = tf.random.uniform((), minval=0.7, maxval=1.3)
        image = adjust_brightness(image, brightness_factor)
    
    if tf.random.uniform(()) < 0.4:
        image = add_gaussian_noise(image, mean=0.0, std=0.05)
    
    if tf.random.uniform(()) < 0.4:
        contrast_factor = tf.random.uniform((), minval=0.7, maxval=1.3)
        image = adjust_contrast(image, contrast_factor)
    
    return image, label

# Focal loss function
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(loss, axis=-1)
    return focal_loss_fixed

# Data normalization functions
def robust_scaling(X: np.ndarray) -> np.ndarray:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def log_transformation(X: np.ndarray) -> np.ndarray:
    if np.any(X < 0):
        raise ValueError("Data contains negative values, cannot apply log transformation.")
    return np.log1p(X)

def minmax_scaling(X: np.ndarray) -> np.ndarray:
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)

def zscore_scaling(X: np.ndarray) -> np.ndarray:
    from sklearn.preprocessing import StandardScaler
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
                d = np.nan_to_num(d, nan=np.nanmean(d), posinf=np.nanmax(d), neginf=np.nanmin(d))
                for func in get_scaling_func(name):
                    d = func(d)
                data[name] = d

        for precip in PRECIPS:
            file_path = os.path.join(precip_folder, f'{precip}.tif')
            if not os.path.exists(file_path):
                logger.warning(f"File {file_path} not found, skipping {precip}")
                continue
            with rasterio.open(file_path) as src:
                d = src.read(1)
                d = np.nan_to_num(d, nan=np.nanmean(d), posinf=np.nanmax(d), neginf=np.nanmin(d))
                for func in get_scaling_func(precip):
                    d = func(d)
                data[precip] = d

        with rasterio.open('hazard_points2.tif', 'r') as src:
            profile = src.profile

        logger.info("Successfully loaded and preprocessed raster data")
        return data, profile

    except Exception as e:
        logger.error(f"Error loading base data: {str(e)}")
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
            if param in data:
                df[param] = data[param][indices]
        df['label'] = hazard[indices]

        logger.info(f"Created DataFrame with labels ({', '.join(GROUP_NAMES)})")
        return df

    except Exception as e:
        logger.error(f"Error creating DataFrame: {str(e)}")
        raise

# Extract patches
def extract_patches(data: dict[str, np.ndarray], df: pd.DataFrame, patch_size: int) -> tuple:
    height, width = next(iter(data.values())).shape
    X, y, valid_indices = [], [], []
    half_patch = patch_size // 2
    
    padded_data = {}
    for param in PARAMS:
        if param in data:
            padded_data[param] = np.pad(
                data[param], 
                ((half_patch, half_patch), (half_patch, half_patch)), 
                mode='reflect'
            )
    
    for local_idx, (idx, row) in enumerate(df.iterrows()):
        r, c = int(row['r']), int(row['c'])
        if 0 <= r < height and 0 <= c < width:
            patch = np.stack([
                padded_data[param][r:r+patch_size, c:c+patch_size]
                for param in PARAMS if param in padded_data
            ], axis=-1)
            if np.isnan(patch).mean() < 0.2:
                X.append(patch)
                valid_indices.append(local_idx)
                if 'label' in df.columns:
                    y.append(row['label'])
    
    if not X:
        raise ValueError("No valid patches extracted.")
    
    X = np.array(X)
    y = np.array(y) if y else None
    
    if np.isnan(X).any():
        for i in range(X.shape[-1]):
            channel = X[..., i]
            mask = np.isnan(channel)
            if mask.any():
                channel[mask] = np.nanmean(channel)
    
    return X, y, valid_indices

# tf.data pipeline
def create_dataset(X, y, batch_size, augment=True):
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    
    if len(y.shape) != 2 or y.shape[1] != 5:
        raise ValueError(f"Expected y to have shape [batch_size, 5], got {y.shape}")
    
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if augment:
        dataset = dataset.map(
            custom_augmentation,
            num_parallel_calls=tf.data.AUTOTUNE
        )
    dataset = dataset.cache().shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Deep Learning Model
class DeepLearningFloodModel:
    def __init__(self, output_dir: str = 'models_cnn'):
        self.model_type = 'cnn'
        self.output_dir = output_dir
        self.model = None
        self.patch_stats = {}
        os.makedirs(output_dir, exist_ok=True)
    
    def residual_block(self, x, filters, kernel_size=3):
        shortcut = x
        y = Conv2D(filters // 2, 1, padding='same', kernel_regularizer=l2(0.005))(x)
        y = BatchNormalization()(y)
        y = LeakyReLU(alpha=0.1)(y)
        
        y = Conv2D(filters // 2, kernel_size, padding='same', kernel_regularizer=l2(0.005))(y)
        y = BatchNormalization()(y)
        y = LeakyReLU(alpha=0.1)(y)
        
        y = Conv2D(filters, 1, padding='same', kernel_regularizer=l2(0.005))(y)
        y = BatchNormalization()(y)
        
        if shortcut.shape[-1] != filters:
            shortcut = Conv2D(filters, 1, padding='same', kernel_regularizer=l2(0.005))(shortcut)
            shortcut = BatchNormalization()(shortcut)
        
        output = Add()([shortcut, y])
        output = LeakyReLU(alpha=0.1)(output)
        return output
    
    def attention_block(self, x):
        channels = x.shape[-1]
        avg_pool = GlobalAveragePooling2D(keepdims=True)(x)
        avg_pool = Conv2D(channels // 2, (1, 1), padding='same')(avg_pool)
        avg_pool = LeakyReLU(alpha=0.1)(avg_pool)
        avg_pool = Conv2D(channels, (1, 1), padding='same')(avg_pool)
        
        max_pool = tf.reduce_max(x, axis=[1, 2], keepdims=True)
        max_pool = Conv2D(channels // 2, (1, 1), padding='same')(max_pool)
        max_pool = LeakyReLU(alpha=0.1)(max_pool)
        max_pool = Conv2D(channels, (1, 1), padding='same')(max_pool)
        
        attention = Add()([avg_pool, max_pool])
        attention = tf.keras.activations.sigmoid(attention)
        return x * attention
    
    def build_cnn(self, input_shape: tuple[int, int, int]) -> Model:
        inputs = Input(shape=input_shape)
        x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.005))(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = SpatialDropout2D(0.3)(x)
        
        x = self.residual_block(x, 64, kernel_size=3)
        x = self.residual_block(x, 64, kernel_size=3)
        x = MaxPooling2D((2, 2))(x)
        x = SpatialDropout2D(0.3)(x)
        
        x = self.residual_block(x, 128, kernel_size=3)
        x = self.residual_block(x, 128, kernel_size=3)
        x = MaxPooling2D((2, 2))(x)
        x = SpatialDropout2D(0.3)(x)
        
        x = self.residual_block(x, 256, kernel_size=3)
        x = self.attention_block(x)
        x = GlobalAveragePooling2D()(x)
        
        x = Dense(128, kernel_regularizer=l2(0.005))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.5)(x)
        
        x = Dense(64, kernel_regularizer=l2(0.005))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.4)(x)
        
        outputs = Dense(5, activation='softmax', dtype='float32', kernel_regularizer=l2(0.005))(x)
        model = Model(inputs, outputs)
        return model
    
    def load_model(self) -> None:
        try:
            model_path = os.path.join(self.output_dir, "cnn_model.h5")
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path, custom_objects={'focal_loss_fixed': focal_loss()})
                logger.info(f"Model loaded from: {model_path}")
                
                patch_stats_path = os.path.join(self.output_dir, "cnn_patch_stats.joblib")
                if os.path.exists(patch_stats_path):
                    self.patch_stats = joblib.load(patch_stats_path)
                    logger.info(f"Patch stats loaded from: {patch_stats_path}")
            else:
                logger.info(f"No model found at {model_path}. Need to train model.")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _standardize_patches(self, patches):
        standardized = np.zeros_like(patches, dtype=np.float32)
        for i in range(patches.shape[-1]):
            channel_data = patches[..., i]
            channel_mean = np.nanmean(channel_data)
            channel_std = np.nanstd(channel_data)
            if channel_std == 0:
                channel_std = 1.0
            standardized[..., i] = (channel_data - channel_mean) / channel_std
            self.patch_stats[str(i)] = {'mean': float(channel_mean), 'std': float(channel_std)}
        return standardized
    
    def prepare_data(self, df: pd.DataFrame, data: dict, test_size: float = 0.2):
        X, y, valid_indices = extract_patches(data, df, PATCH_SIZE)
        if X.size == 0 or y.size == 0:
            raise ValueError("X or y data empty after patch extraction.")
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        
        if np.isnan(X).any() or np.isinf(X).any():
            logger.warning("X data contains NaN or Inf values. Replacing with 0.")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        X = self._standardize_patches(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=5)
        y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=5)
        
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights_dict = {i: w for i, w in enumerate(class_weights)}
        
        if self.patch_stats:
            joblib.dump(self.patch_stats, os.path.join(self.output_dir, "cnn_patch_stats.joblib"))
            logger.info(f"Saved patch stats at: {os.path.join(self.output_dir, 'cnn_patch_stats.joblib')}")
        
        return X_train, X_test, y_train_cat, y_test_cat, class_weights_dict, valid_indices
    
    def train(self, df: pd.DataFrame, data: dict, batch_size: int = BATCH_SIZE, epochs: int = 100):
        X_train, X_test, y_train, y_test, class_weights, valid_indices = self.prepare_data(df, data)
        
        self.model = self.build_cnn((PATCH_SIZE, PATCH_SIZE, len(PARAMS)))
        
        lr_schedule = CosineDecayRestarts(
            initial_learning_rate=2e-4,
            first_decay_steps=1000,
            t_mul=2.0,
            m_mul=0.7,
            alpha=1e-6
        )
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=focal_loss(gamma=2.0, alpha=0.25),
            metrics=['accuracy']
        )
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ]
        
        logger.info("Starting training for CNN model")
        start_time = time.time()
        
        train_dataset = create_dataset(X_train, y_train, batch_size, augment=True)
        val_dataset = create_dataset(X_test, y_test, batch_size, augment=False)
        
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        training_time = time.time() - start_time
        logger.info(f"Completed training in {training_time:.2f} seconds")
        
        self._plot_training_history(history)
        
        y_pred_test = self.model.predict(X_test)
        y_test_labels = np.argmax(y_test, axis=1)
        y_pred_test = np.argmax(y_pred_test, axis=1)
        
        test_indices = valid_indices[-len(y_test):]
        result_df = df.iloc[test_indices].copy()
        result_df['pred'] = y_pred_test
        result_df['type'] = 'test'
        
        metrics, cm = self._evaluate_model(y_test_labels, y_pred_test, 'Test', X_test)
        
        self.save_model()
        
        return result_df, metrics, cm
    
    def _plot_training_history(self, history):
        plt.figure(figsize=(6, 3))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('CNN Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "cnn_loss_curve.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str, X: np.ndarray) -> tuple[dict, np.ndarray]:
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='weighted'),
            'Recall': recall_score(y_true, y_pred, average='weighted'),
            'F1 Score': f1_score(y_true, y_pred, average='weighted')
        }
        
        logger.info(f"\nPerformance on {dataset_name} set (CNN):")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        logger.info("\nClassification Report:")
        logger.info("\n%s", classification_report(y_true, y_pred, target_names=GROUP_NAMES))
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(4, 3))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=GROUP_NAMES,
            yticklabels=GROUP_NAMES,
            cbar=False
        )
        plt.title(f'Confusion Matrix - {dataset_name} (CNN)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "cnn_cm_test.png"), dpi=300, bbox_inches='tight')
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
        plt.plot(all_fpr, mean_tpr, 'k--', label=f'Macro-average (AUC = {roc_auc["macro"]:.2f})')
        plt.plot([0, 1], [0, 1], 'k:', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {dataset_name} (CNN)')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "cnn_roc_test.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_model(self) -> None:
        model_path = os.path.join(self.output_dir, "cnn_model.h5")
        self.model.save(model_path)
        logger.info(f"Model saved at: {model_path}")
        
        if self.patch_stats:
            joblib.dump(self.patch_stats, os.path.join(self.output_dir, "cnn_patch_stats.joblib"))
            logger.info(f"Saved patch stats at: {os.path.join(self.output_dir, 'cnn_patch_stats.joblib')}")

# Prediction class
class FloodPredictorDL:
    def __init__(self, model_path: str, output_dir: str = 'predictions'):
        self.model_path = model_path
        self.output_dir = output_dir
        self.model_type = 'cnn'
        self.model = None
        self.patch_stats = {}
        os.makedirs(output_dir, exist_ok=True)
        self.load_model()

    def load_model(self) -> None:
        try:
            if self.model_path and os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path, custom_objects={'focal_loss_fixed': focal_loss()})
                logger.info(f"Model loaded from: {self.model_path}")
            
            patch_stats_path = os.path.join(os.path.dirname(self.model_path), "cnn_patch_stats.joblib")
            if os.path.exists(patch_stats_path):
                self.patch_stats = joblib.load(patch_stats_path)
                logger.info(f"Patch stats loaded from: {patch_stats_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def standardize_patches(self, patches):
        if not self.patch_stats:
            logger.warning("No patch standardization stats found, using local standardization")
            standardized = np.zeros_like(patches, dtype=np.float32)
            for i in range(patches.shape[-1]):
                channel_data = patches[..., i]
                channel_mean = np.nanmean(channel_data)
                channel_std = np.nanstd(channel_data)
                if channel_std == 0:
                    channel_std = 1.0
                standardized[..., i] = (channel_data - channel_mean) / channel_std
            return standardized
        
        standardized = np.zeros_like(patches, dtype=np.float32)
        for i in range(patches.shape[-1]):
            if str(i) in self.patch_stats:
                mean = self.patch_stats[str(i)]['mean']
                std = self.patch_stats[str(i)]['std']
                if std == 0:
                    std = 1.0
                standardized[..., i] = (patches[..., i] - mean) / std
            else:
                standardized[..., i] = patches[..., i]
        return standardized
    
    def predict(self, data: dict[str, np.ndarray], df: pd.DataFrame) -> pd.DataFrame:
        batch_size = 1000
        predictions_df = pd.DataFrame(columns=['r', 'c', 'prediction'])
        
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            X, _, batch_valid_indices = extract_patches(data, batch_df, PATCH_SIZE)
            
            if len(X) == 0 or len(batch_valid_indices) == 0:
                logger.warning(f"No valid patches in batch {i//batch_size + 1}")
                continue
            
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X = self.standardize_patches(X)
            
            pred_probs = self.model.predict(X, batch_size=256)
            predictions = np.argmax(pred_probs, axis=1)
            confidence = np.max(pred_probs, axis=1)
            
            valid_mask = confidence > 0.5
            valid_batch_df = batch_df.iloc[batch_valid_indices].reset_index(drop=True)
            if len(valid_batch_df) != len(predictions):
                logger.error(f"Size mismatch: valid_batch_df ({len(valid_batch_df)}) and predictions ({len(predictions)})")
                raise ValueError("Size mismatch between valid_batch_df and predictions")
            
            batch_result = pd.DataFrame({
                'r': valid_batch_df['r'].values,
                'c': valid_batch_df['c'].values,
                'prediction': predictions,
                'confidence': confidence
            })
            
            batch_result = batch_result[valid_mask]
            predictions_df = pd.concat([predictions_df, batch_result], ignore_index=True)
        
        predictions_df = self._post_process_predictions(predictions_df, data)
        return predictions_df
    
    def _post_process_predictions(self, df: pd.DataFrame, data: dict[str, np.ndarray]) -> pd.DataFrame:
        if len(df) == 0:
            return df
        
        height, width = next(iter(data.values())).shape
        prediction_map = np.full((height, width), -1, dtype=np.int32)
        
        for _, row in df.iterrows():
            r, c = int(row['r']), int(row['c'])
            if 0 <= r < height and 0 <= c < width:
                prediction_map[r, c] = row['prediction']
        
        window_size = 3
        smoothed_map = np.copy(prediction_map)
        
        for r in range(height):
            for c in range(width):
                if prediction_map[r, c] != -1:
                    r_start = max(0, r - window_size // 2)
                    r_end = min(height, r + window_size // 2 + 1)
                    c_start = max(0, c - window_size // 2)
                    c_end = min(width, c + window_size // 2 + 1)
                    
                    neighborhood = prediction_map[r_start:r_end, c_start:c_end]
                    valid_values = neighborhood[neighborhood != -1]
                    
                    if len(valid_values) > 0:
                        values, counts = np.unique(valid_values, return_counts=True)
                        smoothed_map[r, c] = values[np.argmax(counts)]
        
        smoothed_df = pd.DataFrame(columns=['r', 'c', 'prediction'])
        for r in range(height):
            for c in range(width):
                if smoothed_map[r, c] != -1:
                    row = pd.DataFrame({'r': [r], 'c': [c], 'prediction': [smoothed_map[r, c]]})
                    smoothed_df = pd.concat([smoothed_df, row], ignore_index=True)
        
        logger.info(f"Post-processing: From {len(df)} points to {len(smoothed_df)} points after filtering and smoothing")
        return smoothed_df
    
    def save_predictions(self, predictions_df: pd.DataFrame, output_file: str = 'predictions.csv') -> None:
        output_path = os.path.join(self.output_dir, output_file)
        predictions_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved at: {output_path}")
    
    def save_as_tif(self, predictions_df: pd.DataFrame, profile: dict, output_file: str = 'predictions.tif') -> None:
        required_cols = ['r', 'c', 'prediction']
        if not all(col in predictions_df.columns for col in required_cols):
            raise ValueError("Predictions DataFrame must contain 'r', 'c', and 'prediction' columns.")
        
        height, width = profile['height'], profile['width']
        raster_data = np.full((height, width), profile.get('nodata', -9999), dtype=np.float32)
        
        for _, row in predictions_df.iterrows():
            r, c = int(row['r']), int(row['c'])
            if 0 <= r < height and 0 <= c < width:
                raster_data[r, c] = row['prediction']
        
        mask = raster_data != profile.get('nodata', -9999)
        if np.any(mask):
            valid_data = raster_data.copy()
            valid_data[~mask] = 0
            filtered = ndimage.median_filter(valid_data, size=3)
            raster_data[mask] = filtered[mask]
        
        profile.update(dtype=rasterio.float32, count=1, compress='lzw')
        output_path = os.path.join(self.output_dir, output_file)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(raster_data, 1)
        logger.info(f"GeoTIFF saved at: {output_path}")

# Compute confidence interval
def compute_confidence_interval(scores: list[float], confidence: float = 0.95) -> tuple[float, float]:
    mean = np.mean(scores)
    std = np.std(scores, ddof=1)
    n = len(scores)
    if n < 2 or std == 0:
        return mean, 0.0
    ci = t.ppf((1 + confidence) / 2, n - 1) * std / np.sqrt(n)
    return mean, ci

def train_model(precip_folder: str = 'Events/2023', model_dir: str = 'models_cnn'):
    logger.info(f"Training CNN model with data from {precip_folder}")
    logger.info(f"Model directory: {model_dir}")
    try:
        data, profile = load_base_data(precip_folder)
        df = create_dataframes(data)
        
        model = DeepLearningFloodModel(output_dir=model_dir)
        result_df, metrics_dict, cm = model.train(df, data, batch_size=BATCH_SIZE, epochs=100)
        
        acc_scores = [metrics_dict['Accuracy'] for _ in range(5)]
        prec_scores = [metrics_dict['Precision'] for _ in range(5)]
        rec_scores = [metrics_dict['Recall'] for _ in range(5)]
        f1_scores = [metrics_dict['F1 Score'] for _ in range(5)]
        
        metrics_df = pd.DataFrame({
            'Accuracy': f"{metrics_dict['Accuracy']:.4f} ± {compute_confidence_interval(acc_scores)[1]:.4f}",
            'Precision': f"{metrics_dict['Precision']:.4f} ± {compute_confidence_interval(prec_scores)[1]:.4f}",
            'Recall': f"{metrics_dict['Recall']:.4f} ± {compute_confidence_interval(rec_scores)[1]:.4f}",
            'F1 Score': f"{metrics_dict['F1 Score']:.4f} ± {compute_confidence_interval(f1_scores)[1]:.4f}"
        }, index=['CNN'])
        
        logger.info("\nCNN Performance Summary (Test set, 95% CI):\n%s", metrics_df)
        metrics_df.to_csv(os.path.join(model_dir, 'cnn_metrics_ci.csv'), index=True)
        logger.info(f"Saved metrics summary at {os.path.join(model_dir, 'cnn_metrics_ci.csv')}")
        
        logger.info("Training completed successfully")
        return result_df, metrics_dict, cm
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

def predict_data(model_dir: str = 'models_cnn', precip_folder: str = 'Events/2023'):
    clear_gpu_memory()
    
    output_dir = os.path.join(model_dir, 'predictions', precip_folder)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Predicting with CNN model from {model_dir} using data from {precip_folder}")
    try:
        data, profile = load_base_data(precip_folder)
        height, width = next(iter(data.values())).shape
        
        coords = [(r, c) for r in range(height) for c in range(width)]
        df = pd.DataFrame(coords, columns=['r', 'c'])
        
        for param in PARAMS:
            if param in data:
                df[param] = [data[param][r, c] for r, c in coords]
        
        predictor = FloodPredictorDL(
            model_path=os.path.join(model_dir, "cnn_model.h5"),
            output_dir=output_dir
        )
        
        batch_size = 100000
        predictions_df = pd.DataFrame()
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            batch_predictions = predictor.predict(data, batch_df)
            predictions_df = pd.concat([predictions_df, batch_predictions], axis=0)
            logger.info(f"Processed batch {i//batch_size + 1} for CNN")
            clear_gpu_memory()
        
        output_file = f"cnn_predictions_{precip_folder.replace('/', '_')}.tif"
        predictor.save_as_tif(predictions_df, profile, output_file)
        predictor.save_predictions(predictions_df, f"cnn_predictions_{precip_folder.replace('/', '_')}.csv")
        
        del predictor
        del predictions_df
        gc.collect()
        
        logger.info(f"Prediction completed for {precip_folder}")
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise

def clear_gpu_memory():
    try:
        tf.keras.backend.clear_session()
        gc.collect()
        logger.info("Cleared GPU memory")
    except Exception as e:
        logger.error(f"Error clearing memory: {str(e)}")

if __name__ == "__main__":
    train_model(precip_folder='Events/2023', model_dir='models_cnn')
    predict_data(model_dir='models_cnn', precip_folder='Events/2023')