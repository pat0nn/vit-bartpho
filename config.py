"""Configuration parameters for the ViT-BARTpho Image Captioning model."""

# Model configuration
IMAGE_ENCODER_MODEL = "google/vit-large-patch16-224-in21k"
TEXT_DECODER_MODEL = "vinai/bartpho-word"

# Data paths
TRAIN_DATA_PATH = '/mnt/nvme/Algo/UIT-ViIC/annotations/uitviic_captions_train2017_v2.json'
TEST_DATA_PATH = '/mnt/nvme/Algo/UIT-ViIC/annotations/uitviic_captions_val2017_v2.json'
TRAIN_IMAGES_DIR = '/mnt/nvme/Algo/UIT-ViIC/uitviic_train2017'
TEST_IMAGES_DIR = '/mnt/nvme/Algo/UIT-ViIC/uitviic_val2017'
GROUNDTRUTH_FILE = '/mnt/nvme/Algo/UIT-ViIC/annotations/uitviic_captions_val2017_v2.json'

# # Data paths
# TRAIN_DATA_PATH = '/kaggle/input/uit-viic/annotations/uitviic_captions_train2017_v2.json'
# TEST_DATA_PATH = '/kaggle/input/uit-viic/annotations/uitviic_captions_val2017_v2.json'
# TRAIN_IMAGES_DIR = '/kaggle/input/uit-viic/uitviic_train2017'
# TEST_IMAGES_DIR = '/kaggle/input/uit-viic/uitviic_val2017'

# GROUNDTRUTH_FILE = '/kaggle/working/vit-bartpho/data/groundtruth_captions_val2017.json'

# Evaluation settings
USE_COCO_EVAL = True  # Whether to use COCO evaluation metrics
COCO_ANNOTATION_FILE = '/mnt/nvme/Algo/UIT-ViIC/annotations/uitviic_captions_val2017_v2.json'  # Path to COCO format annotation file
SAVE_COCO_FORMAT = True  # Whether to save predictions in COCO format
SKIP_SPICE = True  # Whether to skip SPICE metric due to Java errors

# Training parameters
MAX_TARGET_LENGTH = 64
SEED = 42
BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
WEIGHT_DECAY = 1e-6
USE_FP16 = True

# Caption type selection
USE_SEGMENT_CAPTION = True  # Set to True to use segment_caption, False to use caption

# Quick testing/subset parameters
USE_SUBSET = True  # Set to True for quick testing with subset of data
TRAIN_SUBSET_SIZE = 3  # Number of samples to use from training set
TEST_SUBSET_SIZE = 3    # Number of samples to use from test set

# Inference parameters
MAX_LENGTH = 24
NUM_BEAMS = 3

# Paths
DATASET_SAVE_PATH = './dataset/image_caption_dataset'
OUTPUT_DIR = './output'
LOGS_DIR = './logs'
WANDB_PROJECT = "ViT-BARTpho_UIT-ViIC_4-19_original-caption"
WANDB_NAME = "experiment"

# Device
DEVICE = 'cpu'  # 'cuda' or 'cpu'
