# ==============================================================================
# NeuroScan Workspace: Full-Stack Adaptive Medical Diagnostic Interface (app.py)
# ==============================================================================
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", message=".*google.generativeai.*")
import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import io


# Force stdout and stderr to use UTF-8 encoding on Windows to prevent UnicodeEncodeError
if sys.stdout and getattr(sys.stdout, 'encoding', None) != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, io.UnsupportedOperation):
        if hasattr(sys.stdout, 'buffer'):
            try:
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            except Exception:
                pass
if sys.stderr and getattr(sys.stderr, 'encoding', None) != 'utf-8':
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, io.UnsupportedOperation):
        if hasattr(sys.stderr, 'buffer'):
            try:
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
            except Exception:
                pass

import cv2
import json
import datetime
import urllib.parse
import shutil
import glob
import time
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

bg_executor = ThreadPoolExecutor(max_workers=3)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torch_models
import torchvision.transforms as transforms
from PIL import Image
import pymongo
import gradio as gr

# Force matplotlib to non-interactive backend to prevent GUI thread conflicts
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Optional dependency loaders for DICOM and S3
try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

try:
    import boto3
    from botocore.exceptions import NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


# ==============================================================================
# 1. OPTIONAL DEPENDENCY LOADERS & CONFIGURATION
# ==============================================================================
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# ==============================================================================
# 2. PYTORCH SPECIALIST MODEL ARCHITECTURES
# ==============================================================================
class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.dw(x)))
        x = F.relu(self.bn2(self.pw(x)))
        return x

class LiteResBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = DSConv(in_channels, out_channels, stride)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x))

class LiteBrainNet2(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = LiteResBlock2(32, 64, stride=2)
        self.layer2 = LiteResBlock2(64, 128, stride=2)
        self.layer3 = LiteResBlock2(128, 128, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class InfectiousBrainNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class StableMetabolicNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class NeoplasticBrainNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop2d_1 = nn.Dropout2d(p=0.2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop2d_2 = nn.Dropout2d(p=0.3)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.6)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.drop2d_1(x)
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        x = self.drop2d_2(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class CustomLiverNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.global_pool(x)
        x = self.fc_layers(x)
        return x

class MicroLiverNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.6)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


# ==============================================================================
# 3. MEDICAL DIAGNOSTIC HUB (T1/T2 CLASSIFICATION)
# ==============================================================================
class MedicalAIHub:
    def __init__(self, paths, gen_class_list):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.paths = paths
        self.gen_classes = gen_class_list
        # Generalist: ResNet-50
        self.gen = torch_models.resnet50()
        num_ftrs = self.gen.fc.in_features
        self.gen.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, len(gen_class_list))
        )
        
        self.generalist_loaded = False
        if os.path.exists(paths['generalist']):
            try:
                self.gen.load_state_dict(torch.load(paths['generalist'], map_location=self.device, weights_only=True))
                self.gen.to(self.device).eval()
                self.generalist_loaded = True
                print("✓ Generalist weights loaded successfully.")
            except Exception as e:
                print(f"⚠️ Error loading generalist weights: {e}. Running initialized model.")
        else:
            print(f"⚠️ Generalist weights not found at '{paths['generalist']}'. Running initialized model.")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def get_specialist_instance(self, category):
        cat = category.lower()
        if 'genetic' in cat:
            diseases = ['Fukuyama Muscular Dystrophy', 'NFM 1 with OGIE', 'Tuberous Sclerosis', 'Walker-Warburg Syndrome']
            return LiteBrainNet2(num_classes=4), self.paths['genetic'], diseases

        if 'infectious' in cat:
            diseases = ['Acute Cerebellitis in HIV', 'Acute Unilateral Cerebellitis in HIV',
                        'Congenital Toxoplasmosis', 'Japanese B Encephalitis or Epstein-Barr Encephalitis',
                        'Rasmussens Encephalitis']
            return InfectiousBrainNet(num_classes=5), self.paths['infectious'], diseases

        if 'malformations' in cat or 'developmental' in cat:
            diseases = ['Balloon Cell Cortical Dysplasia', 'Pachygyria with Cerebellar Hypoplasia', 'Perisylvian Syndrome']
            return LiteBrainNet2(num_classes=3), self.paths['malformations'], diseases

        if 'metabolic' in cat:
            diseases = ['Osmotic Demyelination Syndrome', 'Typical Adrenoleukodystrophy']
            return StableMetabolicNet(num_classes=2), self.paths['metabolic'], diseases

        if 'neoplastic' in cat or 'tumor' in cat or 'tumour' in cat:
            diseases = ['Optic Glioma', 'Plexiform Neurofibroma with Sphenoid Wing Dysplasia']
            return NeoplasticBrainNet(num_classes=2), self.paths['neoplastic'], diseases

        if 'malignant' in cat:
            diseases = ['Hepatocellular Carcinoma (HCC) and Dysplastic Nodule',
                        'Hepatocellular_Carcinoma', 'Inferior Vena Cava (IVC) Leiomyosarcoma']
            return CustomLiverNet(num_classes=3), self.paths['malignant'], diseases

        if 'ductal' in cat or 'ductual' in cat:
            diseases = ['Carolis Disease', 'Cholangiocarcinoma']
            return MicroLiverNet(num_classes=2), self.paths['ductal'], diseases

        return None, None, None

    def _tensor_from_np(self, img_np):
        pil_img = Image.fromarray(img_np.astype('uint8'))
        return self.transform(pil_img).unsqueeze(0).to(self.device)

    def _run_specialist(self, img_tensor, category):
        spec_arch, weight_path, spec_classes = self.get_specialist_instance(category)
        if spec_arch is None:
            return None, None
            
        if os.path.exists(weight_path):
            try:
                spec_arch.load_state_dict(torch.load(weight_path, map_location=self.device, weights_only=True))
                print(f"✓ Loaded specialist weights for category: {category}")
            except Exception as e:
                print(f"⚠️ Error loading specialist weights: {e}. Running initialized model.")
        else:
            print(f"⚠️ Specialist weights not found at '{weight_path}'. Running initialized model.")
            
        spec_arch.to(self.device).eval()
        with torch.no_grad():
            probs = torch.softmax(spec_arch(img_tensor), dim=1)
            conf, idx = torch.max(probs, 1)
        label = spec_classes[idx.item()] if spec_classes else f"Class {idx.item()}"
        return label, float(conf)

    def diagnose_array(self, img_np):
        img_tensor = self._tensor_from_np(img_np)
        with torch.no_grad():
            gen_probs = torch.nn.functional.softmax(self.gen(img_tensor), dim=1)
            gen_conf, gen_idx = torch.max(gen_probs, 1)
            category = self.gen_classes[gen_idx.item()]

        print(f"  [Generalist Hub] Classified Category: {category} (Conf: {gen_conf.item()*100:.2f}%)")
        
        if gen_conf.item() < 0.90 and self.generalist_loaded:
            return f"Low Confidence: {category}", float(gen_conf)
        spec_label, spec_conf = self._run_specialist(img_tensor, category)
        if spec_label is not None:
            return spec_label, spec_conf
            
        return category, float(gen_conf)


# ==============================================================================
# 4. REINFORCEMENT LEARNING PREPROCESSING ENVIRONMENT
# ==============================================================================
def apply_clahe(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

def apply_median_blur(img):
    return cv2.medianBlur(img, 5)

def apply_gaussian_blur(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def apply_sharpen(img):
    kernel = np.array([[0, -1, 0], 
                       [-1, 5, -1], 
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

FUNCTION_MAP = {
    "clahe": apply_clahe,
    "median": apply_median_blur,
    "gaussian": apply_gaussian_blur,
    "sharpen": apply_sharpen
}

def analyze_image(img):
    img_uint8 = img.astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    
    brightness = np.mean(gray) / 255.0
    contrast = np.std(gray) / 128.0
    noise = float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 1000.0
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges > 0)
    
    return np.array([brightness, contrast, noise, edge_density])

def process_image_with_agent_and_hub(img, Q_table, hub, max_steps=2):
    current_img = img.copy()
    execution_steps = []
    log_messages = []
    
    # 1. Run classifier on the raw image to check initial confidence
    img_tensor = hub._tensor_from_np(current_img)
    with torch.no_grad():
        gen_probs = torch.nn.functional.softmax(hub.gen(img_tensor), dim=1)
        gen_conf, gen_idx = torch.max(gen_probs, 1)
        category = hub.gen_classes[gen_idx.item()]
        
    initial_conf = gen_conf.item()
    log_messages.append(f"🔍 [Initial Scan Check] Category: '{category}' | Confidence: {initial_conf*100:.2f}%")
    
    if initial_conf >= 0.90:
        log_messages.append("🎯 Initial confidence is already >= 90%. Skipping preprocessing.")
        return current_img, ["identity_pass"], "\n".join(log_messages)
        
    conf = initial_conf
    log_messages.append(f"⚠️ Confidence ({initial_conf*100:.2f}%) is below 90% threshold. Initiating RL Preprocessing...")
    
    for step in range(max_steps):
        state_metrics = analyze_image(current_img)
        state_key = ",".join(map(str, (state_metrics * 10).astype(int)))
        
        log_messages.append(f"\n[Step {step+1}]")
        log_messages.append(f" - Image Metrics: Brightness={state_metrics[0]:.2f}, Contrast={state_metrics[1]:.2f}, Noise={state_metrics[2]:.2f}, Edges={state_metrics[3]:.2f}")
        log_messages.append(f" - Current State Key: [{state_key}]")
        
        # Decide action using Q-table
        if state_key in Q_table:
            actions_q = Q_table[state_key]
            action = max(actions_q, key=actions_q.get)
            log_messages.append(f" - Q-Table recommendation: '{action}'")
        else:
            action = "none"
            log_messages.append(f" - Q-Table state not found (out-of-distribution). Defaulting to: 'none'")
            
        if action == "stop":
            log_messages.append(f" - Action applied: 'stop' (No transformation performed)")
            execution_steps.append("stop")
            break
            
        prev_conf = conf
        if action == "none":
            # If Q-table doesn't know, find the best fallback that boosts confidence
            log_messages.append(f" - Finding best fallback transformation to boost confidence from {prev_conf*100:.2f}%...")
            best_action = "none"
            best_conf = conf
            best_img = current_img.copy()
            
            for act_name, transform_func in FUNCTION_MAP.items():
                if act_name in execution_steps:
                    continue
                temp_img = transform_func(current_img)
                temp_tensor = hub._tensor_from_np(temp_img)
                with torch.no_grad():
                    temp_probs = torch.nn.functional.softmax(hub.gen(temp_tensor), dim=1)
                    temp_conf, _ = torch.max(temp_probs, 1)
                    
                if temp_conf.item() > best_conf:
                    best_conf = temp_conf.item()
                    best_img = temp_img
                    best_action = act_name
            
            action = best_action
            current_img = best_img
            conf = best_conf
            execution_steps.append(action)
            log_messages.append(f" - Action applied: Fallback [{action}]")
            log_messages.append(f" - Confidence change: {prev_conf*100:.2f}% ➡️ {conf*100:.2f}% (Change: {(conf - prev_conf)*100:+.2f}%)")
        else:
            # Apply normal Q-table action
            if action in FUNCTION_MAP:
                current_img = FUNCTION_MAP[action](current_img)
                img_tensor = hub._tensor_from_np(current_img)
                with torch.no_grad():
                    temp_probs = torch.nn.functional.softmax(hub.gen(img_tensor), dim=1)
                    temp_conf, _ = torch.max(temp_probs, 1)
                conf = temp_conf.item()
            execution_steps.append(action)
            log_messages.append(f" - Action applied: RL [{action}]")
            log_messages.append(f" - Confidence change: {prev_conf*100:.2f}% ➡️ {conf*100:.2f}% (Change: {(conf - prev_conf)*100:+.2f}%)")
            
        # Stop early if the step pushed us over 90%
        if conf >= 0.90:
            log_messages.append(f"🎯 Success! Preprocessing Step {step+1} successfully pushed confidence to {conf*100:.2f}% (>= 90%). Stopping.")
            break
            
    if not execution_steps:
        execution_steps.append("raw_identity_passthrough")
        
    return current_img, execution_steps, "\n".join(log_messages)


# ==============================================================================
# 5. PERSISTENT STORAGE (MONGODB INTEGRATION)
# ==============================================================================
def get_mongodb_connection():
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    try:
        # Check connection with a brief timeout
        client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
        client.admin.command('ping')
        return client["NeuroScan_DB"], "Connected ✅"
    except Exception as e:
        print(f"⚠️ MongoDB Connection offline: {e}")
        return None, "Offline ❌ (Inference active)"

def upload_to_s3(file_path):
    """
    Uploads a file to an S3 bucket and returns the file URL.
    """
    if not BOTO3_AVAILABLE:
        print("[WARNING] S3 Upload: boto3 is not available.")
        return None

    bucket_name = os.getenv("AWS_S3_BUCKET")
    aws_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

    if not bucket_name or not aws_key or not aws_secret:
        print("[WARNING] S3 Upload: Missing AWS credentials/configurations. Skipping upload.")
        return None
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret,
            region_name=aws_region
        )
        object_name = os.path.basename(file_path)
        s3_client.upload_file(file_path, bucket_name, object_name)
        s3_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': object_name},
            ExpiresIn=604800  # URL valid for 7 days
        )
        print(f"[OK] S3 Upload: File successfully uploaded; presigned URL generated.")
        return s3_url
    except NoCredentialsError:
        print("[WARNING] S3 Upload: Credentials not available.")
        return None
    except Exception as e:
        print(f"[WARNING] S3 Upload failed: {e}")
        return None

def load_medical_image(file_path):
    """
    Loads a standard image or a DICOM file.
    Returns a normalized 3-channel RGB numpy array.
    """
    if file_path.lower().endswith('.dcm'):
        if not PYDICOM_AVAILABLE:
            print("[WARNING] DICOM: pydicom is not available. Cannot parse DICOM.")
            return np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        try:
            ds = pydicom.dcmread(file_path)
            pixel_array = ds.pixel_array
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
            p_min = pixel_array.min()
            p_max = pixel_array.max()
            if p_max > p_min:
                normalized = ((pixel_array - p_min) / (p_max - p_min) * 255.0).astype(np.uint8)
            else:
                normalized = np.zeros_like(pixel_array, dtype=np.uint8)
            if len(normalized.shape) == 2:
                normalized = np.stack([normalized] * 3, axis=-1)
            elif len(normalized.shape) == 3 and normalized.shape[2] == 1:
                normalized = np.concatenate([normalized] * 3, axis=-1)
            print(f"[OK] DICOM Loaded: {file_path}")
            return normalized
        except Exception as e:
            print(f"[WARNING] DICOM Loading failed: {e}. Falling back to random array.")
            return np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    else:
        img = cv2.imread(file_path)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            print(f"[WARNING] Image loading failed: {file_path}. Falling back to random array.")
            return np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

def get_mongodb_connection():
    # If the user has a MONGO_URI env var set, use it. Otherwise, use the Atlas cluster connection string.
    username = "agentic_logs"
    password = "Agentic_log@123"
    escaped_username = urllib.parse.quote_plus(username)
    escaped_password = urllib.parse.quote_plus(password)
    default_atlas_uri = f"mongodb+srv://{escaped_username}:{escaped_password}@cluster0.a3jb3u6.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    
    mongo_uri = os.getenv("MONGO_URI", default_atlas_uri)
    try:
        # Check connection with a brief timeout
        client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=4000)
        client.admin.command('ping')
        return client["NeuroScan_DB"], "Connected ✅"
    except Exception as e:
        print(f"⚠️ MongoDB Connection offline: {e}")
        return None, "Offline ❌ (Inference active)"

def log_agent_draft(db, status="Incomplete", step_logs=None, confidence=None, diagnosis=None, s3_url=None, patient_name=None, patient_id=None, doc_id=None):
    if db is None:
        return None
        
    logs_collection = db["agent_result_logs"]
    
    # Clean step prefixes
    clean_steps = []
    if step_logs:
        for step in step_logs:
            if ":" in str(step):
                parts = str(step).split(":")
                clean_steps.append(parts[-1].strip())
            else:
                clean_steps.append(str(step))
    else:
        clean_steps = ["none"]

    draft_doc = {
        "timestamp": datetime.datetime.utcnow(),
        "execution_status": status,
        "patient_name": patient_name if patient_name else "Unknown",
        "patient_id": patient_id if patient_id else "Unknown",
        "s3_url": s3_url,
        "agent_steps": clean_steps,
        "diagnostic_context": {
            "disease": diagnosis if diagnosis else "Unknown",
            "confidence": float(confidence) if confidence is not None else 0.0
        },
        "human_in_the_loop": {
            "status": "Pending UI Feedback"
        }
    }
    if doc_id:
        draft_doc["_id"] = doc_id
        
    try:
        res = logs_collection.insert_one(draft_doc)
        return res.inserted_id
    except Exception as e:
        print(f"⚠️ Logging failed: {e}")
        return None

def submit_doctor_feedback(doc_id_str, feedback_text):
    global db_client
    if not doc_id_str:
        return "⚠️ No active analysis session. Please analyze a scan first."
    if not feedback_text:
        return "⚠️ Feedback text cannot be empty."
        
    # Check if the database is online and the document exists
    if db_client is not None:
        try:
            from bson.objectid import ObjectId
            logs_collection = db_client["agent_result_logs"]
            res = logs_collection.update_one(
                {"_id": ObjectId(doc_id_str)},
                {
                    "$set": {
                        "human_in_the_loop": {
                            "status": "Feedback Submitted",
                            "doctor_feedback": feedback_text,
                            "feedback_timestamp": datetime.datetime.utcnow()
                        }
                    }
                }
            )
            if res.modified_count > 0:
                print(f"[OK] Doctor Feedback updated for Document {doc_id_str}")
                return "✅ Doctor Feedback submitted and saved to MongoDB successfully!"
        except Exception as e:
            print(f"[WARNING] Online feedback update failed: {e}. Checking offline queue...")
            
    # Fallback/Offline check: look in offline_queue/
    import glob
    import json
    try:
        log_files = glob.glob("offline_queue/log_*.json")
        for file_path in log_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
            if data.get("doc_id") == doc_id_str:
                data["human_in_the_loop"] = {
                    "status": "Feedback Submitted",
                    "doctor_feedback": feedback_text,
                    "feedback_timestamp": datetime.datetime.utcnow().isoformat()
                }
                with open(file_path, 'w') as f_w:
                    json.dump(data, f_w, indent=4)
                print(f"[OFFLINE QUEUE] Local doctor feedback updated for {doc_id_str}")
                return "✅ App is offline, but Doctor Feedback was saved locally and will sync once online!"
    except Exception as e:
        print(f"[WARNING] Failed to update local feedback file: {e}")
            
    return "❌ Failed to submit feedback: Document not found online or in offline queue."

def bg_save_task(filepath, actions, confidence, label, patient_name, patient_id, doc_id):
    """
    Background worker task to upload raw scans to S3 and write diagnostic logs to MongoDB.
    If either service is offline or fails, saves logs locally to the offline_queue/ directory.
    """
    global db_client
    s3_url = None
    s3_uploaded = False
    mongo_logged = False
    
    # 1. Attempt S3 upload
    try:
        if filepath and os.path.exists(filepath):
            s3_url = upload_to_s3(filepath)
            if s3_url:
                s3_uploaded = True
                print(f"[BG SAVE] S3 upload successful: {s3_url}")
    except Exception as e:
        print(f"[BG SAVE WARNING] S3 upload failed: {e}")
        
    # 2. Attempt MongoDB logging using the pre-generated doc_id
    if db_client is not None:
        try:
            status = "Successful Run" if "Low Confidence" not in label else "Low Confidence Flagged"
            inserted_id = log_agent_draft(
                db=db_client,
                status=status,
                step_logs=actions,
                confidence=confidence,
                diagnosis=label,
                s3_url=s3_url,
                patient_name=patient_name,
                patient_id=patient_id,
                doc_id=doc_id
            )
            if inserted_id:
                mongo_logged = True
                print(f"[BG SAVE] MongoDB Atlas log created: {inserted_id}")
        except Exception as e:
            print(f"[BG SAVE WARNING] MongoDB logging failed: {e}")

    # 3. Fallback: Local offline queue logging if any step failed
    if not s3_uploaded or not mongo_logged:
        try:
            os.makedirs("offline_queue", exist_ok=True)
            offline_filepath = filepath
            
            # Copy to offline folder if it's a temporary file that might be cleaned up
            if filepath and os.path.exists(filepath):
                base_name = os.path.basename(filepath)
                timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                safe_name = f"offline_queue/scan_{timestamp_str}_{base_name}"
                shutil.copy(filepath, safe_name)
                offline_filepath = os.path.abspath(safe_name)
            
            offline_log = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "doc_id": str(doc_id),
                "filepath": offline_filepath,
                "patient_name": patient_name,
                "patient_id": patient_id,
                "actions": actions,
                "confidence": confidence,
                "label": label,
                "s3_url": s3_url,
                "human_in_the_loop": {
                    "status": "Pending UI Feedback"
                }
            }
            
            log_path = f"offline_queue/log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{doc_id}.json"
            with open(log_path, 'w') as f_out:
                json.dump(offline_log, f_out, indent=4)
            print(f"[BG SAVE] Local offline log written: {log_path}")
        except Exception as err:
            print(f"[BG SAVE ERROR] Failed to write local fallback log: {err}")

def sync_offline_jobs():
    """
    Checks for pending offline logs and attempts to sync them to S3 and MongoDB.
    """
    global db_client, db_status
    
    # Attempt to reconnect if currently offline
    if db_client is None:
        db_client, db_status = get_mongodb_connection()
        if db_client is None:
            return # Still offline, retry on next loop
            
    log_files = glob.glob("offline_queue/log_*.json")
    if not log_files:
        return
        
    print(f"[OFFLINE SYNC] Found {len(log_files)} pending offline logs. Starting sync...")
    for file_path in log_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            doc_id_str = data.get("doc_id")
            from bson.objectid import ObjectId
            doc_id = ObjectId(doc_id_str)
            
            # Check if S3 upload is needed
            s3_url = data.get("s3_url")
            scan_file = data.get("filepath")
            if not s3_url and scan_file and os.path.exists(scan_file):
                s3_url = upload_to_s3(scan_file)
                if s3_url:
                    data["s3_url"] = s3_url
                    # Update local json
                    with open(file_path, 'w') as f_w:
                        json.dump(data, f_w, indent=4)
            
            # Write to MongoDB Atlas
            status = "Successful Run" if "Low Confidence" not in data["label"] else "Low Confidence Flagged"
            
            # Ensure we preserve doctor feedback if submitted offline
            hitl_data = data.get("human_in_the_loop", {"status": "Pending UI Feedback"})
            
            logs_collection = db_client["agent_result_logs"]
            
            # Clean step prefixes
            clean_steps = []
            if data["actions"]:
                for step in data["actions"]:
                    if ":" in str(step):
                        parts = str(step).split(":")
                        clean_steps.append(parts[-1].strip())
                    else:
                        clean_steps.append(str(step))
            else:
                clean_steps = ["none"]
                
            draft_doc = {
                "_id": doc_id,
                "timestamp": datetime.datetime.utcnow(),
                "execution_status": status,
                "patient_name": data["patient_name"] if data["patient_name"] else "Unknown",
                "patient_id": data["patient_id"] if data["patient_id"] else "Unknown",
                "s3_url": s3_url,
                "agent_steps": clean_steps,
                "diagnostic_context": {
                    "disease": data["label"] if data["label"] else "Unknown",
                    "confidence": float(data["confidence"]) if data["confidence"] is not None else 0.0
                },
                "human_in_the_loop": hitl_data
            }
            
            logs_collection.insert_one(draft_doc)
            print(f"[OFFLINE SYNC] Successfully synced document {doc_id_str} to MongoDB.")
            
            # Delete processed logs
            os.remove(file_path)
            if scan_file and scan_file.startswith("offline_queue/") and os.path.exists(scan_file):
                try:
                    os.remove(scan_file)
                except Exception as file_err:
                    print(f"[OFFLINE SYNC] Failed to delete scan file {scan_file}: {file_err}")
                    
        except Exception as e:
            print(f"[OFFLINE SYNC ERROR] Failed to sync log {file_path}: {e}")

def start_offline_sync_loop():
    """
    Launches a daemon thread that polls the offline queue folder for pending logs.
    """
    def loop():
        while True:
            time.sleep(30)
            try:
                sync_offline_jobs()
            except Exception as e:
                print(f"[OFFLINE SYNC LOOP ERROR] {e}")
                
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    print("🔄 [OFFLINE SYNC] Background synchronization thread started.")

# ==============================================================================
# 6. DUAL-COLLECTION VECTOR DATABASE & RETRIEVAL (Lightweight Cosine Search)
# ==============================================================================
class LightVectorDB:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedder = SentenceTransformer(model_name)
                print("✓ SentenceTransformer loaded successfully.")
            except Exception as e:
                self.embedder = None
                print(f"⚠️ Could not load SentenceTransformer: {e}. Vector search will run in mock mode.")
        else:
            self.embedder = None
            print("⚠️ SentenceTransformers not installed. Running in mock mode.")
            
        self.collections = {
            "medical_base": [],
            "agent_result_logs": []
        }

    def add_documents(self, collection_name, texts, metadatas=None):
        if collection_name not in self.collections:
            return
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]
            
        if self.embedder is not None:
            vectors = self.embedder.encode(texts, convert_to_numpy=True)
            for text, vector, meta in zip(texts, vectors, metadatas):
                norm = np.linalg.norm(vector)
                norm_vec = vector / norm if norm > 0 else vector
                self.collections[collection_name].append({
                    "text": text,
                    "vector": norm_vec,
                    "metadata": meta
                })
        else:
            for text, meta in zip(texts, metadatas):
                self.collections[collection_name].append({
                    "text": text,
                    "vector": None,
                    "metadata": meta
                })

    def query(self, collection_name, query_text, top_k=2):
        if collection_name not in self.collections or not self.collections[collection_name]:
            return []
            
        docs = self.collections[collection_name]
        if self.embedder is not None:
            if hasattr(query_text, 'shape'):
                query_vector = query_text
            else:
                query_vector = self.embedder.encode([query_text], convert_to_numpy=True)[0]
            norm = np.linalg.norm(query_vector)
            query_vector = query_vector / norm if norm > 0 else query_vector
            
            scores = []
            for doc in docs:
                if doc["vector"] is not None:
                    similarity = np.dot(query_vector, doc["vector"])
                    scores.append((similarity, doc))
                else:
                    scores.append((0.0, doc))
            scores.sort(key=lambda x: x[0], reverse=True)
            return [item[1] for item in scores[:top_k]]
        else:
            words = query_text.lower().split()
            matched = []
            for doc in docs:
                score = sum(1 for w in words if w in doc["text"].lower())
                matched.append((score, doc))
            matched.sort(key=lambda x: x[0], reverse=True)
            return [item[1] for item in matched[:top_k]]


# ==============================================================================
# 7. CLINICAL RAG EXPLANATION GENERATOR
# ==============================================================================
class MedicalRAGPipeline:
    def __init__(self, vector_db):
        self.db = vector_db
        self.gemini_ready = False
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if GEMINI_AVAILABLE and api_key:
            try:
                genai.configure(api_key=api_key)
                
                # Check available models dynamically
                available_models = []
                try:
                    for m in genai.list_models():
                        if 'generateContent' in m.supported_generation_methods:
                            available_models.append(m.name)
                except Exception as model_err:
                    print(f"⚠️ Warning listing models: {model_err}")
                    
                candidates = [
                    'models/gemini-2.5-flash',
                    'models/gemini-1.5-flash', 
                    'models/gemini-1.5-pro', 
                    'models/gemini-pro',
                    'gemini-1.5-flash', 
                    'gemini-pro'
                ]
                
                selected_model = None
                if available_models:
                    for cand in candidates:
                        if cand in available_models or cand.replace('models/', '') in [m.replace('models/', '') for m in available_models]:
                            selected_model = cand
                            break
                    if not selected_model:
                        selected_model = available_models[0]
                else:
                    selected_model = 'gemini-1.5-flash'
                    
                print(f"✓ Using Gemini Model: {selected_model}")
                self.model = genai.GenerativeModel(selected_model)
                self.gemini_ready = True
            except Exception as e:
                print(f"⚠️ Error setting up Gemini: {e}. Fallback active.")
        else:
            print("⚠️ GEMINI_API_KEY not found or google-generativeai not installed. Local fallback active.")

    def run_query(self, clinical_query):
        # Parallel RAG Retrieval using ThreadPoolExecutor for maximum speed
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_textbook = executor.submit(self.db.query, "medical_base", clinical_query, 2)
            future_history = executor.submit(self.db.query, "agent_result_logs", clinical_query, 2)
            textbook_results = future_textbook.result()
            historical_results = future_history.result()

        context_textbook = "\n".join([f"- {doc['text']}" for doc in textbook_results])
        context_history = "\n".join([f"- [Case {doc['metadata'].get('patient_id', 'N/A')}]: {doc['text']}" for doc in historical_results])

        # Step D: Synthesis and RAG generation
        prompt = f"""
Synthesize a comprehensive, professional clinical justification report based on the medical imaging case below.
DO NOT include conversational introductory filler (such as "As an expert..."). Start DIRECTLY with the structured report headers.

Patient Scan Case:
"{clinical_query}"

---
CONTEXT 1: FOUNDATIONAL TEXTBOOK CRITERIA & PATHOLOGY
{context_textbook}

---
CONTEXT 2: HISTORICAL AGENT EXECUTION RECORDS
{context_history}
---

Your report MUST follow this exact Markdown structure:

### **1. DIAGNOSTIC PIPELINE INTERPRETATION**
*   **Primary Classification:** State the diagnosed condition clearly.
*   **Preprocessing Agent Actions:** Explain the RL agent actions taken to optimize image quality.
*   **Historical Execution Comparison:** Compare findings with historical patient logs.

---

### **2. DETAILED EMPATHETIC CLINICAL EXPLANATION**
Provide a detailed, reassuring medical explanation translated into compassionate clinical language:
*   **Disease Definition:** Explain the pathology clearly.
*   **MRI Findings:** List key MRI imaging characteristics with bold bullet points.
*   **Clinical Features:** List typical clinical symptoms and features.
*   **Diagnostic Protocol & Prognosis:** Outline management and next steps.
"""
        if self.gemini_ready:
            try:
                # Full report generation without premature token truncation (max_output_tokens=2048)
                response = self.model.generate_content(
                    prompt,
                    generation_config={"max_output_tokens": 2048, "temperature": 0.2}
                )
                if hasattr(response, 'text') and response.text:
                    badge = "<div style='display: inline-block; background-color: #EBF8FF; border: 1px solid #BEE3F8; color: #2B6CB0; padding: 4px 12px; border-radius: 12px; font-weight: 700; font-size: 12px; margin-bottom: 15px;'>🤖 Report Source: Google Gemini LLM (Live Synthesis)</div>\n\n"
                    return badge + response.text
                return self._generate_fallback_report(clinical_query, textbook_results, historical_results)
            except Exception:
                return self._generate_fallback_report(clinical_query, textbook_results, historical_results)
        else:
            return self._generate_fallback_report(clinical_query, textbook_results, historical_results)

    def _generate_fallback_report(self, query, textbook_docs, history_docs):
        diagnosis = "Indicated Clinical Pathology"
        textbook_raw = textbook_docs[0]['text'] if textbook_docs else ""
        textbook_context_snippet = ""
        
        try:
            db_entry = json.loads(textbook_raw)
            diagnosis = db_entry.get("disease", "Indicated Clinical Pathology")
            mri_list = "\n".join([f"   *   {item}" for item in db_entry.get("mri_findings", [])])
            clinical_list = "\n".join([f"   *   {item}" for item in db_entry.get("clinical_features", [])])
            
            textbook_context_snippet = f"""*   **Disease:** **{db_entry.get('disease')}**
*   **Definition:** {db_entry.get('definition')}

### **3. MRI Findings:**
{mri_list}

### **4. Clinical Features:**
{clinical_list}

*   **Diagnostic Protocol:** {db_entry.get('diagnostic_protocol')}
*   **Prognosis:** {db_entry.get('prognosis')}
*   **Reference:** *{db_entry.get('reference')}*"""
        except Exception:
            textbook_context_snippet = textbook_raw if textbook_raw else "No textbook rules loaded."
            
        history_context_snippet = history_docs[0]['text'] if history_docs else "No clinical cases loaded."
        
        badge = "<div style='display: inline-block; background-color: #FEFCBF; border: 1px solid #FAF089; color: #744210; padding: 4px 12px; border-radius: 12px; font-weight: 700; font-size: 12px; margin-bottom: 15px;'>⚙️ Report Source: Local Medical Rule-Engine (Offline / Fallback)</div>\n\n"
        fallback_txt = badge + f"""### **[CLINICAL EXECUTIVE SUMMARY]**

### **1. DIAGNOSTIC PIPELINE INTERPRETATION**
The diagnostic system evaluated the patient's query: **"{query}"**.
*   **Primary Classification:** **{diagnosis}**
*   **Pipeline Path:** Image processing metrics indicate adaptive enhancement was applied. Contrast and structural edges were optimized to maximize output confidence.
*   **Execution Log Reference:** {history_context_snippet}

---

### **2. DETAILED EMPATHETIC CLINICAL EXPLANATION**
We have reviewed your scan results. According to clinical protocols, the findings relate to what medical textbooks define as:

{textbook_context_snippet}

---
*Please be reassured that this report has been analyzed by a specialized digital assistant. Your healthcare provider will discuss these results in detail, coordinate next steps, and tailor a management plan specific to your health journey.*
"""
        return fallback_txt


# ==============================================================================
# 8. CLINICAL KNOWLEDGE DATABASE & SEEDING
# ==============================================================================
DISEASE_DB = [
    {
        "disease": "Liver Hemangioma (Benign)",
        "definition": "A liver hemangioma is a non-cancerous mass composed of a cluster of blood vessels and is the most common benign liver tumor.",
        "mri_findings": [
            "Well-defined lesion",
            "Hyperintense appearance on T2-weighted MRI",
            "Peripheral nodular enhancement after contrast administration",
            "Progressive centripetal fill-in on delayed phases"
        ],
        "clinical_features": [
            "Usually asymptomatic",
            "Often discovered incidentally during imaging",
            "Large lesions may rarely cause abdominal discomfort"
        ],
        "diagnostic_protocol": "MRI with contrast is highly useful for differentiating hemangiomas from malignant liver lesions.",
        "prognosis": "Most hemangiomas remain stable and do not require treatment unless symptomatic.",
        "reference": "Harrison's Principles of Internal Medicine"
    },
    {
        "disease": "Fukuyama Muscular Dystrophy",
        "definition": "An autosomal recessive congenital muscular dystrophy characterized by brain malformations, progressive muscle weakness, and intellectual disability.",
        "mri_findings": [
            "Cobblestone lissencephaly (type II lissencephaly)",
            "Diffuse white matter myelination delay",
            "Cerebellar cysts and hypoplasia",
            "Ventriculomegaly"
        ],
        "clinical_features": [
            "Severe hypotonia and floppy infant syndrome",
            "Generalized motor delays and muscle wasting",
            "Seizures and severe cognitive deficits"
        ],
        "diagnostic_protocol": "Brain MRI combined with genetic testing for FKTN gene mutations.",
        "prognosis": "Progressive disease. Most patients lose mobility in childhood and require supportive care.",
        "reference": "Nelson Textbook of Pediatrics"
    },
    {
        "disease": "NFM 1 with OGIE",
        "definition": "Neurofibromatosis Type 1 (NF1) complicated by Optic Pathway Glioma and Intracranial Ectasis (OGIE) affecting visual and cerebral pathways.",
        "mri_findings": [
            "Thickening and elongation of the optic nerve",
            "T2/FLAIR hyperintensities in basal ganglia and brainstem (UBOs)",
            "Dural ectasia in skull base"
        ],
        "clinical_features": [
            "Visual field loss and reduced visual acuity",
            "Proptosis and skin café-au-lait spots",
            "Lisch nodules in iris"
        ],
        "diagnostic_protocol": "Brain and orbital MRI with contrast, detailed eye exams, and genetic analysis.",
        "prognosis": "Slowly progressive optic nerve expansion. Demands regular ophthalmological monitoring.",
        "reference": "Fitzpatrick's Dermatology and Adams & Victor's Principles of Neurology"
    },
    {
        "disease": "Tuberous Sclerosis",
        "definition": "A genetic multisystem disorder causing benign tumors (hamartomas) to grow in the brain, kidneys, heart, and skin.",
        "mri_findings": [
            "Cortical tubers (T2/FLAIR hyperintensities)",
            "Subependymal nodules (SEN) along lateral ventricles",
            "Subependymal giant cell astrocytomas (SEGA)"
        ],
        "clinical_features": [
            "Refractory infantile spasms and seizures",
            "Cognitive impairment or developmental delays",
            "Skin ash-leaf spots and facial angiofibromas"
        ],
        "diagnostic_protocol": "Brain MRI, renal ultrasound, skin exam, and TSC1/TSC2 genetic screening.",
        "prognosis": "Varies by seizure severity and tumor growths (e.g. SEGA hydrocephalus risks).",
        "reference": "Harrison's Principles of Internal Medicine"
    },
    {
        "disease": "Walker-Warburg Syndrome",
        "definition": "A severe congenital muscular dystrophy presenting with cobblestone lissencephaly, eye malformations, and profound developmental deficits.",
        "mri_findings": [
            "Cobblestone lissencephaly (type II)",
            "Severe ventriculomegaly and hydrocephalus",
            "Severe pontocerebellar hypoplasia"
        ],
        "clinical_features": [
            "Congenital blindness, cataracts, or retinal detachment",
            "Profound hypotonia, contractures, and developmental arrest",
            "Neonatal seizures"
        ],
        "diagnostic_protocol": "Fetal/Neonatal brain MRI, ophthalmologic exams, and POMT1/POMT2 genetic panels.",
        "prognosis": "Extremely poor, with most infants not surviving past the first year.",
        "reference": "Nelson Textbook of Pediatrics"
    },
    {
        "disease": "Acute Cerebellitis in HIV",
        "definition": "Acute inflammatory cerebellar syndrome caused by direct HIV infection of astrocytes/microglia or opportunistic pathogens.",
        "mri_findings": [
            "Bilateral cerebellar cortical T2/FLAIR hyperintensity",
            "Cerebellar swelling and effacement of cerebellar sulci",
            "Meningeal enhancement over cerebellum"
        ],
        "clinical_features": [
            "Acute onset of ataxia, unsteady gait, and dysmetria",
            "Headache, vomiting, and nystagmus",
            "Low CD4 cell count"
        ],
        "diagnostic_protocol": "Brain MRI, lumbar puncture for CSF virus/PCR panels, and HIV viral load.",
        "prognosis": "Improves with active antiretroviral therapy (ART) and treatment of underlying pathogens.",
        "reference": "Mandell's Principles and Practice of Infectious Diseases"
    },
    {
        "disease": "Acute Unilateral Cerebellitis in HIV",
        "definition": "A localized inflammatory cerebellar process restricted to a single hemisphere in HIV-positive patients, simulating a tumor or stroke.",
        "mri_findings": [
            "Unilateral T2/FLAIR hyperintensity in one cerebellar hemisphere",
            "Mass effect causing fourth ventricle compression",
            "Patchy cortical contrast enhancement"
        ],
        "clinical_features": [
            "Hemi-ataxia and ipsilateral coordination loss",
            "Unilateral dysmetria and intention tremor",
            "Acute headache and positional vertigo"
        ],
        "diagnostic_protocol": "Brain MRI with contrast and CSF analysis to rule out lymphoma, stroke, or bacterial abscess.",
        "prognosis": "Responsive to corticosteroid therapy and optimization of antiretroviral regimens.",
        "reference": "Mandell's Principles and Practice of Infectious Diseases"
    },
    {
        "disease": "Congenital Toxoplasmosis",
        "definition": "A fetal infection by Toxoplasma gondii transmitted from mother during pregnancy, causing severe neurological damage.",
        "mri_findings": [
            "Diffuse parenchymal and periventricular calcifications",
            "Hydrocephalus secondary to aqueduct stenosis",
            "Ring-enhancing focal necrotic brain lesions"
        ],
        "clinical_features": [
            "Classic triad: Chorioretinitis, hydrocephalus, and intracranial calcifications",
            "Seizures, mental retardation, and hepatosplenomegaly"
        ],
        "diagnostic_protocol": "Postnatal brain CT or MRI, ophthalmology review, and CSF/serum IgM serology.",
        "prognosis": "Improves with early antiparasitic therapy (pyrimethamine and sulfadiazine).",
        "reference": "Nelson Textbook of Pediatrics"
    },
    {
        "disease": "Japanese B Encephalitis or Epstein-Barr Encephalitis",
        "definition": "Severe viral brain infection causing inflammation and necrosis, selectively targeting deep grey matter nuclei.",
        "mri_findings": [
            "Bilateral symmetric T2/FLAIR hyperintensities in the thalami",
            "Substantia nigra, basal ganglia, and midbrain involvement",
            "DWI restricted diffusion in acute phase"
        ],
        "clinical_features": [
            "Sudden high fever, neck rigidity, and altered consciousness",
            "Parkinsonian symptoms (rigidity, resting tremor)",
            "Generalized seizures and coma"
        ],
        "diagnostic_protocol": "Brain MRI, CSF lymphocytic pleocytosis, and ELISA/PCR testing for JEV/EBV.",
        "prognosis": "High mortality rate (up to 30%). Long-term neurological deficits are common in survivors.",
        "reference": "Harrison's Principles of Internal Medicine"
    },
    {
        "disease": "Rasmussens Encephalitis",
        "definition": "A rare, chronic progressive inflammatory disorder characterized by unilateral brain hemisphere destruction and epilepsy.",
        "mri_findings": [
            "Progressive unilateral cerebral hemispheric atrophy",
            "FLAIR/T2 hyperintensity in cortical gray matter of affected side",
            "Head of caudate nucleus atrophy"
        ],
        "clinical_features": [
            "Intractable focal motor seizures (epilepsia partialis continua)",
            "Progressive hemiplegia and hemiparesis",
            "Cognitive decline and speech deterioration"
        ],
        "diagnostic_protocol": "Serial brain MRIs to document progressive unilateral volume loss, plus EEG.",
        "prognosis": "Progressive. Hemispherectomy is often required to control life-threatening seizures.",
        "reference": "Adams and Victor's Principles of Neurology"
    },
    {
        "disease": "Balloon Cell Cortical Dysplasia",
        "definition": "A focal malformation of cortical development (FCD Type IIb) containing abnormal giant balloon cells, causing intractable epilepsy.",
        "mri_findings": [
            "Grey-white matter junction blurring with cortical thickening",
            "T2/FLAIR transmantle sign (hyperintensity from cortex to ventricle)",
            "Localized abnormal gyration"
        ],
        "clinical_features": [
            "Drug-resistant focal motor or sensory seizures starting in childhood",
            "Developmental delay proportional to dysplasia size"
        ],
        "diagnostic_protocol": "High-resolution 3T epilepsy-protocol Brain MRI and video-EEG monitoring.",
        "prognosis": "Seizures are resistant to medication. Surgical resection of dysplasia provides high cure rates.",
        "reference": "Adams and Victor's Principles of Neurology"
    },
    {
        "disease": "Pachygyria with Cerebellar Hypoplasia",
        "definition": "A genetic migration disorder causing broad, flat gyri (pachygyria) in the cerebrum and underdevelopment of the cerebellum.",
        "mri_findings": [
            "Broad, thickened gyri with simplified sulcation (pachygyria)",
            "Hypoplasia of the cerebellar vermis and hemispheres",
            "Enlargement of ventricles and subarachnoid spaces"
        ],
        "clinical_features": [
            "Severe developmental delay and microcephaly",
            "Spastic diplegia or quadriplegia",
            "Infantile spasms or childhood epilepsy"
        ],
        "diagnostic_protocol": "Brain MRI showing migration defects, paired with whole-exome sequencing.",
        "prognosis": "Static neurological condition. Management is focused on supportive care.",
        "reference": "Nelson Textbook of Pediatrics"
    },
    {
        "disease": "Perisylvian Syndrome",
        "definition": "A developmental disorder characterized by bilateral polymicrogyria symmetrically clustered around the perisylvian fissures.",
        "mri_findings": [
            "Bilateral symmetric polymicrogyria lining deep perisylvian cortex",
            "Abnormal grey-white gray matter distribution",
            "Vertical orientation of sylvian fissures"
        ],
        "clinical_features": [
            "Congenital pseudobulbar palsy (dysphagia, dysarthria, drooling)",
            "Severe expressive language impairment",
            "Refractory epilepsy and spasticity"
        ],
        "diagnostic_protocol": "Brain MRI showing bilateral perisylvian polymicrogyria and clinical speech exams.",
        "prognosis": "Life-long speech and swallowing challenges. Seizures require multiple antiepileptic agents.",
        "reference": "Adams and Victor's Principles of Neurology"
    },
    {
        "disease": "Osmotic Demyelination Syndrome",
        "definition": "A non-inflammatory demyelinating syndrome primarily affecting the central pons, caused by rapid correction of chronic hyponatremia.",
        "mri_findings": [
            "Symmetric T2/FLAIR hyperintensity in the central basis pontis",
            "Sparing of the peripheral pontine tracts and corticospinal tracts",
            "Classic trident-shaped or pig-snout pontine lesion"
        ],
        "clinical_features": [
            "Acute spastic quadriparesis and pseudobulbar palsy",
            "Dysarthria, dysphagia, and locked-in syndrome in severe cases",
            "Fluctuating level of consciousness"
        ],
        "diagnostic_protocol": "Brain MRI with diffusion-weighted imaging (DWI) and monitoring of sodium levels.",
        "prognosis": "High rate of morbidity. Slow correction of hyponatremia is the crucial prevention method.",
        "reference": "Adams and Victor's Principles of Neurology"
    },
    {
        "disease": "Typical Adrenoleukodystrophy",
        "definition": "An X-linked peroxisomal disease leading to accumulation of very long-chain fatty acids (VLCFA), causing brain demyelination.",
        "mri_findings": [
            "Symmetric T2/FLAIR hyperintensity in parieto-occipital white matter",
            "Involvement of the splenium of the corpus callosum",
            "Rim of peripheral contrast enhancement representing active inflammation"
        ],
        "clinical_features": [
            "Rapid cognitive and behavioral decline in school-aged boys",
            "Vision loss, hearing loss, and spastic gait",
            "Skin hyperpigmentation (adrenal Addisonian crisis)"
        ],
        "diagnostic_protocol": "Brain MRI (Loes score evaluation), plasma VLCFA, and ABCD1 gene mutation analysis.",
        "prognosis": "Rapid progression to vegetative state if untreated. Early stem-cell transplantation is curative.",
        "reference": "Harrison's Principles of Internal Medicine"
    },
    {
        "disease": "Optic Glioma",
        "definition": "A low-grade pilocytic astrocytoma arising from the optic pathways, strongly associated with Neurofibromatosis Type 1 (NF1).",
        "mri_findings": [
            "Fusiform enlargement and kinking of the optic nerve",
            "T2/FLAIR hyperintense expansion of optic pathways",
            "Homogeneous contrast enhancement"
        ],
        "clinical_features": [
            "Slow, painless visual acuity loss",
            "Proptosis (bulging eye) and optic disc edema",
            "Strabismus or optic atrophy"
        ],
        "diagnostic_protocol": "Brain/orbit MRI with contrast, visual fields, and NF1 clinical criteria evaluation.",
        "prognosis": "Generally slow-growing or benign course, but requires visual preservation therapy if progressive.",
        "reference": "Harrison's Principles of Internal Medicine"
    },
    {
        "disease": "Plexiform Neurofibroma with Sphenoid Wing Dysplasia",
        "definition": "A tumor of peripheral nerves (plexiform neurofibroma) associated with underdevelopment of the sphenoid bone, diagnostic of NF1.",
        "mri_findings": [
            "Infiltrating orbital/temporal soft-tissue mass ('bag of worms' sign)",
            "Absence or hypoplasia of the greater sphenoid wing",
            "Temporal lobe herniation into orbit"
        ],
        "clinical_features": [
            "Pulsatile exophthalmos (orbit pulses with heartbeat)",
            "Facial and orbital asymmetry",
            "Palpable facial soft tissue mass"
        ],
        "diagnostic_protocol": "Skull CT for bony dysplasia, combined with orbit MRI for tumor margins.",
        "prognosis": "Benign but progressive local invasion. Surgery is complex due to high vascularity.",
        "reference": "Fitzpatrick's Dermatology"
    },
    {
        "disease": "Hepatocellular Carcinoma (HCC) and Dysplastic Nodule",
        "definition": "A malignant primary liver cancer (HCC) arising from pre-existing high-grade dysplastic nodules in cirrhotic livers.",
        "mri_findings": [
            "Arterial phase hyperenhancement (APHE)",
            "Portal venous/delayed phase washout",
            "Enhancing capsule or pseudocapsule around lesion"
        ],
        "clinical_features": [
            "Right upper quadrant abdominal pain and hepatomegaly",
            "Unexplained weight loss and jaundice",
            "Elevated serum Alpha-Fetoprotein (AFP)"
        ],
        "diagnostic_protocol": "Multiphase liver MRI using LI-RADS classification criteria, and AFP checks.",
        "prognosis": "Depends on tumor size, vascular invasion, and degree of liver cirrhosis.",
        "reference": "Sleisenger and Fordtran's Gastrointestinal and Liver Disease"
    },
    {
        "disease": "Hepatocellular_Carcinoma",
        "definition": "Primary hepatocellular malignancy associated with chronic HBV/HCV hepatitis, showing rapid hypervascular invasion.",
        "mri_findings": [
            "Intense wash-in during hepatic arterial phase",
            "Rapid wash-out in portal venous and delayed phases",
            "Restricted diffusion on DWI with portal vein tumor thrombus"
        ],
        "clinical_features": [
            "Jaundice, ascites, and variceal bleeding",
            "Weight loss, cachexia, and palpable liver mass"
        ],
        "diagnostic_protocol": "Dynamic contrast CT or MRI, AFP testing, and core biopsy in non-cirrhotic livers.",
        "prognosis": "Poor prognosis if portal vein thrombosis is present. Early liver resection/transplant offers cure.",
        "reference": "Sherlock's Diseases of the Liver and Biliary System"
    },
    {
        "disease": "Inferior Vena Cava (IVC) Leiomyosarcoma",
        "definition": "A rare malignant retroperitoneal tumor originating from smooth muscle cells of the inferior vena cava wall.",
        "mri_findings": [
            "Large intraluminal IVC mass causing occlusion or distension",
            "Heterogeneous T2 hyperintensity with invasion of adjacent retroperitoneum",
            "Irregular post-contrast enhancement"
        ],
        "clinical_features": [
            "Bilateral lower extremity edema due to venous blockage",
            "Abdominal pain, weight loss, or Budd-Chiari symptoms"
        ],
        "diagnostic_protocol": "CT or MR venography to evaluate IVC lumen, and tumor core biopsy.",
        "prognosis": "Poor. Requires complete surgical resection with negative margins.",
        "reference": "Sleisenger and Fordtran's Gastrointestinal and Liver Disease"
    },
    {
        "disease": "Carolis Disease",
        "definition": "A rare congenital biliary disorder characterized by segmental saccular dilatation of intrahepatic bile ducts.",
        "mri_findings": [
            "Segmental dilatation of intrahepatic ducts sparing extrahepatic ducts",
            "Central dot sign (enhancing portal branch inside dilated duct)",
            "Intraductal biliary calculi"
        ],
        "clinical_features": [
            "Recurrent bacterial cholangitis (fever, RUQ pain, chills)",
            "Biliary colic, jaundice, and risk of portal hypertension"
        ],
        "diagnostic_protocol": "MRCP (Magnetic Resonance Cholangiopancreatography) and ultrasound.",
        "prognosis": "High risk of cholangitis and development of cholangiocarcinoma. Curable by segmentectomy or liver transplant.",
        "reference": "Sherlock's Diseases of the Liver and Biliary System"
    },
    {
        "disease": "Cholangiocarcinoma",
        "definition": "A highly aggressive adenocarcinoma arising from intrahepatic or extrahepatic biliary tract epithelial cells.",
        "mri_findings": [
            "Intraductal mass causing abrupt biliary obstruction and proximal dilation",
            "Centripetal progressive contrast enhancement on delayed phases",
            "Capsular retraction of liver tissue overlying the mass"
        ],
        "clinical_features": [
            "Progressive painless obstructive jaundice, dark urine, pale stools",
            "Pruritus, fatigue, weight loss, and elevated CA19-9"
        ],
        "diagnostic_protocol": "MRCP, ERCP with brushing cytology, and CA19-9 tumor marker checks.",
        "prognosis": "Poor due to aggressive local invasion. Resection is curative only in early stages.",
        "reference": "Sleisenger and Fordtran's Gastrointestinal and Liver Disease"
    }
]

def seed_vector_database(vector_db):
    """Indexes the structured clinical profiles and logs."""
    textbook_docs = [json.dumps(d) for d in DISEASE_DB]
    textbook_metadata = [{"source": d.get("reference")} for d in DISEASE_DB]
    
    agent_logs = [
        "Patient Case neuro_908: Male infant presenting with hypotonia. RL agent selected: CLAHE and Gaussian Blur (Q-values: clahe: 2.15, blur: 1.05). Broad classification: genetic brain malformations. Specialist classified Fukuyama Muscular Dystrophy. Conf: 91.5%. Log status: Confirmed.",
        "Patient Case liver_102: Female, age 56, history of Hepatitis B. RL agent selected: Sharpening (Q-values: sharpen: 4.89, none: -0.62). Broad category: Malignant. Specialist classified Hepatocellular Carcinoma (HCC). Conf: 94.2%. Log status: Flagged for Biopsy.",
        "Patient Case liver_045: Male, age 32, abdominal pain. RL agent skipped preprocessing (action: none). Broad category: Benign. Specialist classified Liver Hemangioma. Conf: 95.8%. Log status: Confirmed Benign.",
        "Patient Case neuro_002: Male, age 41, HIV positive. RL agent selected: Gaussian Blur. Broad category: infectious. Specialist classified Acute Unilateral Cerebellitis in HIV. Conf: 89.1%. Log status: Urgent Clinical Alert.",
        "Patient Case liver_773: Female, age 29, elevated liver enzymes. RL agent selected: CLAHE. Broad category: Ductual. Specialist classified Carolis Disease. Conf: 93.0%. Log status: Confirmed."
    ]
    agent_metadata = [
        {"patient_id": "neuro_908", "domain": "brain", "agent_reward": 0.915},
        {"patient_id": "liver_102", "domain": "liver", "agent_reward": 0.942},
        {"patient_id": "liver_045", "domain": "liver", "agent_reward": 0.958},
        {"patient_id": "neuro_002", "domain": "brain", "agent_reward": 0.891},
        {"patient_id": "liver_773", "domain": "liver", "agent_reward": 0.930}
    ]
    
    vector_db.add_documents("medical_base", textbook_docs, textbook_metadata)
    vector_db.add_documents("agent_result_logs", agent_logs, agent_metadata)


# ==============================================================================
# 9. INTEGRATED RUNNER PIPELINE FOR GRADIO
# ==============================================================================
def find_file(filename):
    if os.path.exists(filename):
        return filename
    # Search /kaggle/input
    for root, dirs, files in os.walk('/kaggle/input'):
        if filename in files:
            return os.path.join(root, filename)
    # Search recursively in the current workspace
    for root, dirs, files in os.walk('.'):
        if filename in files:
            return os.path.join(root, filename)
    return None

# Resolve model assets paths dynamically
qtable_file = find_file('rl_agent.json')
weights_file = find_file('rlagent.pth')
if not weights_file:
    weights_file = find_file('rlagent.wt')

my_model_paths = {
    'generalist':    weights_file if weights_file else 'rlagent.wt',
    'genetic':       find_file('brain_genetic_custom_lite_..pth') or find_file('brain_genetic_custom_lite_.pth') or 'brain_genetic_custom_lite_.pth',
    'infectious':    find_file('infectious_custom_specialist.pth') or 'infectious_custom_specialist.pth',
    'malformations': find_file('developmental_malformations_lite_92plus.pth') or 'developmental_malformations_lite_92plus.pth',
    'metabolic':     find_file('metabolic_custom_specialist.pth') or 'metabolic_custom_specialist.pth',
    'neoplastic':    find_file('neoplastic_custom_specialist.pth') or 'neoplastic_custom_specialist.pth',
    'malignant':     find_file('liver_custom_malignant_classifier.pth') or 'liver_custom_malignant_classifier.pth',
    'ductal':        find_file('liver_custom_Ductual_micro_final.pth') or 'liver_custom_Ductual_micro_final.pth'
}

Q_table = {}
if qtable_file and os.path.exists(qtable_file):
    try:
        with open(qtable_file, 'r') as f:
            data = json.load(f)
            Q_table = data.get("Q", {})
    except Exception as e:
        print(f"Error loading Q-table: {e}")

gen_classes = sorted([
    'Healthy', 'genetic', 'vascular', 'Benign',
    'Retinoblastoma with Intracranial Spread Along Cranial Nerve',
    'Ductual', 'Magnetic Resonance (MR) Brain',
    'developmental brain malformations', 'infectious',
    'tumours or neoplastic', 'metabolic', 'Malignant'
])

# Initialize global singletons
hub = MedicalAIHub(my_model_paths, gen_classes)
vector_db = LightVectorDB()
seed_vector_database(vector_db)
rag_pipeline = MedicalRAGPipeline(vector_db)
db_client, db_status = get_mongodb_connection()

def analyze_scan(input_file, patient_name, patient_id):
    global db_client, db_status
    if input_file is None:
        yield None, None, "⚠️ Please upload a scan slice first.", "N/A", "N/A", "0.0%", "N/A (S3 Offline)", "", "⚠️ Inference aborted: No input image."
        return
        
    filepath = input_file.name if hasattr(input_file, 'name') else str(input_file)
    
    # Pre-generate unique MongoDB document ID locally
    from bson.objectid import ObjectId
    doc_id = ObjectId()
    doc_id_str = str(doc_id)
    
    # Re-evaluate connection if it was offline
    if db_client is None:
        db_client, db_status = get_mongodb_connection()

    # Load original scan array (DICOM or standard image)
    try:
        input_img = load_medical_image(filepath)
    except Exception as e:
        yield None, None, f"❌ Image Loading Error: {e}", "N/A", "N/A", "0.0%", "N/A (S3 Offline)", "", f"❌ Image Loading Error: {e}"
        return

    # 1. Run classifier-guided RL Preprocessor (< 0.4s)
    try:
        processed_img, actions, rl_logs = process_image_with_agent_and_hub(input_img, Q_table, hub)
    except Exception as e:
        yield input_img, input_img, rl_logs, "N/A", "N/A", "0.0%", "N/A (S3 Offline)", "", f"❌ Preprocessing Error: {e}"
        return

    # 2. Run Classification Hub (< 0.2s)
    try:
        img_tensor = hub._tensor_from_np(processed_img)
        with torch.no_grad():
            gen_probs = torch.nn.functional.softmax(hub.gen(img_tensor), dim=1)
            gen_conf, gen_idx = torch.max(gen_probs, 1)
            generalist_pred = hub.gen_classes[gen_idx.item()]
            
        label, confidence = hub.diagnose_array(processed_img)
    except Exception as e:
        yield input_img, processed_img, rl_logs, "N/A", "N/A", "0.0%", "N/A (S3 Offline)", "", f"❌ Classification Error: {e}"
        return

    # 3. STAGE 1 INSTANT UI RESPONSE (< 1.0s Total Time)
    pending_report = """<div style='padding: 16px; background-color: #EBF8FF; border: 1px solid #BEE3F8; border-radius: 8px; color: #2B6CB0; font-weight: 600;'>
🤖 <b>Gemini AI is generating your detailed clinical justification report...</b><br/>
<span style='font-size: 12px; color: #4A5568;'>Your scan preprocessed images, RL metrics, and classification scores are displayed above. The full report will appear below shortly as you review your patient data.</span>
</div>"""

    yield input_img, processed_img, rl_logs, generalist_pred, label, f"{confidence*100:.1f}%", "Uploading in background...", doc_id_str, pending_report

    # 4. Handle threshold guardrail logic
    if "Low Confidence" in label:
        rag_report = f"""### 🚨 [ALERT] CLASSIFICATION CONFIDENCE TOO LOW
The Generalist classified the scan category with low confidence (**{confidence*100:.2f}%**, below the required **90%** threshold).

*   **Bypassed Path:** Biliary / Specialist neural diagnostic routing was aborted for safety.
*   **Action Required:** This case has been flagged and pushed to MongoDB Atlas. Scan slice requires manual clinical review by a human Radiologist.
"""
        bg_executor.submit(
            bg_save_task,
            filepath,
            actions,
            confidence,
            label,
            patient_name,
            patient_id,
            doc_id
        )
        yield input_img, processed_img, rl_logs, generalist_pred, label, f"{confidence*100:.1f}%", "Saved to offline queue ✓", doc_id_str, rag_report
        return

    # 5. STAGE 2: GENERATE FULL GEMINI LLM REPORT IN BACKGROUND
    try:
        execution_context = f"Current Run Preprocessing: The RL Preprocessing Agent took actions: {actions} (logs: {rl_logs})."
        query = f"Give me a clinical breakdown for a scan diagnosed with {label}. {execution_context}"
        rag_report = rag_pipeline.run_query(query)
    except Exception as e:
        rag_report = f"⚠️ RAG Pipeline Error: {e}"

    # Submit S3 upload and MongoDB logging to background thread pool
    bg_executor.submit(
        bg_save_task,
        filepath,
        actions,
        confidence,
        label,
        patient_name,
        patient_id,
        doc_id
    )
    
    # 6. STAGE 2 UI RESPONSE: POPULATE FULL CLINICAL REPORT
    yield input_img, processed_img, rl_logs, generalist_pred, label, f"{confidence*100:.1f}%", "Saved to MongoDB & S3 ✓", doc_id_str, rag_report

def reset_workspace():
    return None, "", "", None, None, "", "", "", "", "", None, "", "", ""

custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate"
).set(
    body_background_fill="#EEF2F9",
    body_text_color="#0F172A",
    body_text_color_subdued="#64748B",
    background_fill_primary="#FFFFFF",
    background_fill_secondary="#F8FAFC",
    block_background_fill="#FFFFFF",
    block_border_color="#E2E8F0",
    block_title_text_color="#0F172A",
    input_background_fill="#FFFFFF",
    input_border_color="#CBD5E1",
    button_primary_background_fill="#2563EB",
    button_primary_background_fill_hover="#1D4ED8",
    block_title_text_weight="bold",
    block_border_width="1px"
)

import inspect
theme_in_launch = 'theme' in inspect.signature(gr.Blocks.launch).parameters

custom_css = """
/* ==============================================================================
   FORCE SETCONNECT LIGHT THEME & FIXED LAYOUT ANCHORING
   ============================================================================== */

/* Reset Gradio Dark Theme Variables */
:root, .dark, body.dark, [data-testid="app-container"].dark, [data-theme="dark"] {
    --body-background-fill: #EEF2F9 !important;
    --body-text-color: #0F172A !important;
    --body-text-color-subdued: #64748B !important;
    --background-fill-primary: #FFFFFF !important;
    --background-fill-secondary: #F8FAFC !important;
    --block-background-fill: #FFFFFF !important;
    --block-border-color: #E2E8F0 !important;
    --block-title-text-color: #0F172A !important;
    --block-label-background-fill: #F1F5F9 !important;
    --block-label-text-color: #0F172A !important;
    --input-background-fill: #FFFFFF !important;
    --input-border-color: #CBD5E1 !important;
    --button-primary-background-fill: #2563EB !important;
    --button-primary-text-color: #FFFFFF !important;
    --button-secondary-background-fill: #F1F5F9 !important;
    --button-secondary-text-color: #334155 !important;
}

/* Global Page Background & Grid Texture */
html, body, .gradio-container, .main, #root, .app, .wrap, footer, .contain, [data-testid="app-container"], .dark {
    background-color: #EEF2F9 !important;
    background-image: radial-gradient(#CBD5E1 0.75px, transparent 0.75px), radial-gradient(#CBD5E1 0.75px, #EEF2F9 0.75px) !important;
    background-size: 24px 24px !important;
    background-position: 0 0, 12px 12px !important;
    font-family: -apple-system, BlinkMacSystemFont, "Plus Jakarta Sans", "Inter", "Segoe UI", Roboto, sans-serif !important;
    color: #0F172A !important;
}

.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Fix outer layout while preserving horizontal flexbox for inner rows */
.gradio-container > .main > .wrap > .row,
.gradio-container > .contain > .row {
    display: block !important;
    width: 100% !important;
}

/* Restore Horizontal Flexbox for Tab Row & Card Rows */
.row, .gr-row, [data-testid="row"] {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    gap: 12px !important;
    width: 100% !important;
}

/* 1. FIXED LEFT SIDEBAR - Hard-locked 260px Width, Z-Index 10000 */
.sidebar-container,
[data-testid="column"].sidebar-container {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 260px !important;
    min-width: 260px !important;
    max-width: 260px !important;
    flex: 0 0 260px !important;
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important; /* NO SCROLLBAR */
    z-index: 10000 !important;
    background: #0B0F19 !important;
    border-right: 1px solid #1E293B !important;
    padding: 14px 12px !important;
    box-sizing: border-box !important;
    box-shadow: 4px 0 20px rgba(0,0,0,0.15) !important;
}

.sidebar-container::-webkit-scrollbar,
.sidebar-container *::-webkit-scrollbar {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
}

.sidebar-container .block, 
.sidebar-container .panel, 
.sidebar-container .box, 
.sidebar-container .gr-box,
.sidebar-container .gr-block {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

.sidebar-nav-btn {
    text-align: left !important;
    justify-content: flex-start !important;
    background-color: transparent !important;
    border: none !important;
    color: #94A3B8 !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    padding: 10px 14px !important;
    width: 100% !important;
    border-radius: 12px !important;
    cursor: pointer;
    margin-bottom: 4px;
    box-shadow: none !important;
}

.sidebar-nav-btn:hover {
    background-color: #161E2E !important;
    color: #F8FAFC !important;
}

/* Active Sidebar Button - Bright Electric Blue Pill */
.active-nav-btn {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4) !important;
    border-radius: 12px !important;
}

/* 2. DEDICATED SCROLLABLE MAIN CONTENT VIEWPORT (Starts AT 165px, Height calc(100vh - 165px)) */
.main-content-container,
[data-testid="column"].main-content-container {
    position: fixed !important;
    top: 165px !important; /* Starts EXACTLY below the 165px fixed top header bar */
    left: 260px !important;
    right: 0 !important;
    bottom: 0 !important;
    width: calc(100% - 260px) !important;
    max-width: calc(100% - 260px) !important;
    height: calc(100vh - 165px) !important;
    max-height: calc(100vh - 165px) !important;
    overflow-y: auto !important; /* Dedicated independent scrollbar ONLY inside this window */
    overflow-x: hidden !important;
    padding-top: 20px !important;
    padding-left: 48px !important;
    padding-right: 36px !important;
    padding-bottom: 50px !important;
    background-color: #EEF2F9 !important;
    box-sizing: border-box !important;
    z-index: 1 !important;
}

/* 3. SOLID OPAQUE DEDICATED FIXED TOP HEADER BLOCK (Z-Index 9999) */
.top-sticky-header-container,
[data-testid="column"].top-sticky-header-container {
    position: fixed !important;
    top: 0 !important;
    left: 260px !important;
    right: 0 !important;
    width: calc(100% - 260px) !important;
    height: 165px !important; /* Fixed 165px height for title + status + 4 card boxes */
    box-sizing: border-box !important;
    z-index: 9999 !important; /* STRICT Z-INDEX 9999 */
    background-color: #EEF2F9 !important; /* SOLID OPAQUE CREAM/LIGHT BLUE BACKGROUND */
    background-image: radial-gradient(#CBD5E1 0.75px, transparent 0.75px), radial-gradient(#CBD5E1 0.75px, #EEF2F9 0.75px) !important;
    background-size: 24px 24px !important;
    background-position: 0 0, 12px 12px !important;
    padding: 14px 36px 12px 48px !important;
    border-bottom: 2.5px solid #CBD5E1 !important;
    box-shadow: 0 8px 25px rgba(15, 23, 42, 0.08) !important;
}

.top-sticky-header-container .block,
.top-sticky-header-container .gr-block,
.top-sticky-header-container .panel,
.top-sticky-header-container .box,
.top-sticky-header-container .card-container {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

.top-sticky-header-container .row,
.top-sticky-header-container .gr-row,
.top-sticky-header-container [data-testid="row"] {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    gap: 12px !important;
    width: 100% !important;
    margin-top: 10px !important;
}

.top-tab-card {
    flex: 1 1 0% !important;
    min-width: 0 !important;
    background: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 14px !important;
    padding: 10px 14px !important;
    cursor: pointer;
    text-align: center !important;
    font-size: 12.5px !important;
    font-weight: 600 !important;
    color: #334155 !important;
    transition: all 0.2s ease;
}

.top-tab-card:hover {
    border-color: #93C5FD !important;
    background: #F8FAFC !important;
}

.active-top-card {
    border: 2px solid #3B82F6 !important;
    background: #EFF6FF !important;
    color: #1D4ED8 !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15) !important;
}

/* 4. CONTENT CARDS - Crisp Pure White with Slate Navy Text (Only target explicit card-container) */
.card-container {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 20px !important;
    padding: 24px !important;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.03) !important;
    margin-bottom: 20px !important;
    color: #0F172A !important;
}

/* Ensure empty or wrapper blocks remain 100% transparent on #EEF2F9 canvas */
.main-content-container > .block:not(.card-container),
.main-content-container > .panel:not(.card-container),
.main-content-container > div:empty,
.top-sticky-header-container,
.top-sticky-header-container .block,
.top-sticky-header-container .panel,
.top-sticky-header-container .box,
.top-sticky-header-container [data-testid="column"],
.top-sticky-header-container [data-testid="block"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    min-height: 0 !important;
}

.card-container h1, .card-container h2, .card-container h3, .card-container h4, .card-container p, .card-container span, .card-container label {
    color: #0F172A !important;
}

/* Upload Dropzone Container */
[data-testid="file-upload"], .upload-container, div[data-testid="image"] {
    background-color: #F8FAFC !important;
    border: 2px dashed #93C5FD !important;
    border-radius: 16px !important;
    padding: 20px !important;
}

/* Primary Blue Action Buttons */
button.primary-btn, button.variant-primary, .primary-btn {
    background: #2563EB !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
    padding: 10px 20px !important;
    cursor: pointer;
}

button.primary-btn:hover, button.variant-primary:hover {
    background: #1D4ED8 !important;
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35) !important;
}

button.secondary-btn, button.variant-secondary {
    background: #F1F5F9 !important;
    color: #334155 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    border: 1px solid #CBD5E1 !important;
}

/* Input Fields & Textareas */
input, textarea, select {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    color: #0F172A !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
}

/* Code & Pre Blocks */
code, pre, .prose code {
    background-color: #F8FAFC !important;
    color: #0F172A !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
}

th {
    border-bottom: 2px solid #E2E8F0 !important;
    font-weight: 700 !important;
    color: #475569 !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

td {
    border-bottom: 1px solid #F1F5F9 !important;
    color: #0F172A !important;
}

.center-upload-card {
    max-width: 650px !important;
    margin: 0 auto 24px auto !important;
    text-align: center !important;
}

.action-btn-row {
    margin-top: 16px !important;
}
"""

import base64

def get_logo_html():
    logo_file = (
        find_file('setconnect_logo.png.jpeg') or 
        find_file('setconnect_logo.jpeg') or 
        find_file('setconnect_logo.jpg') or 
        find_file('setconnect_logo.png') or 
        find_file('logo.png') or 
        find_file('logo.jpeg') or 
        find_file('logo.jpg') or
        find_file('setconnect.png')
    )
    if logo_file and os.path.exists(logo_file):
        try:
            mime = "image/jpeg" if logo_file.lower().endswith(('.jpeg', '.jpg')) else "image/png"
            with open(logo_file, "rb") as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')
            return f'<img src="data:{mime};base64,{encoded}" style="height: 32px; width: auto; object-fit: contain; border-radius: 6px;" alt="SetCONNECT Logo" />'
        except Exception as e:
            print(f"⚠️ Error reading logo image file: {e}")
    return '''<div style="width: 34px; height: 34px; background: linear-gradient(135deg, #00A8E8 0%, #2563EB 100%); border-radius: 10px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(0, 168, 232, 0.4);">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            <polyline points="9 12 11 14 15 10"/>
        </svg>
    </div>'''

with gr.Blocks(title="SetConnect Diagnostic Workstation", theme=custom_theme) as demo:
    # Inject Custom CSS directly into DOM to bypass Gradio theme caching
    gr.HTML(f"<style>{custom_css}</style>")
    current_doc_id = gr.State("")

    with gr.Row():
        # Sidebar menu (Dark Navy - Fixed Pinned with NO Scrollbar)
        with gr.Column(scale=1, elem_classes="sidebar-container"):
            gr.HTML(f"""
                <div style="display: flex; align-items: center; justify-content: space-between; padding-bottom: 12px; margin-bottom: 12px; border-bottom: 1px solid #1E293B;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        {get_logo_html()}
                        <div>
                            <div style="font-size: 17px; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em; line-height: 1.1;">Set<span style="color: #38BDF8;">CONNECT</span></div>
                            <div style="font-size: 10px; font-weight: 500; color: #64748B;">Medical AI Diagnostic</div>
                        </div>
                    </div>
                </div>
                
                <div style="font-size: 10px; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; padding-left: 4px;">MAIN MENU</div>
            """)
            
            nav_overview_btn = gr.Button("🎯  Overview", elem_classes="sidebar-nav-btn active-nav-btn")
            nav_load_data_btn = gr.Button("📥  Load Data", elem_classes="sidebar-nav-btn")
            nav_clinical_review_btn = gr.Button("📋  Clinical Review", elem_classes="sidebar-nav-btn")
            nav_clinical_report_btn = gr.Button("📄  Clinical Report", elem_classes="sidebar-nav-btn")
            
            gr.HTML("""
                <div style="margin-top: 14px;">
                    <!-- RECENT PATIENT SESSIONS (Matches Reference Image 1 & 2) -->
                    <div style="font-size: 10px; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; padding-left: 4px;">RECENT PATIENT SESSIONS</div>
                    
                    <div style="display: flex; flex-direction: column; gap: 6px; margin-bottom: 16px;">
                        <div style="display: flex; align-items: center; justify-content: space-between; padding: 7px 10px; background: #161E2E; border: 1px solid #1E293B; border-radius: 8px; color: #94A3B8; font-size: 11.5px; font-weight: 500;">
                            <div style="display: flex; align-items: center; gap: 6px; overflow: hidden;">
                                <span style="color: #38BDF8; font-size: 11px;">💬</span>
                                <span style="color: #F8FAFC; font-weight: 600;">neuro_908 (Emma)</span>
                            </div>
                            <div style="display: flex; gap: 4px; flex-shrink: 0;">
                                <span style="background: #1E293B; width: 20px; height: 20px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 9px; cursor: pointer; color: #94A3B8;">✏️</span>
                                <span style="background: #1E293B; width: 20px; height: 20px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 9px; cursor: pointer; color: #94A3B8;">🗑️</span>
                            </div>
                        </div>
                        
                        <div style="display: flex; align-items: center; justify-content: space-between; padding: 7px 10px; background: #161E2E; border: 1px solid #1E293B; border-radius: 8px; color: #94A3B8; font-size: 11.5px; font-weight: 500;">
                            <div style="display: flex; align-items: center; gap: 6px; overflow: hidden;">
                                <span style="color: #38BDF8; font-size: 11px;">💬</span>
                                <span style="color: #F8FAFC; font-weight: 600;">liver_102 (Liam)</span>
                            </div>
                            <div style="display: flex; gap: 4px; flex-shrink: 0;">
                                <span style="background: #1E293B; width: 20px; height: 20px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 9px; cursor: pointer; color: #94A3B8;">✏️</span>
                                <span style="background: #1E293B; width: 20px; height: 20px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 9px; cursor: pointer; color: #94A3B8;">🗑️</span>
                            </div>
                        </div>
                    </div>

                    <!-- QUICK ACTIONS (Matches Reference Image 1 & 2) -->
                    <div style="font-size: 10px; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; padding-left: 4px;">QUICK ACTIONS</div>
                    <div style="display: flex; flex-direction: column; gap: 6px; margin-bottom: 18px;">
                        <div style="padding: 8px 10px; background: #161E2E; border: 1px solid #1E293B; border-radius: 8px; color: #94A3B8; font-size: 11.5px; font-weight: 500; display: flex; align-items: center; gap: 8px; cursor: pointer;">
                            <span style="font-size: 12px;">📥</span> <span style="color: #E2E8F0; font-weight: 600;">Ingest Scan</span>
                        </div>
                        <div style="padding: 8px 10px; background: #161E2E; border: 1px solid #1E293B; border-radius: 8px; color: #94A3B8; font-size: 11.5px; font-weight: 500; display: flex; align-items: center; gap: 8px; cursor: pointer;">
                            <span style="font-size: 12px;">💬</span> <span style="color: #E2E8F0; font-weight: 600;">Ask Clinical AI</span>
                        </div>
                    </div>

                    <!-- USER ADMIN FOOTER CARD (Matches Reference Image 1 & 2 Footer) -->
                    <div style="padding-top: 12px; border-top: 1px solid #1E293B; display: flex; align-items: center; justify-content: space-between;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <div style="width: 30px; height: 30px; background: linear-gradient(135deg, #2563EB, #1D4ED8); border-radius: 50%; color: white; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 11px; box-shadow: 0 2px 8px rgba(37,99,235,0.4);">MD</div>
                            <div>
                                <div style="font-size: 11.5px; font-weight: 700; color: #FFFFFF;">MD Admin</div>
                                <div style="font-size: 9.5px; color: #64748B;">Radiologist</div>
                            </div>
                        </div>
                        <span style="color: #64748B; font-size: 14px; font-weight: bold;">›</span>
                    </div>
                </div>
            """)
            
        # Main content area (Scrollable container offset by sidebar width)
        with gr.Column(scale=4, elem_classes="main-content-container"):
            # DEDICATED FIXED TOP HEADER CONTAINER (Contains Title + Icons + 4 Horizontal Card Boxes)
            with gr.Column(elem_classes="top-sticky-header-container"):
                gr.HTML("""
                    <div style="display: flex; align-items: center; justify-content: space-between; width: 100%; margin-bottom: 6px;">
                        <div>
                            <div style="font-size: 26px; font-weight: 800; font-style: italic; color: #0F172A; letter-spacing: -0.03em; white-space: nowrap;">Set<span style="color: #0F172A;">CONNECT</span></div>
                            <div style="font-size: 11.5px; font-weight: 600; color: #64748B; margin-top: 1px; white-space: nowrap;">Medical AI Diagnostic • Scan Intelligence • Hierarchical AI</div>
                        </div>
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <div style="background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 9999px; padding: 5px 12px; font-size: 11.5px; font-weight: 700; color: #334155; display: flex; align-items: center; gap: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                                <span style="width: 7px; height: 7px; background-color: #22C55E; border-radius: 50%; display: inline-block;"></span>
                                API Online
                            </div>
                            <div style="width: 34px; height: 34px; background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #64748B; font-size: 13px; cursor: pointer; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">🌙</div>
                            <div style="width: 34px; height: 34px; background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #64748B; font-size: 13px; cursor: pointer; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">🔔</div>
                            <div style="width: 34px; height: 34px; background: #2563EB; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #FFFFFF; font-size: 12px; font-weight: 700; box-shadow: 0 2px 6px rgba(37,99,235,0.3);">MD</div>
                        </div>
                    </div>
                """)

                # 4 Horizontal Card Boxes (Fixed inside top container - NEVER scroll out of view)
                with gr.Row():
                    top_tab_overview_btn = gr.Button("🎯  Overview\nDashboard", elem_classes="top-tab-card active-top-card")
                    top_tab_load_data_btn = gr.Button("📥  Load Data\nIngest & analyze", elem_classes="top-tab-card")
                    top_tab_clinical_review_btn = gr.Button("📋  Clinical Review\nValidation & notes", elem_classes="top-tab-card")
                    top_tab_clinical_report_btn = gr.Button("📄  Clinical Report\nDiagnostic report", elem_classes="top-tab-card")

            # --- TABS / SECTIONS ---
            
            # SECTION 1: OVERVIEW & DASHBOARD
            with gr.Column(visible=True) as overview_section:
                with gr.Column(elem_classes="card-container"):
                    gr.HTML("""
                        <h2 style="margin-top: 0; color: #0F172A; font-weight: 800; font-size: 20px;">📊 Diagnostics System Overview</h2>
                        <p style="color: #64748B; font-size: 14px; margin-top: 4px;">Real-time diagnostics tracking, system performance metrics, and clinical pipelines.</p>
                        
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-top: 20px; margin-bottom: 10px;">
                            <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 14px; padding: 18px;">
                                <div style="font-size: 11px; font-weight: 700; color: #1D4ED8; text-transform: uppercase;">Active AI Networks</div>
                                <div style="font-size: 30px; font-weight: 800; color: #1E40AF; margin-top: 6px;">8 Models</div>
                                <div style="font-size: 12px; color: #3B82F6; margin-top: 4px;">1 Generalist + 7 Specialists</div>
                            </div>
                            <div style="background: #F5F3FF; border: 1px solid #DDD6FE; border-radius: 14px; padding: 18px;">
                                <div style="font-size: 11px; font-weight: 700; color: #6D28D9; text-transform: uppercase;">Knowledge Bases</div>
                                <div style="font-size: 30px; font-weight: 800; color: #5B21B6; margin-top: 6px;">2 databases</div>
                                <div style="font-size: 12px; color: #7C3AED; margin-top: 4px;">Medical texts + Clinical logs</div>
                            </div>
                            <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 14px; padding: 18px;">
                                <div style="font-size: 11px; font-weight: 700; color: #15803D; text-transform: uppercase;">Pipeline Latency</div>
                                <div style="font-size: 30px; font-weight: 800; color: #166534; margin-top: 6px;">&lt; 1.5s</div>
                                <div style="font-size: 12px; color: #22C55E; margin-top: 4px;">Inference & Preprocessing</div>
                            </div>
                        </div>
                    """)
                    
                with gr.Column(elem_classes="card-container"):
                    gr.HTML("""
                        <h3 style="margin-top: 0; color: #0F172A; font-size: 16px; font-weight: 700; margin-bottom: 16px;">🔄 AI Agent Diagnostic Flow</h3>
                        <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px; background-color: #F8FAFC; padding: 20px; border-radius: 14px; border: 1px solid #E2E8F0;">
                            <div style="flex: 1; min-width: 130px; text-align: center; padding: 12px; background: white; border-radius: 10px; border: 1px solid #CBD5E1; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                                <div style="font-weight: 700; color: #0F172A; font-size: 13px;">1. Scan Ingestion</div>
                                <div style="font-size: 11px; color: #64748B; margin-top: 4px;">DICOM / Image Upload</div>
                            </div>
                            <div style="color: #94A3B8; font-size: 18px; font-weight: bold;">➔</div>
                            <div style="flex: 1; min-width: 130px; text-align: center; padding: 12px; background: white; border-radius: 10px; border: 1px solid #CBD5E1; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                                <div style="font-weight: 700; color: #0F172A; font-size: 13px;">2. RL Agent</div>
                                <div style="font-size: 11px; color: #64748B; margin-top: 4px;">Contrast/Blur Policy</div>
                            </div>
                            <div style="color: #94A3B8; font-size: 18px; font-weight: bold;">➔</div>
                            <div style="flex: 1; min-width: 130px; text-align: center; padding: 12px; background: white; border-radius: 10px; border: 1px solid #CBD5E1; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                                <div style="font-weight: 700; color: #0F172A; font-size: 13px;">3. Hierarchical AI</div>
                                <div style="font-size: 11px; color: #64748B; margin-top: 4px;">Generalist + Specialist</div>
                            </div>
                            <div style="color: #94A3B8; font-size: 18px; font-weight: bold;">➔</div>
                            <div style="flex: 1; min-width: 130px; text-align: center; padding: 12px; background: white; border-radius: 10px; border: 1px solid #CBD5E1; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                                <div style="font-weight: 700; color: #0F172A; font-size: 13px;">4. Agentic RAG</div>
                                <div style="font-size: 11px; color: #64748B; margin-top: 4px;">Clinical Justification</div>
                            </div>
                        </div>
                    """)
                    
                with gr.Column(elem_classes="card-container"):
                    gr.HTML("""
                        <h3 style="margin-top: 0; color: #0F172A; font-size: 16px; font-weight: 700; margin-bottom: 12px;">📋 Recent Patient Entries</h3>
                        <table style="width: 100%; border-collapse: collapse; text-align: left; margin-top: 8px;">
                            <thead>
                                <tr>
                                    <th style="padding: 10px 8px;">Patient ID</th>
                                    <th style="padding: 10px 8px;">Patient Name</th>
                                    <th style="padding: 10px 8px;">Scan Type</th>
                                    <th style="padding: 10px 8px;">Broad Category</th>
                                    <th style="padding: 10px 8px;">Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td style="padding: 12px 8px; font-family: monospace; font-weight: 600;">neuro_908</td>
                                    <td style="padding: 12px 8px; font-weight: 600;">Emma Watson</td>
                                    <td style="padding: 12px 8px;">Brain MRI (.dcm)</td>
                                    <td style="padding: 12px 8px; font-weight: 600; color: #D97706;">genetic</td>
                                    <td style="padding: 12px 8px;"><span style="background-color: #DCFCE7; color: #166534; padding: 4px 10px; border-radius: 9999px; font-size: 11px; font-weight: 700;">Analyzed</span></td>
                                </tr>
                                <tr>
                                    <td style="padding: 12px 8px; font-family: monospace; font-weight: 600;">liver_102</td>
                                    <td style="padding: 12px 8px; font-weight: 600;">Liam Neeson</td>
                                    <td style="padding: 12px 8px;">Abdominal scan</td>
                                    <td style="padding: 12px 8px; font-weight: 600; color: #DC2626;">Malignant</td>
                                    <td style="padding: 12px 8px;"><span style="background-color: #FEF3C7; color: #92400E; padding: 4px 10px; border-radius: 9999px; font-size: 11px; font-weight: 700;">Flagged</span></td>
                                </tr>
                                <tr>
                                    <td style="padding: 12px 8px; font-family: monospace; font-weight: 600;">neuro_002</td>
                                    <td style="padding: 12px 8px; font-weight: 600;">Daniel Craig</td>
                                    <td style="padding: 12px 8px;">Brain MRI (.png)</td>
                                    <td style="padding: 12px 8px; font-weight: 600; color: #2563EB;">infectious</td>
                                    <td style="padding: 12px 8px;"><span style="background-color: #FEE2E2; color: #991B1B; padding: 4px 10px; border-radius: 9999px; font-size: 11px; font-weight: 700;">Urgent Alert</span></td>
                                </tr>
                            </tbody>
                        </table>
                    """)
            
            # SECTION 2: LOAD DATA & ANALYSIS (Original Content Wrapped in Reference Image Visual Layout)
            with gr.Column(visible=False) as load_patient_section:
                # Large Center Card Container (Matches Reference Image 1 Layout)
                with gr.Column(elem_classes="card-container center-upload-card"):
                    gr.HTML("""
                        <div style="display: flex; justify-content: center; margin-bottom: 16px;">
                            <div style="width: 64px; height: 64px; background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 20px; display: flex; align-items: center; justify-content: center; color: #2563EB; font-size: 28px; box-shadow: 0 4px 12px rgba(37,99,235,0.1);">
                                📥
                            </div>
                        </div>
                        <h2 style="margin-top: 0; font-weight: 800; font-size: 22px; color: #0F172A; margin-bottom: 6px;">Load Scan & Patient Details</h2>
                        <p style="color: #64748B; font-size: 13.5px; margin-bottom: 20px; max-width: 480px; margin-left: auto; margin-right: auto; line-height: 1.5;">
                            Upload DICOM, JPEG, or PNG clinical scan slices for automated RL preprocessing, hierarchical AI classification, and RAG diagnosis.
                        </p>
                    """)
                    
                    patient_name_input = gr.Textbox(label="Patient Name", placeholder="Enter patient name...")
                    patient_id_input = gr.Textbox(label="Patient ID", placeholder="Enter patient ID...")
                    
                    input_image = gr.File(
                        file_types=[".png", ".jpg", ".jpeg", ".dcm"],
                        label="Upload Scan (JPEG/PNG/DICOM)"
                    )
                    
                    with gr.Row(elem_classes="action-btn-row"):
                        analyze_btn = gr.Button("⚡ Analyse Scan", elem_classes="primary-btn")
                        reset_btn = gr.Button("🔄 Reset", elem_classes="secondary-btn")

                # Visual Workspace & RL Preprocessing Output Cards
                with gr.Row():
                    with gr.Column(scale=1, elem_classes="card-container"):
                        gr.Markdown("### 👁️ Image Preprocessing Workspace")
                        with gr.Row():
                            original_display = gr.Image(label="Original Scan", interactive=False)
                            processed_display = gr.Image(label="RL-Agent Preprocessed", interactive=False)
                            
                        with gr.Accordion("📋 RL Preprocessor Execution Logs", open=False):
                            rl_logs_box = gr.Code(label="RL Steps & Metrics", language="markdown", interactive=False)

                # Classification Outputs & Cloud Records Card
                with gr.Column(elem_classes="card-container"):
                    gr.Markdown("### 🏷️ Classification Outputs & Cloud Records")
                    with gr.Row():
                        lvl1_box = gr.Textbox(label="Level 1: Broad Category / Organ System", interactive=False, scale=1)
                        lvl2_box = gr.Textbox(label="Level 2: Specific Pathology / Diagnosis", interactive=False, scale=1)
                    with gr.Row():
                        confidence_box = gr.Label(label="Confidence Score", scale=1)
                        s3_url_box = gr.Textbox(label="S3 Cloud Storage Presigned URL", interactive=False, show_copy_button=True, scale=1)
                            
            # SECTION 3: DOCTOR FEEDBACK LOOP (CLINICAL REVIEW)
            with gr.Column(visible=False) as clinical_review_section:
                with gr.Column(elem_classes="card-container"):
                    review_case_display = gr.Markdown(value="### 📋 No patient scan has been analyzed yet in this session.\n*Go to **Load Data** to start an analysis.*")
                        
                with gr.Column(elem_classes="card-container"):
                    gr.Markdown("### 💬 Clinical Correction & Feedback Loop")
                    doctor_feedback_input = gr.Textbox(
                        label="Doctor Feedback / Validation Notes", 
                        placeholder="Enter clinical notes, validations, corrections, or follow-ups for permanent MongoDB logging...",
                        lines=5
                    )
                    submit_feedback_btn = gr.Button("Submit Feedback", elem_classes="primary-btn")
                    feedback_status = gr.Markdown(value="", container=True)
                    
            # SECTION 4: CLINICAL DIAGNOSIS & INTERPRETATION (CLINICAL REPORT)
            with gr.Column(visible=False) as clinical_report_section:
                with gr.Column(elem_classes="card-container"):
                    gr.Markdown("### 📄 Clinical Justification & RAG Report")
                    rag_output = gr.Markdown(value="*Awaiting scan analysis...*", container=True)

    # Navigation Logic Functions
    def navigate_to_overview():
        return (
            gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            gr.update(elem_classes="sidebar-nav-btn active-nav-btn"),
            gr.update(elem_classes="sidebar-nav-btn"),
            gr.update(elem_classes="sidebar-nav-btn"),
            gr.update(elem_classes="sidebar-nav-btn"),
            gr.update(elem_classes="top-tab-card active-top-card"),
            gr.update(elem_classes="top-tab-card"),
            gr.update(elem_classes="top-tab-card"),
            gr.update(elem_classes="top-tab-card")
        )

    def navigate_to_load_data():
        return (
            gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
            gr.update(elem_classes="sidebar-nav-btn"),
            gr.update(elem_classes="sidebar-nav-btn active-nav-btn"),
            gr.update(elem_classes="sidebar-nav-btn"),
            gr.update(elem_classes="sidebar-nav-btn"),
            gr.update(elem_classes="top-tab-card"),
            gr.update(elem_classes="top-tab-card active-top-card"),
            gr.update(elem_classes="top-tab-card"),
            gr.update(elem_classes="top-tab-card")
        )

    def navigate_to_clinical_review():
        return (
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
            gr.update(elem_classes="sidebar-nav-btn"),
            gr.update(elem_classes="sidebar-nav-btn"),
            gr.update(elem_classes="sidebar-nav-btn active-nav-btn"),
            gr.update(elem_classes="sidebar-nav-btn"),
            gr.update(elem_classes="top-tab-card"),
            gr.update(elem_classes="top-tab-card"),
            gr.update(elem_classes="top-tab-card active-top-card"),
            gr.update(elem_classes="top-tab-card")
        )

    def navigate_to_clinical_report():
        return (
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
            gr.update(elem_classes="sidebar-nav-btn"),
            gr.update(elem_classes="sidebar-nav-btn"),
            gr.update(elem_classes="sidebar-nav-btn"),
            gr.update(elem_classes="sidebar-nav-btn active-nav-btn"),
            gr.update(elem_classes="top-tab-card"),
            gr.update(elem_classes="top-tab-card"),
            gr.update(elem_classes="top-tab-card"),
            gr.update(elem_classes="top-tab-card active-top-card")
        )

    # Wire up navigation triggers for SIDEBAR
    nav_overview_btn.click(
        fn=navigate_to_overview,
        inputs=[],
        outputs=[
            overview_section, load_patient_section, clinical_review_section, clinical_report_section,
            nav_overview_btn, nav_load_data_btn, nav_clinical_review_btn, nav_clinical_report_btn,
            top_tab_overview_btn, top_tab_load_data_btn, top_tab_clinical_review_btn, top_tab_clinical_report_btn
        ]
    )

    nav_load_data_btn.click(
        fn=navigate_to_load_data,
        inputs=[],
        outputs=[
            overview_section, load_patient_section, clinical_review_section, clinical_report_section,
            nav_overview_btn, nav_load_data_btn, nav_clinical_review_btn, nav_clinical_report_btn,
            top_tab_overview_btn, top_tab_load_data_btn, top_tab_clinical_review_btn, top_tab_clinical_report_btn
        ]
    )

    nav_clinical_review_btn.click(
        fn=navigate_to_clinical_review,
        inputs=[],
        outputs=[
            overview_section, load_patient_section, clinical_review_section, clinical_report_section,
            nav_overview_btn, nav_load_data_btn, nav_clinical_review_btn, nav_clinical_report_btn,
            top_tab_overview_btn, top_tab_load_data_btn, top_tab_clinical_review_btn, top_tab_clinical_report_btn
        ]
    )

    nav_clinical_report_btn.click(
        fn=navigate_to_clinical_report,
        inputs=[],
        outputs=[
            overview_section, load_patient_section, clinical_review_section, clinical_report_section,
            nav_overview_btn, nav_load_data_btn, nav_clinical_review_btn, nav_clinical_report_btn,
            top_tab_overview_btn, top_tab_load_data_btn, top_tab_clinical_review_btn, top_tab_clinical_report_btn
        ]
    )

    # Wire up navigation triggers for TOP HORIZONTAL TAB CARDS
    top_tab_overview_btn.click(
        fn=navigate_to_overview,
        inputs=[],
        outputs=[
            overview_section, load_patient_section, clinical_review_section, clinical_report_section,
            nav_overview_btn, nav_load_data_btn, nav_clinical_review_btn, nav_clinical_report_btn,
            top_tab_overview_btn, top_tab_load_data_btn, top_tab_clinical_review_btn, top_tab_clinical_report_btn
        ]
    )

    top_tab_load_data_btn.click(
        fn=navigate_to_load_data,
        inputs=[],
        outputs=[
            overview_section, load_patient_section, clinical_review_section, clinical_report_section,
            nav_overview_btn, nav_load_data_btn, nav_clinical_review_btn, nav_clinical_report_btn,
            top_tab_overview_btn, top_tab_load_data_btn, top_tab_clinical_review_btn, top_tab_clinical_report_btn
        ]
    )

    top_tab_clinical_review_btn.click(
        fn=navigate_to_clinical_review,
        inputs=[],
        outputs=[
            overview_section, load_patient_section, clinical_review_section, clinical_report_section,
            nav_overview_btn, nav_load_data_btn, nav_clinical_review_btn, nav_clinical_report_btn,
            top_tab_overview_btn, top_tab_load_data_btn, top_tab_clinical_review_btn, top_tab_clinical_report_btn
        ]
    )

    top_tab_clinical_report_btn.click(
        fn=navigate_to_clinical_report,
        inputs=[],
        outputs=[
            overview_section, load_patient_section, clinical_review_section, clinical_report_section,
            nav_overview_btn, nav_load_data_btn, nav_clinical_review_btn, nav_clinical_report_btn,
            top_tab_overview_btn, top_tab_load_data_btn, top_tab_clinical_review_btn, top_tab_clinical_report_btn
        ]
    )

    # Reactive synchronization triggers between Load Data tab and Clinical Review tab
    def update_review_context(name, pat_id, doc_id_val):
        if not name and not pat_id:
            return "### 📋 No patient scan has been analyzed yet in this session.\n*Go to **Load Data** to start an analysis.*"
        return f"""### 📋 Clinical Case Under Review
*   **Patient Name**: **{name if name else 'Unknown'}**
*   **Patient ID**: **{pat_id if pat_id else 'Unknown'}**
*   **Session Document ID (MongoDB _id)**: `{doc_id_val if doc_id_val else 'Pending Background Upload...'}`
"""

    patient_name_input.change(fn=update_review_context, inputs=[patient_name_input, patient_id_input, current_doc_id], outputs=[review_case_display])
    patient_id_input.change(fn=update_review_context, inputs=[patient_name_input, patient_id_input, current_doc_id], outputs=[review_case_display])
    current_doc_id.change(fn=update_review_context, inputs=[patient_name_input, patient_id_input, current_doc_id], outputs=[review_case_display])

    # Wire up Gradio triggers
    analyze_btn.click(
        fn=analyze_scan,
        inputs=[input_image, patient_name_input, patient_id_input],
        outputs=[
            original_display, 
            processed_display, 
            rl_logs_box, 
            lvl1_box, 
            lvl2_box, 
            confidence_box, 
            s3_url_box,
            current_doc_id,
            rag_output
        ]
    )
    
    reset_btn.click(
        fn=reset_workspace,
        inputs=[],
        outputs=[
            input_image,
            patient_name_input,
            patient_id_input,
            original_display, 
            processed_display, 
            rl_logs_box, 
            lvl1_box, 
            lvl2_box, 
            confidence_box, 
            s3_url_box,
            current_doc_id,
            doctor_feedback_input,
            feedback_status,
            rag_output
        ]
    )

    submit_feedback_btn.click(
        fn=submit_doctor_feedback,
        inputs=[current_doc_id, doctor_feedback_input],
        outputs=[feedback_status]
    )

if __name__ == "__main__":
    # Close any previously running Gradio instances in the kernel environment
    try:
        gr.close_all()
    except Exception:
        pass

    # Start background synchronization worker for offline fallback logging
    try:
        start_offline_sync_loop()
    except Exception as sync_err:
        print(f"[WARNING] Failed to start offline sync worker: {sync_err}")

    launch_kwargs = {"server_name": "0.0.0.0"}
    launched = False
    for target_port in range(7860, 7920):
        try:
            launch_kwargs["server_port"] = target_port
            demo.queue()
            demo.launch(**launch_kwargs)
            launched = True
            print(f"🚀 SetCONNECT Medical AI running successfully on port {target_port}!")
            break
        except OSError as e:
            if "port" in str(e).lower() or "use" in str(e).lower() or "address" in str(e).lower():
                continue
            else:
                raise e
        except Exception as e:
            if "theme" in str(e).lower() or "css" in str(e).lower() or "port" in str(e).lower():
                continue
            else:
                raise e

    if not launched:
        launch_kwargs.pop("server_port", None)
        demo.queue()
        demo.launch(**launch_kwargs)



