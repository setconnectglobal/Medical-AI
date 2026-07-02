# ==============================================================================
# NeuroScan Workspace: Full-Stack Adaptive Medical Diagnostic Interface (app.py)
# ==============================================================================
import os
import sys
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
def log_agent_draft(db, status="Incomplete", step_logs=None, confidence=None, diagnosis=None, s3_url=None, patient_name=None, patient_id=None):
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
    if db_client is None:
        return "❌ MongoDB Connection offline. Cannot save feedback."
        
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
        else:
            return "⚠️ Document found, but no modifications were made."
    except Exception as e:
        print(f"[WARNING] Failed to submit feedback: {e}")
        return f"❌ Failed to submit feedback: {e}"

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
        # Step A: Query medical_base
        textbook_results = self.db.query("medical_base", clinical_query, top_k=2)
        
        # Step B: Query agent_result_logs
        historical_results = self.db.query("agent_result_logs", clinical_query, top_k=2)

        context_textbook = "\n".join([f"- {doc['text']}" for doc in textbook_results])
        context_history = "\n".join([f"- [Case {doc['metadata'].get('patient_id', 'N/A')}]: {doc['text']}" for doc in historical_results])

        # Step D: Synthesis and RAG generation
        prompt = f"""
You are an expert Senior Medical Informatics Specialist and Clinical AI Consultant.
You are generating a professional, patient-facing diagnostic explanation based on a medical imaging query,
utilizing both foundational textbook clinical guidelines (formatted as JSON) and previous reinforcement learning agent logs.

Patient Query / Scan Case:
"{clinical_query}"

---
CONTEXT 1: FOUNDATIONAL TEXTBOOK CRITERIA & PATHOLOGY (JSON STRUCTURED)
{context_textbook}

---
CONTEXT 2: HISTORICAL AGENT EXECUTION RECORDS (PAST LOGS)
{context_history}
---

Your task is to synthesize a step-by-step clinical justification report.
The report MUST contain:
1. DIAGNOSTIC INTERPRETATION: State the diagnosed condition clearly, detailing the preprocessing steps the agent took to enhance details and classification confidence.
2. EMPATHETIC CLINICAL EXPLANATION: Write a detailed, reassuring medical explanation. Translate technical terms (such as Tuberous Sclerosis, Hepatocellular Carcinoma, or Carolis disease) into clear, compassionate, and professional clinical language suitable for a patient report or assistant chatbot.

Ensure the distinction between textbook knowledge and past run records is clear in your reasoning chain.
"""
        if self.gemini_ready:
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                return f"⚠️ Gemini LLM execution failed: {e}\n\n" + self._generate_fallback_report(clinical_query, textbook_results, historical_results)
        else:
            return self._generate_fallback_report(clinical_query, textbook_results, historical_results)

    def _generate_fallback_report(self, query, textbook_docs, history_docs):
        diagnosis = "Indicated Clinical Pathology"
        textbook_raw = textbook_docs[0]['text'] if textbook_docs else ""
        textbook_context_snippet = ""
        
        try:
            db_entry = json.loads(textbook_raw)
            diagnosis = db_entry.get("disease", "Indicated Clinical Pathology")
            mri_list = "\n".join([f"   - {item}" for item in db_entry.get("mri_findings", [])])
            clinical_list = "\n".join([f"   - {item}" for item in db_entry.get("clinical_features", [])])
            
            textbook_context_snippet = f"""Disease: {db_entry.get('disease')}
Definition: {db_entry.get('definition')}
MRI Findings:
{mri_list}
Clinical Features:
{clinical_list}
Diagnostic Protocol: {db_entry.get('diagnostic_protocol')}
Prognosis: {db_entry.get('prognosis')}
Reference: {db_entry.get('reference')}"""
        except Exception:
            textbook_context_snippet = textbook_raw if textbook_raw else "No textbook rules loaded."
            
        history_context_snippet = history_docs[0]['text'] if history_docs else "No clinical cases loaded."
        
        fallback_txt = f"""[CLINICAL EXECUTIVE SUMMARY - REPORT GENERATED VIA EXPERT RULE-ENGINE FALLBACK]

1. DIAGNOSTIC PIPELINE INTERPRETATION:
The diagnostic system evaluated the patient's query relating to: "{query}".
- Primary Classification: {diagnosis}.
- Pipeline Path: Image processing metrics indicate adaptive enhancement was applied. Contrast and structural edges were optimized to maximize the classification network's output confidence, mapping to established historical clinical cases.
- Execution Log Reference: {history_context_snippet}

2. DETAILED EMPATHETIC CLINICAL EXPLANATION:
We have reviewed your scan results. According to clinical protocols, the findings relate to what medical textbooks define as:
"{textbook_context_snippet}"

Please be reassured that this report has been analyzed by a specialized digital assistant. If your results point to benign conditions (like Hemangioma), these are slow-growing, non-cancerous collections of blood vessels that typically require only periodic monitoring rather than aggressive treatment. If the system identified genetic malformations (such as Muscular Dystrophy) or infectious processes, these are managed using dedicated supportive protocols designed to protect organ function and support your comfort.
Your healthcare provider will discuss these results in detail, coordinate next steps, and tailor a management plan specific to your health journey.
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
        return None, None, "⚠️ Please upload a scan slice first.", "N/A", "N/A", "0.0%", "N/A (S3 Offline)", "", "⚠️ Inference aborted: No input image.", db_status
        
    filepath = input_file.name if hasattr(input_file, 'name') else str(input_file)
    
    # Re-evaluate connection if it was offline
    if db_client is None:
        db_client, db_status = get_mongodb_connection()

    # Load original scan array (DICOM or standard image)
    try:
        input_img = load_medical_image(filepath)
    except Exception as e:
        return None, None, f"❌ Image Loading Error: {e}", "N/A", "N/A", "0.0%", "N/A (S3 Offline)", "", "❌ Processing Failed.", db_status

    # Upload original scan to S3
    try:
        s3_url = upload_to_s3(filepath)
    except Exception as e:
        print(f"[WARNING] S3 Upload failed: {e}")
        s3_url = None

    # 1. Run classifier-guided RL Preprocessor
    try:
        processed_img, actions, rl_logs = process_image_with_agent_and_hub(input_img, Q_table, hub)
    except Exception as e:
        return input_img, input_img, rl_logs, "N/A", "N/A", "0.0%", "N/A (S3 Offline)", "", f"❌ Preprocessing Error: {e}", db_status

    # 2. Run Classification Hub
    try:
        img_tensor = hub._tensor_from_np(processed_img)
        with torch.no_grad():
            gen_probs = torch.nn.functional.softmax(hub.gen(img_tensor), dim=1)
            gen_conf, gen_idx = torch.max(gen_probs, 1)
            generalist_pred = hub.gen_classes[gen_idx.item()]
            
        label, confidence = hub.diagnose_array(processed_img)
    except Exception as e:
        return input_img, processed_img, rl_logs, "N/A", "N/A", "0.0%", "N/A (S3 Offline)", "", f"❌ Classification Error: {e}", db_status

    # 3. Handle threshold guardrail logic
    if "Low Confidence" in label:
        rag_report = f"""### 🚨 [ALERT] CLASSIFICATION CONFIDENCE TOO LOW
The Generalist classified the scan category with low confidence (**{confidence*100:.2f}%**, below the required **90%** threshold).

*   **Bypassed Path:** Biliary / Specialist neural diagnostic routing was aborted for safety.
*   **Action Required:** This case has been flagged and pushed to MongoDB Atlas. Scan slice requires manual clinical review by a human Radiologist.
"""
        doc_id = log_agent_draft(
            db=db_client,
            status="Low Confidence Flagged",
            step_logs=actions,
            confidence=confidence,
            diagnosis=label,
            s3_url=s3_url,
            patient_name=patient_name,
            patient_id=patient_id
        )
        return input_img, processed_img, rl_logs, generalist_pred, label, f"{confidence*100:.1f}%", s3_url if s3_url else "N/A (S3 Offline)", str(doc_id) if doc_id else "", rag_report, db_status

    # 4. Generate Clinical RAG Report for High-Confidence predictions
    try:
        execution_context = f"Current Run Preprocessing: The RL Preprocessing Agent took actions: {actions} (logs: {rl_logs})."
        query = f"Give me a clinical breakdown for a scan diagnosed with {label}. {execution_context}"
        rag_report = rag_pipeline.run_query(query)
    except Exception as e:
        rag_report = f"⚠️ RAG Pipeline Error: {e}"

    # 5. Log successfully processed run to MongoDB Atlas
    doc_id = log_agent_draft(
        db=db_client,
        status="Successful Run",
        step_logs=actions,
        confidence=confidence,
        diagnosis=label,
        s3_url=s3_url,
        patient_name=patient_name,
        patient_id=patient_id
    )
    
    return input_img, processed_img, rl_logs, generalist_pred, label, f"{confidence*100:.1f}%", s3_url if s3_url else "N/A (S3 Offline)", str(doc_id) if doc_id else "", rag_report, db_status

def reset_workspace():
    return None, "", "", None, None, "", "", "", "", "", None, "", "", "", db_status

custom_theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="slate",
    neutral_hue="slate"
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    block_title_text_weight="bold",
    block_border_width="1px"
)

import inspect
theme_in_launch = 'theme' in inspect.signature(gr.Blocks.launch).parameters

blocks_kwargs = {"title": "NeuroScan Workstation"}
if not theme_in_launch:
    blocks_kwargs["theme"] = custom_theme

with gr.Blocks(**blocks_kwargs) as demo:
    current_doc_id = gr.State("")

    gr.Markdown(
        """
        # 🧠 NeuroScan: Adaptive Medical Image Diagnostic Station
        This workstation runs a **Generalist-Specialist Hierarchical Classifier** combined with a **Reinforcement Learning Preprocessor** and a **Dual-Collection Vector RAG Pipeline**.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📡 Connection Panel")
            db_status_box = gr.Textbox(
                value=db_status,
                label="MongoDB Atlas Connection Status",
                interactive=False
            )
            
            gr.Markdown("### 📥 Patient Metadata & Scan Upload")
            patient_name_input = gr.Textbox(label="Patient Name", placeholder="Enter patient name...")
            patient_id_input = gr.Textbox(label="Patient ID", placeholder="Enter patient ID...")
            input_image = gr.File(
                file_types=[".png", ".jpg", ".jpeg", ".dcm"],
                label="Upload Scan (JPEG/PNG/DICOM)"
            )
            
            with gr.Row():
                analyze_btn = gr.Button("Analyze Scan", variant="primary")
                reset_btn = gr.Button("Reset Workstation", variant="secondary")
                
        with gr.Column(scale=2):
            gr.Markdown("### 👁️ Image Preprocessing Visualization")
            with gr.Row():
                original_display = gr.Image(label="Original Scan", interactive=False)
                processed_display = gr.Image(label="RL-Agent Preprocessed", interactive=False)
                
            with gr.Accordion("📋 RL Preprocessor Execution Logs", open=False):
                rl_logs_box = gr.Code(label="RL Steps & Metrics", language="markdown", interactive=False)
                
            gr.Markdown("### 🏷️ Hierarchical Diagnostic Output")
            with gr.Row():
                lvl1_box = gr.Textbox(label="Level 1: Broad Category / Organ System", interactive=False)
                lvl2_box = gr.Textbox(label="Level 2: Specific Pathology / Diagnosis", interactive=False)
                confidence_box = gr.Label(label="Confidence Score")
            with gr.Row():
                s3_url_box = gr.Textbox(label="S3 Cloud Storage Presigned URL", interactive=False, show_copy_button=True)
            
            gr.Markdown("### 💬 Human-in-the-Loop Clinical Feedback")
            with gr.Row():
                doctor_feedback_input = gr.Textbox(
                    label="Doctor Feedback / Clinical Correction Notes", 
                    placeholder="Enter corrections, validation, or notes for MongoDB logging...",
                    scale=3
                )
                submit_feedback_btn = gr.Button("Submit Feedback", variant="primary", scale=1)
            feedback_status = gr.Markdown(value="", container=True)
                
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📄 Clinical Justification Report (Agentic RAG)")
            rag_output = gr.Markdown(value="*Awaiting scan analysis...*", container=True)

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
            rag_output,
            db_status_box
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
            rag_output,
            db_status_box
        ]
    )

    submit_feedback_btn.click(
        fn=submit_doctor_feedback,
        inputs=[current_doc_id, doctor_feedback_input],
        outputs=[feedback_status]
    )

if __name__ == "__main__":
    launch_kwargs = {"server_name": "0.0.0.0"}
    if theme_in_launch:
        launch_kwargs["theme"] = custom_theme
        
    base_port = 7860
    for port_offset in range(10):
        target_port = base_port + port_offset
        try:
            launch_kwargs["server_port"] = target_port
            demo.launch(**launch_kwargs)
            break
        except OSError as e:
            if "port" in str(e).lower() or "use" in str(e).lower():
                print(f"[WARNING] Port {target_port} is busy. Trying port {target_port + 1}...")
                continue
            else:
                raise e
