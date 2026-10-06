# model_loader.py
import os
import pickle
import torch
import numpy as np
import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer
from sentence_transformers import SentenceTransformer

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Base Paths ─────────────────────────────────────────────────────────────────
PROJECT_DIR = r'C:\Users\Anandita\PycharmProjects\PythonProject\Micromodel'
WEIGHTS_DIR = r'D:\mpr files'
MODELS_DIR  = os.path.join(PROJECT_DIR, 'models')

# ── BioBART Model Name ─────────────────────────────────────────────────────────
BART_MODEL_NAME = 'GanjinZero/biobart-v2-base'

# ── All File Paths ─────────────────────────────────────────────────────────────
PATHS = {
    'router': {
        'router'        : os.path.join(MODELS_DIR, 'router', 'hospital_router.pkl'),
        'label_encoder' : os.path.join(MODELS_DIR, 'router', 'hospital_label_encoder.pkl'),
    },
    'Admin': {
        'weights'       : os.path.join(WEIGHTS_DIR, 'admin_biobart_weights.pt'),
        'embeddings'    : os.path.join(MODELS_DIR,  'admin', 'admin_qa_embeddings.npy'),
        'qa_data'       : os.path.join(MODELS_DIR,  'admin', 'admin_qa_data.csv'),
    },
    'Billing': {
        'weights'       : os.path.join(WEIGHTS_DIR, 'billing_biobart_weights.pt'),
        'embeddings'    : os.path.join(MODELS_DIR,  'billing', 'billing_qa_embeddings.npy'),
        'qa_data'       : os.path.join(MODELS_DIR,  'billing', 'billing_qa_data.csv'),
    },
    'Doctor_Appointment': {
        'weights'       : os.path.join(WEIGHTS_DIR, 'da_biobart_weights.pt'),
        'embeddings'    : os.path.join(MODELS_DIR,  'doctor_appointment', 'da_qa_embeddings.npy'),
        'qa_data'       : os.path.join(MODELS_DIR,  'doctor_appointment', 'da_qa_data.csv'),
    },
    'Emergency': {
        'weights'       : os.path.join(WEIGHTS_DIR, 'emergency_biobart_weights.pt'),
        'embeddings'    : os.path.join(MODELS_DIR,  'emergency', 'emergency_qa_embeddings.npy'),
        'qa_data'       : os.path.join(MODELS_DIR,  'emergency', 'emergency_qa_data.csv'),
    },
    'Pharmacy': {
        'weights'       : os.path.join(WEIGHTS_DIR, 'pharma_biobart_weights.pt'),
        'embeddings'    : os.path.join(MODELS_DIR,  'pharmacy', 'pharma_qa_embeddings.npy'),
        'qa_data'       : os.path.join(MODELS_DIR,  'pharmacy', 'pharma_qa_data.csv'),
    },
}

# ── Verify All Paths Exist ─────────────────────────────────────────────────────
def verify_paths():
    print("\n── Verifying all file paths ──────────────────────────────────────")
    all_ok = True
    for domain, path_dict in PATHS.items():
        for file_type, path in path_dict.items():
            exists = os.path.exists(path)
            status = "✅" if exists else "❌ MISSING"
            print(f"   {status} [{domain}] {file_type}")
            print(f"         {path}")
            if not exists:
                all_ok = False
    if all_ok:
        print("\n✅ All files found — ready to load")
    else:
        print("\n❌ Some files are missing — check paths above")
    return all_ok

# ── Load Router ────────────────────────────────────────────────────────────────
def load_router():
    print("\n── Loading Router ────────────────────────────────────────────────")
    with open(PATHS['router']['router'], 'rb') as f:
        router = pickle.load(f)
    with open(PATHS['router']['label_encoder'], 'rb') as f:
        le = pickle.load(f)
    print("✅ Router loaded")
    print("✅ Label Encoder loaded")
    print(f"   Classes : {le.classes_}")
    return router, le

# ── Load Embedder ──────────────────────────────────────────────────────────────
def load_embedder():
    print("\n── Loading Embedder ──────────────────────────────────────────────")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Embedder loaded")
    return embedder

# ── Load Domain Model ──────────────────────────────────────────────────────────
def load_domain_model(domain):
    print(f"\n── Loading {domain} Model ────────────────────────────────────────")
    model = BartForConditionalGeneration.from_pretrained(BART_MODEL_NAME)
    model.load_state_dict(
        torch.load(
            PATHS[domain]['weights'],
            map_location=DEVICE
        )
    )
    model.to(DEVICE)
    model.eval()
    print(f"✅ {domain} BioBART weights loaded")
    tokenizer = BartTokenizer.from_pretrained(BART_MODEL_NAME)
    print(f"✅ {domain} Tokenizer loaded")
    return model, tokenizer

# ── Load Domain Index ──────────────────────────────────────────────────────────
def load_domain_index(domain):
    print(f"\n── Loading {domain} Index ────────────────────────────────────────")
    embeddings = np.load(PATHS[domain]['embeddings'])
    print(f"✅ {domain} embeddings loaded — shape {embeddings.shape}")
    qa_data = pd.read_csv(PATHS[domain]['qa_data'])
    print(f"✅ {domain} Q&A data loaded  — {len(qa_data)} rows")
    assert embeddings.shape[0] == len(qa_data), \
        f"❌ MISMATCH — embeddings {embeddings.shape[0]} rows " \
        f"vs qa_data {len(qa_data)} rows"
    return embeddings, qa_data

# ── Quick Test ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("══════════════════════════════════════════════════════")
    print("  model_loader.py — Path Verification Test")
    print("══════════════════════════════════════════════════════")
    verify_paths()

    print("\n── Testing Router Load ───────────────────────────────────────────")
    router, le = load_router()

    print("\n── Testing Embedder Load ─────────────────────────────────────────")
    embedder = load_embedder()

    print("\n── Testing Index Load (Admin) ────────────────────────────────────")
    embeddings, qa_data = load_domain_index('Admin')

    print("\n══════════════════════════════════════════════════════")
    print("  All tests passed — model_loader.py is ready")
    print("══════════════════════════════════════════════════════")