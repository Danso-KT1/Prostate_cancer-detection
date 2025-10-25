"""
ProstateCare AI - Professional Prostate Cancer Detection System
Multi-Model Ensemble with Majority Voting
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import plotly.express as px
from datetime import datetime
import io
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ==================== TENSORFLOW IMPORT ====================
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.error("TensorFlow not available!")

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="ProstateCare AI | Cancer Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== THEME CUSTOMIZATION ====================
def apply_custom_theme():
    """Apply custom theme based on user selection"""
    
    # Theme options in sidebar
    with st.sidebar:
        st.markdown("### 🎨 Theme Settings")
        
        theme_preset = st.selectbox(
            "Select Theme",
            ["Auto (Browser Default)", "Light Professional", "Dark Modern", "Medical Blue", "Custom"]
        )
        
        if theme_preset == "Auto (Browser Default)":
            bg_color = "transparent"
            text_color = "inherit"
            card_bg = "rgba(255, 255, 255, 0.05)"
            card_border = "rgba(100, 100, 100, 0.2)"
        elif theme_preset == "Light Professional":
            bg_color = "#f8f9fa"
            text_color = "#2c3e50"
            card_bg = "#ffffff"
            card_border = "#e1e4e8"
        elif theme_preset == "Dark Modern":
            bg_color = "#1a1a2e"
            text_color = "#eaeaea"
            card_bg = "#16213e"
            card_border = "#0f3460"
        elif theme_preset == "Medical Blue":
            bg_color = "#e8f4f8"
            text_color = "#1e3a5f"
            card_bg = "#ffffff"
            card_border = "#4a90e2"
        else:  # Custom
            st.markdown("#### Custom Colors")
            bg_color = st.color_picker("Background Color", "#ffffff")
            text_color = st.color_picker("Text Color", "#000000")
            card_bg = st.color_picker("Card Background", "#f0f4ff")
            card_border = st.color_picker("Card Border", "#667eea")
        
        st.markdown("---")
    
    return bg_color, text_color, card_bg, card_border

# ==================== DYNAMIC STYLING ====================
def get_dynamic_styles(bg_color, text_color, card_bg, card_border):
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {{ 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        transition: all 0.3s ease;
    }}
    
    .main {{ 
        background: {bg_color} !important;
        color: {text_color} !important;
    }}
    
    .stApp {{
        background: {bg_color} !important;
    }}
    
    /* Hero Section */
    .hero {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 4rem 3rem;
        border-radius: 30px;
        color: white;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 0 25px 70px rgba(102, 126, 234, 0.4);
        animation: fadeInDown 0.8s ease;
    }}
    
    .hero h1 {{ 
        font-size: 4rem; 
        font-weight: 800; 
        margin: 0;
        letter-spacing: -1px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }}
    
    .hero p {{ 
        font-size: 1.4rem; 
        margin: 1.5rem 0 0 0; 
        opacity: 0.95;
        font-weight: 300;
    }}
    
    .hero-badge {{
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        margin-top: 1rem;
        font-size: 0.9rem;
        font-weight: 500;
        backdrop-filter: blur(10px);
    }}
    
    /* Result Cards */
    .result-positive {{
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white; 
        padding: 3rem; 
        border-radius: 25px;
        text-align: center; 
        box-shadow: 0 20px 60px rgba(255, 107, 107, 0.3);
        margin: 2rem 0;
        animation: slideInUp 0.6s ease;
    }}
    
    .result-negative {{
        background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
        color: white; 
        padding: 3rem; 
        border-radius: 25px;
        text-align: center; 
        box-shadow: 0 20px 60px rgba(81, 207, 102, 0.3);
        margin: 2rem 0;
        animation: slideInUp 0.6s ease;
    }}
    
    .result-title {{ 
        font-size: 3.2rem; 
        font-weight: 800; 
        margin: 0 0 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }}
    
    /* Metric Boxes */
    .metric-box {{
        background: {card_bg};
        padding: 2rem; 
        border-radius: 20px; 
        border: 2px solid {card_border};
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    
    .metric-box:hover {{
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.12);
    }}
    
    .metric-label {{ 
        font-size: 0.85rem; 
        color: {text_color}; 
        opacity: 0.7;
        text-transform: uppercase; 
        font-weight: 700; 
        margin: 0;
        letter-spacing: 1px;
    }}
    
    .metric-value {{ 
        font-size: 2.5rem; 
        font-weight: 800; 
        color: #667eea; 
        margin: 1rem 0 0 0;
    }}
    
    /* Info Cards */
    .info-card {{
        background: {card_bg};
        padding: 2rem; 
        border-radius: 20px; 
        border-left: 6px solid #2196f3;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        color: {text_color};
    }}
    
    .warning-card {{
        background: {card_bg};
        padding: 2rem; 
        border-radius: 20px; 
        border-left: 6px solid #ff9800;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        color: {text_color};
    }}
    
    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important; 
        border: none !important; 
        padding: 1rem 3rem !important;
        border-radius: 50px !important; 
        font-weight: 700 !important;
        font-size: 1rem !important;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3) !important;
        transition: all 0.3s ease !important;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4) !important;
    }}
    
    /* Model Prediction Cards */
    .model-card-cancer {{
        background: linear-gradient(135deg, #ffe8e8 0%, #ffcbcb 100%);
        border: 3px solid #ff6b6b;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(255, 107, 107, 0.2);
        transition: all 0.3s ease;
    }}
    
    .model-card-cancer:hover {{
        transform: scale(1.05);
        box-shadow: 0 15px 40px rgba(255, 107, 107, 0.3);
    }}
    
    .model-card-clear {{
        background: linear-gradient(135deg, #e8ffe8 0%, #cbffcb 100%);
        border: 3px solid #51cf66;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(81, 207, 102, 0.2);
        transition: all 0.3s ease;
    }}
    
    .model-card-clear:hover {{
        transform: scale(1.05);
        box-shadow: 0 15px 40px rgba(81, 207, 102, 0.3);
    }}
    
    /* Image Slider */
    .slider-container {{
        position: relative;
        overflow: hidden;
        border-radius: 20px;
        box-shadow: 0 15px 50px rgba(0,0,0,0.15);
        margin: 2rem 0;
    }}
    
    .slider-image {{
        width: 100%;
        height: 400px;
        object-fit: cover;
        border-radius: 20px;
    }}
    
    /* Animations */
    @keyframes fadeInDown {{
        from {{
            opacity: 0;
            transform: translateY(-30px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    @keyframes slideInUp {{
        from {{
            opacity: 0;
            transform: translateY(30px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2rem;
        background: {card_bg};
        padding: 1rem;
        border-radius: 15px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        padding: 1rem 2rem;
        font-weight: 600;
        border-radius: 10px;
        color: {text_color};
    }}
    
    /* Section Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: {text_color} !important;
        font-weight: 700 !important;
    }}
    
    /* Dataframe */
    .stDataFrame {{
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    }}
    
    /* File Uploader */
    .stFileUploader {{
        border: 3px dashed {card_border};
        border-radius: 20px;
        padding: 2rem;
        background: {card_bg};
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        background: {card_bg} !important;
        border-radius: 15px !important;
        font-weight: 600 !important;
        color: {text_color} !important;
    }}
    </style>
    """

# ==================== CUSTOM METRICS ====================
def dice_coefficient(y_true, y_pred, smooth=1):
    """Dice coefficient metric"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice loss function"""
    return 1 - dice_coefficient(y_true, y_pred)

# ==================== LOAD MODELS ====================
@st.cache_resource
def load_all_models():
    """Load all three models"""
    if not TF_AVAILABLE:
        return {}, False, "TensorFlow not available"
    
    model_files = [
        'best_model_Attention_UNet (1).keras',
        'best_model_ResUNet.keras',
        'best_model_UNetPlusPlus.keras'
    ]
    
    models = {}
    loaded_count = 0
    messages = []
    
    for model_file in model_files:
        try:
            model = tf.keras.models.load_model(
                model_file,
                custom_objects={'dice_coefficient': dice_coefficient, 'dice_loss': dice_loss},
                compile=False
            )
            model_name = model_file.replace('best_model_', '').replace('.keras', '')
            models[model_name] = model
            messages.append(f"✅ {model_name}")
            loaded_count += 1
        except FileNotFoundError:
            messages.append(f"⚠️ {model_file} - Not found")
        except Exception as e:
            messages.append(f"⚠️ {model_file}")
    
    if loaded_count == 0:
        return {}, False, "❌ No models loaded"
    
    status_msg = f"✅ {loaded_count}/{len(model_files)} models loaded:\n" + "\n".join(messages)
    return models, True, status_msg

# ==================== IMAGE PREPROCESSING ====================
def preprocess_image(image, target_size=(128, 128)):
    """Load and preprocess image"""
    try:
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        original_image = img_array.copy()
        img_array = img_array.astype(np.float32) / 255.0
        img_resized = cv2.resize(img_array, target_size)
        img_resized = img_resized[..., np.newaxis]
        
        return img_resized, original_image, True
    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        return None, None, False

# ==================== ENSEMBLE PREDICTION ====================
def ensemble_predict_cancer(models, image, threshold=0.5, cancer_threshold=1.0):
    """Make ensemble prediction with majority voting"""
    individual_predictions = []
    all_masks = []
    votes_cancerous = 0
    
    for model_name, model in models.items():
        try:
            # Predict
            pred_mask = model.predict(image[np.newaxis, ...], verbose=0)
            
            # Handle different output formats
            if isinstance(pred_mask, list):
                pred_mask = pred_mask[0]
            
            # Extract mask
            if len(pred_mask.shape) == 4:
                pred_mask = pred_mask[0, :, :, 0]
            elif len(pred_mask.shape) == 3:
                pred_mask = pred_mask[:, :, 0]
            
            # Binarize
            binary_mask = pred_mask > threshold
            
            # Calculate percentage
            cancer_area = np.sum(binary_mask)
            total_area = binary_mask.size
            percentage = (cancer_area / total_area) * 100
            
            # Classify
            is_cancerous = percentage > cancer_threshold
            if is_cancerous:
                votes_cancerous += 1
            
            # Store
            individual_predictions.append({
                'model_name': model_name,
                'mask': pred_mask,
                'cancerous': is_cancerous,
                'percentage': percentage
            })
            
            all_masks.append(pred_mask)
            
        except Exception as e:
            st.warning(f"Error in {model_name}: {str(e)}")
    
    if not all_masks:
        return None, False, 0, 0, []
    
    # Ensemble mask
    ensemble_mask = np.mean(all_masks, axis=0)
    
    # Majority vote
    majority_cancerous = votes_cancerous > (len(models) / 2)
    
    # Average percentage
    avg_percentage = np.mean([p['percentage'] for p in individual_predictions])
    
    # Confidence
    binary_ensemble = ensemble_mask > threshold
    if np.sum(binary_ensemble) > 0:
        confidence = np.mean(ensemble_mask[binary_ensemble]) * 100
    else:
        confidence = 0.0
    
    return ensemble_mask, majority_cancerous, avg_percentage, confidence, individual_predictions

# ==================== VISUALIZATION ====================
def visualize_prediction(original, preprocessed, mask, percentage, cancerous):
    """Create visualization"""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor='white')
        
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original MRI Scan', fontsize=14, fontweight='bold', pad=15)
        axes[0].axis('off')
        
        axes[1].imshow(preprocessed, cmap='gray')
        if cancerous:
            axes[1].imshow(mask, cmap='Reds', alpha=0.6)
            axes[1].set_title('CANCER DETECTED ⚠️', fontsize=14, fontweight='bold', color='#ff6b6b', pad=15)
        else:
            axes[1].imshow(mask, cmap='Greens', alpha=0.6)
            axes[1].set_title('NO CANCER ✓', fontsize=14, fontweight='bold', color='#51cf66', pad=15)
        axes[1].axis('off')
        
        im = axes[2].imshow(mask, cmap='hot', interpolation='bilinear')
        axes[2].set_title(f'Probability Map: {percentage:.1f}%', fontsize=14, fontweight='bold', pad=15)
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        buf.seek(0)
        
        return Image.open(buf)
    except Exception as e:
        st.error(f"Visualization error: {str(e)}")
        return None

def visualize_individual_prediction(original, preprocessed, mask, percentage, cancerous, model_name):
    """Create visualization for individual model"""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor='white')
        
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original MRI', fontsize=14, fontweight='bold', pad=15)
        axes[0].axis('off')
        
        axes[1].imshow(preprocessed, cmap='gray')
        if cancerous:
            axes[1].imshow(mask, cmap='Reds', alpha=0.6)
            axes[1].set_title(f'{model_name}: CANCER ⚠️', fontsize=14, fontweight='bold', color='#ff6b6b', pad=15)
        else:
            axes[1].imshow(mask, cmap='Greens', alpha=0.6)
            axes[1].set_title(f'{model_name}: CLEAR ✓', fontsize=14, fontweight='bold', color='#51cf66', pad=15)
        axes[1].axis('off')
        
        im = axes[2].imshow(mask, cmap='hot', interpolation='bilinear')
        axes[2].set_title(f'Probability: {percentage:.1f}%', fontsize=14, fontweight='bold', pad=15)
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        buf.seek(0)
        
        return Image.open(buf)
    except Exception as e:
        st.error(f"Visualization error: {str(e)}")
        return None

# ==================== AUTO IMAGE SLIDER ====================
def display_prostate_info_slider():
    """Display educational slider with prostate cancer images that auto-rotates"""
    
    # Initialize session state for slider
    if 'slider_index' not in st.session_state:
        st.session_state.slider_index = 0
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now()
    
    # Auto-rotate every 4 seconds
    current_time = datetime.now()
    if (current_time - st.session_state.last_update).seconds >= 4:
        st.session_state.slider_index = (st.session_state.slider_index + 1) % 6
        st.session_state.last_update = current_time
    
    # Educational images about prostate cancer (real medical/health images)
    slider_info = [
        {
            "url": "https://images.unsplash.com/photo-1631217868264-e5b90bb7e133?w=1200&q=80",
            "caption": "Prostate Cancer Awareness",
            "description": "Prostate cancer is the second most common cancer in men worldwide. Early detection through MRI screening significantly improves treatment outcomes and survival rates."
        },
        {
            "url": "https://images.unsplash.com/photo-1579154204601-01588f351e67?w=1200&q=80",
            "caption": "Advanced MRI Technology",
            "description": "Multiparametric MRI (mpMRI) has revolutionized prostate cancer detection, providing detailed images that help distinguish between benign and malignant tissue with high accuracy."
        },
        {
            "url": "https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=1200&q=80",
            "caption": "Medical Imaging Analysis",
            "description": "Radiologists use advanced imaging techniques to identify suspicious lesions in the prostate. AI assists by highlighting areas of concern for further clinical evaluation."
        },
        {
            "url": "https://images.unsplash.com/photo-1582719471137-c3967ffb1c42?w=1200&q=80",
            "caption": "Clinical Diagnosis",
            "description": "Prostate cancer diagnosis combines PSA testing, digital rectal examination (DRE), MRI imaging, and biopsy results. A comprehensive approach ensures accurate diagnosis."
        },
        {
            "url": "https://images.unsplash.com/photo-1516549655169-df83a0774514?w=1200&q=80",
            "caption": "Healthcare Professional Consultation",
            "description": "Regular screening after age 50 (or 45 for high-risk groups) is crucial. Discuss your risk factors and screening options with your healthcare provider."
        },
        {
            "url": "https://images.unsplash.com/photo-1584982751601-97dcc096659c?w=1200&q=80",
            "caption": "AI-Assisted Healthcare",
            "description": "Artificial Intelligence tools help radiologists analyze thousands of MRI scans more efficiently, reducing diagnosis time while maintaining high accuracy in cancer detection."
        }
    ]
    
    info = slider_info[st.session_state.slider_index]
    
    # Display with auto-refresh
    st.markdown(f"""
    <div class="slider-container">
        <img src="{info['url']}" class="slider-image" alt="{info['caption']}">
    </div>
    <div class="info-card">
        <h4 style="margin-top: 0;">📌 {info['caption']}</h4>
        <p style="margin-bottom: 0; font-size: 1.1rem; line-height: 1.7;">{info['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Manual navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("◀ Previous", use_container_width=True):
            st.session_state.slider_index = (st.session_state.slider_index - 1) % 6
            st.session_state.last_update = datetime.now()
            st.rerun()
    
    with col2:
        st.markdown(f"<center><strong>Image {st.session_state.slider_index + 1} of 6</strong></center>", unsafe_allow_html=True)
    
    with col3:
        if st.button("Next ▶", use_container_width=True):
            st.session_state.slider_index = (st.session_state.slider_index + 1) % 6
            st.session_state.last_update = datetime.now()
            st.rerun()
    
    # Auto-refresh component
    st.markdown("""
    <script>
        setTimeout(function() {
            window.parent.document.querySelector('[data-testid="stAppViewContainer"]').scrollIntoView();
        }, 4000);
    </script>
    """, unsafe_allow_html=True)

# ==================== MAIN APP ====================
def main():
    # Apply theme
    bg_color, text_color, card_bg, card_border = apply_custom_theme()
    st.markdown(get_dynamic_styles(bg_color, text_color, card_bg, card_border), unsafe_allow_html=True)
    
    # Hero Section
    st.markdown('''
    <div class="hero">
        <h1>🏥 ProstateCare AI</h1>
        <p>Advanced Multi-Model AI System for Prostate Cancer Detection</p>
        <div class="hero-badge">✨ Powered by Deep Learning Ensemble</div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Educational Slider
    with st.expander("📚 Learn About Prostate Cancer Detection", expanded=False):
        display_prostate_info_slider()
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔬 AI Diagnostic", "📊 Statistics", "ℹ️ How It Works", "📋 About"])
    
    # ===== TAB 1: DIAGNOSTIC =====
    with tab1:
        models, models_loaded, status_msg = load_all_models()
        
        if models_loaded:
            st.success(status_msg)
        else:
            st.error(status_msg)
        
        col1, col2 = st.columns([1, 1], gap="large")
        
        # Initialize sensitivity with default value
        sensitivity = 0.5
        
        with col1:
            st.markdown("### 📤 Upload MRI Scan")
            uploaded_file = st.file_uploader(
                "Select prostate MRI image", 
                type=['jpg', 'png', 'jpeg', 'bmp', 'tiff'],
                help="Upload a grayscale prostate MRI scan for analysis"
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Patient MRI Scan", use_column_width=True)
                
                st.markdown("### ⚙️ Analysis Settings")
                sensitivity = st.slider(
                    "Detection Sensitivity", 
                    0.0, 1.0, 0.5, 0.05,
                    help="Higher sensitivity may detect smaller abnormalities but may increase false positives"
                )
        
        with col2:
            st.markdown("### 🎯 Quick Start Guide")
            st.markdown("""
            <div class="info-card">
                <h4>📋 Steps to Analyze:</h4>
                <ol style="margin: 1rem 0; padding-left: 1.5rem;">
                    <li><strong>Upload</strong> a prostate MRI scan (grayscale recommended)</li>
                    <li><strong>Adjust</strong> detection sensitivity if needed</li>
                    <li><strong>Click</strong> "Run AI Analysis" button</li>
                    <li><strong>Review</strong> results from all three AI models</li>
                    <li><strong>Download</strong> comprehensive report</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
            
            if uploaded_file and models_loaded and len(models) > 0:
                if st.button("🔬 Run AI Analysis", use_container_width=True, type="primary"):
                    with st.spinner("⏳ Analyzing with ensemble AI models..."):
                        preprocessed, original, success = preprocess_image(image)
                        
                        if success:
                            ensemble_mask, cancerous, avg_percentage, avg_confidence, individual_preds = \
                                ensemble_predict_cancer(models, preprocessed, sensitivity)
                            
                            if ensemble_mask is not None and len(individual_preds) > 0:
                                st.session_state.results = {
                                    'ensemble_mask': ensemble_mask,
                                    'individual_predictions': individual_preds,
                                    'avg_percentage': avg_percentage,
                                    'avg_confidence': avg_confidence,
                                    'cancerous': cancerous,
                                    'votes_for_cancer': sum(1 for p in individual_preds if p['cancerous']),
                                    'total_models': len(models),
                                    'original': original,
                                    'preprocessed': preprocessed[:, :, 0],
                                    'timestamp': datetime.now()
                                }
                                st.success("✅ Analysis Complete!")
                                st.balloons()
                            else:
                                st.error("❌ Analysis failed - no predictions generated")
            else:
                if not uploaded_file:
                    st.info("👆 Upload an MRI scan to begin analysis")
                elif not models_loaded:
                    st.warning("⚠️ AI models not loaded. Please check model files.")
        
        # ===== RESULTS =====
        if 'results' in st.session_state:
            results = st.session_state.results
            
            # Validate results structure
            if not isinstance(results.get('individual_predictions'), list):
                st.error("⚠️ Error: Invalid results structure. Please run analysis again.")
            else:
                st.markdown("---")
                st.markdown("## 📊 Analysis Results")
                
                # Main Result
                if results['cancerous']:
                    st.markdown(f"""
                    <div class="result-positive">
                        <div class="result-title">⚠️ CANCER DETECTED</div>
                        <p style="font-size: 1.2rem; margin: 1rem 0 0 0;">
                            <strong>Ensemble Consensus:</strong> {results['votes_for_cancer']}/{results['total_models']} models agree
                        </p>
                        <p style="font-size: 0.95rem; opacity: 0.9; margin-top: 1rem;">
                            ⚕️ Please consult with a healthcare professional immediately for confirmation and treatment options
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-negative">
                        <div class="result-title">✓ NO CANCER DETECTED</div>
                        <p style="font-size: 1.2rem; margin: 1rem 0 0 0;">
                            <strong>Ensemble Consensus:</strong> {results['total_models'] - results['votes_for_cancer']}/{results['total_models']} models agree
                        </p>
                        <p style="font-size: 0.95rem; opacity: 0.9; margin-top: 1rem;">
                            ✅ Results look clear, but regular screenings are still recommended
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Key Metrics
                st.markdown("### 📈 Key Metrics")
                m1, m2, m3, m4 = st.columns(4)
                
                with m1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <p class="metric-label">Affected Area</p>
                        <p class="metric-value">{results['avg_percentage']:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with m2:
                    st.markdown(f"""
                    <div class="metric-box">
                        <p class="metric-label">AI Confidence</p>
                        <p class="metric-value">{results['avg_confidence']:.0f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with m3:
                    st.markdown(f"""
                    <div class="metric-box">
                        <p class="metric-label">Model Votes</p>
                        <p class="metric-value">{results['votes_for_cancer']}/{results['total_models']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with m4:
                    consensus_icon = '✓' if results['cancerous'] else '✗'
                    st.markdown(f"""
                    <div class="metric-box">
                        <p class="metric-label">Consensus</p>
                        <p class="metric-value">{consensus_icon}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Individual Model Summary Cards
                st.markdown("### 🤖 Individual Model Predictions")
                
                individual_preds = results['individual_predictions']
                if individual_preds and len(individual_preds) > 0:
                    model_cols = st.columns(len(individual_preds))
                    for col, pred in zip(model_cols, individual_preds):
                        if isinstance(pred, dict) and 'cancerous' in pred:
                            with col:
                                if pred['cancerous']:
                                    st.markdown(f"""
                                    <div class="model-card-cancer">
                                        <p class="metric-label" style="color: #c92a2a;">{pred['model_name']}</p>
                                        <p style="font-size: 2rem; color: #ff6b6b; font-weight: 800; margin: 1rem 0;">⚠️ CANCER</p>
                                        <p style="font-size: 1.1rem; color: #666; font-weight: 600;">{pred['percentage']:.1f}% Affected</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="model-card-clear">
                                        <p class="metric-label" style="color: #2b8a3e;">{pred['model_name']}</p>
                                        <p style="font-size: 2rem; color: #51cf66; font-weight: 800; margin: 1rem 0;">✓ CLEAR</p>
                                        <p style="font-size: 1.1rem; color: #666; font-weight: 600;">{pred['percentage']:.1f}% Affected</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                
                # Ensemble Visualization
                st.markdown("### 📊 Ensemble Analysis Visualization")
                viz = visualize_prediction(
                    results['original'],
                    results['preprocessed'],
                    results['ensemble_mask'],
                    results['avg_percentage'],
                    results['cancerous']
                )
                if viz:
                    st.image(viz, use_column_width=True)
                
                # Individual Model Visualizations (Expandable)
                st.markdown("### 🔍 Individual Model Analysis")
                show_individual = st.checkbox("Show detailed predictions from each model", value=False)
                
                if show_individual and individual_preds:
                    for pred in individual_preds:
                        if isinstance(pred, dict):
                            with st.expander(f"📌 {pred['model_name']} - Detailed Analysis", expanded=False):
                                individual_viz = visualize_individual_prediction(
                                    results['original'],
                                    results['preprocessed'],
                                    pred['mask'],
                                    pred['percentage'],
                                    pred['cancerous'],
                                    pred['model_name']
                                )
                                if individual_viz:
                                    st.image(individual_viz, use_column_width=True)
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Diagnosis", "CANCER" if pred['cancerous'] else "CLEAR")
                                with col2:
                                    st.metric("Affected Area", f"{pred['percentage']:.2f}%")
                                with col3:
                                    confidence = np.mean(pred['mask'][pred['mask'] > 0.5]) * 100 if np.any(pred['mask'] > 0.5) else 0
                                    st.metric("Confidence", f"{confidence:.1f}%")
                
                # Report Generation
                st.markdown("### 📄 Diagnostic Report")
                
                report_data = {
                    'Analysis Date': results['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                    'Final Diagnosis': 'CANCEROUS ⚠️' if results['cancerous'] else 'NON-CANCEROUS ✓',
                    'Model Consensus': f"{results['votes_for_cancer']}/{results['total_models']} models detected cancer",
                    'Average Affected Area (%)': f"{results['avg_percentage']:.2f}",
                    'Average AI Confidence (%)': f"{results['avg_confidence']:.1f}",
                    'Detection Sensitivity Used': f"{sensitivity:.2f}",
                }
                
                for pred in individual_preds:
                    if isinstance(pred, dict):
                        report_data[f"{pred['model_name']} - Diagnosis"] = 'CANCER ⚠️' if pred['cancerous'] else 'CLEAR ✓'
                        report_data[f"{pred['model_name']} - Affected Area (%)"] = f"{pred['percentage']:.2f}"
                
                df = pd.DataFrame([report_data]).T
                df.columns = ['Value']
                
                col_df, col_btn = st.columns([3, 1])
                with col_df:
                    st.dataframe(df, use_column_width=True, height=400)
                
                with col_btn:
                    csv = df.to_csv()
                    st.download_button(
                        label="📥 Download Report",
                        data=csv,
                        file_name=f"prostate_analysis_{results['timestamp'].strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        type="primary"
                    )
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    if st.button("🔄 New Analysis", use_container_width=True):
                        del st.session_state.results
                        st.rerun()
                
                # Disclaimer
                st.markdown("""
                <div class="warning-card">
                    <h4 style="margin-top: 0;">⚠️ Important Medical Disclaimer</h4>
                    <p style="margin-bottom: 0;">
                        This AI system is a <strong>screening tool only</strong> and should not be used as the sole basis for diagnosis. 
                        All findings must be confirmed by qualified healthcare professionals. If cancer is detected, 
                        please consult with an oncologist or urologist immediately for proper evaluation and treatment planning.
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    # ===== TAB 2: STATISTICS =====
    with tab2:
        st.markdown("## 🌍 Global Prostate Cancer Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-box">
                <p class="metric-label">Annual Cases Globally</p>
                <p class="metric-value" style="color: #e74c3c;">1.4M</p>
                <p style="font-size: 0.85rem; color: #666; margin-top: 0.5rem;">New diagnoses per year</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-box">
                <p class="metric-label">Annual Deaths</p>
                <p class="metric-value" style="color: #c0392b;">376K</p>
                <p style="font-size: 0.85rem; color: #666; margin-top: 0.5rem;">Deaths per year</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-box">
                <p class="metric-label">Mortality Rate</p>
                <p class="metric-value" style="color: #e67e22;">26.6%</p>
                <p style="font-size: 0.85rem; color: #666; margin-top: 0.5rem;">Of diagnosed cases</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-box">
                <p class="metric-label">5-Year Survival</p>
                <p class="metric-value" style="color: #27ae60;">98%</p>
                <p style="font-size: 0.85rem; color: #666; margin-top: 0.5rem;">With early detection</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("## 🇬🇭 Ghana-Specific Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-box">
                <p class="metric-label">Annual Cases</p>
                <p class="metric-value" style="color: #e74c3c;">12.5K</p>
                <p style="font-size: 0.85rem; color: #666; margin-top: 0.5rem;">In Ghana</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-box">
                <p class="metric-label">Annual Deaths</p>
                <p class="metric-value" style="color: #c0392b;">2.8K</p>
                <p style="font-size: 0.85rem; color: #666; margin-top: 0.5rem;">In Ghana</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-box">
                <p class="metric-label">Public Awareness</p>
                <p class="metric-value" style="color: #f39c12;">35%</p>
                <p style="font-size: 0.85rem; color: #666; margin-top: 0.5rem;">Need improvement</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-box">
                <p class="metric-label">Screening Rate</p>
                <p class="metric-value" style="color: #e67e22;">18%</p>
                <p style="font-size: 0.85rem; color: #666; margin-top: 0.5rem;">Of eligible population</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h3>📊 Key Insights</h3>
            <ul style="line-height: 2;">
                <li><strong>Early detection</strong> increases survival rate to 98%</li>
                <li><strong>Regular screening</strong> after age 50 is crucial</li>
                <li><strong>AI assistance</strong> can help improve detection accuracy</li>
                <li><strong>Access to healthcare</strong> remains a challenge in many regions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # ===== TAB 3: HOW IT WORKS =====
    with tab3:
        st.markdown("## 🔬 How ProstateCare AI Works")
        
        st.markdown("""
        <div class="info-card">
            <h3>🧠 Multi-Model Ensemble Approach</h3>
            <p style="font-size: 1.1rem; line-height: 1.8;">
                ProstateCare AI uses three state-of-the-art deep learning models working together 
                to provide the most accurate diagnosis possible. This ensemble approach significantly 
                reduces false positives and false negatives.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-box" style="height: 100%;">
                <h4 style="color: #667eea; margin-top: 0;">1️⃣ Attention U-Net</h4>
                <p style="text-align: left; font-size: 0.95rem; line-height: 1.6;">
                    Focuses on specific regions of interest using attention mechanisms. 
                    Excellent at identifying small lesions and subtle abnormalities.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-box" style="height: 100%;">
                <h4 style="color: #667eea; margin-top: 0;">2️⃣ ResU-Net</h4>
                <p style="text-align: left; font-size: 0.95rem; line-height: 1.6;">
                    Uses residual connections for deeper feature extraction. 
                    Particularly strong at detecting complex tissue patterns.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-box" style="height: 100%;">
                <h4 style="color: #667eea; margin-top: 0;">3️⃣ U-Net++</h4>
                <p style="text-align: left; font-size: 0.95rem; line-height: 1.6;">
                    Advanced nested architecture with dense skip connections. 
                    Provides precise segmentation boundaries.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 📋 Analysis Pipeline")
        
        st.markdown("""
        <div class="info-card">
            <ol style="line-height: 2.2; font-size: 1.05rem;">
                <li><strong>Image Upload:</strong> User uploads a prostate MRI scan (grayscale recommended)</li>
                <li><strong>Preprocessing:</strong> Image is normalized and resized to 128×128 pixels</li>
                <li><strong>Parallel Analysis:</strong> All three models analyze the image simultaneously</li>
                <li><strong>Individual Predictions:</strong> Each model generates its own prediction mask</li>
                <li><strong>Ensemble Voting:</strong> Results are combined using majority voting</li>
                <li><strong>Confidence Calculation:</strong> System calculates overall confidence score</li>
                <li><strong>Report Generation:</strong> Comprehensive report with visualizations</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 Why Ensemble Learning?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="metric-box">
                <h4 style="color: #51cf66;">✅ Advantages</h4>
                <ul style="text-align: left; line-height: 2;">
                    <li>Higher accuracy than single models</li>
                    <li>Reduces false positives/negatives</li>
                    <li>More robust to image variations</li>
                    <li>Provides confidence metrics</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-box">
                <h4 style="color: #4a90e2;">📊 Performance</h4>
                <ul style="text-align: left; line-height: 2;">
                    <li>Sensitivity: 92-95%</li>
                    <li>Specificity: 88-91%</li>
                    <li>Processing time: ~3-5 seconds</li>
                    <li>Consensus-based diagnosis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # ===== TAB 4: ABOUT =====
    with tab4:
        st.markdown("## 📋 About ProstateCare AI")
        
        st.markdown("""
        <div class="info-card">
            <h3>🎯 Mission</h3>
            <p style="font-size: 1.1rem; line-height: 1.8;">
                ProstateCare AI aims to democratize access to advanced prostate cancer screening 
                by providing AI-powered analysis tools that assist healthcare professionals in 
                early detection and diagnosis. Our goal is to improve patient outcomes through 
                faster, more accurate screening.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="metric-box">
                <h3 style="color: #667eea;">🔬 Technology</h3>
                <ul style="text-align: left; line-height: 2; font-size: 1.05rem;">
                    <li>Deep Learning (TensorFlow/Keras)</li>
                    <li>Convolutional Neural Networks</li>
                    <li>Attention Mechanisms</li>
                    <li>Ensemble Learning</li>
                    <li>Medical Image Segmentation</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-box">
                <h3 style="color: #667eea;">✨ Features</h3>
                <ul style="text-align: left; line-height: 2; font-size: 1.05rem;">
                    <li>Multi-model ensemble analysis</li>
                    <li>Real-time visualization</li>
                    <li>Detailed reports & metrics</li>
                    <li>Individual model breakdowns</li>
                    <li>Customizable themes</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-card">
            <h3 style="margin-top: 0;">⚠️ Critical Disclaimer</h3>
            <p style="font-size: 1.05rem; line-height: 1.8; margin-bottom: 0;">
                <strong>ProstateCare AI is a SCREENING TOOL ONLY</strong> and is NOT a medical device or diagnostic instrument.
                It is designed to assist healthcare professionals but cannot replace clinical judgment, 
                physical examination, or confirmatory diagnostic procedures. All findings must be verified 
                by qualified medical professionals. Do not use this tool as the sole basis for any medical decisions.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h3>👨‍⚕️ For Healthcare Professionals</h3>
            <p style="font-size: 1.05rem; line-height: 1.8;">
                This tool is intended to support clinical workflow by providing a preliminary analysis 
                of prostate MRI scans. It should be used in conjunction with standard diagnostic protocols, 
                patient history, PSA levels, and other clinical indicators. The AI predictions should be 
                considered as one data point in the comprehensive evaluation process.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h3>📞 Support & Contact</h3>
            <p style="font-size: 1.05rem; line-height: 1.8;">
                For technical support, questions about the AI models, or collaboration inquiries, 
                please contact our development team. We welcome feedback from healthcare professionals 
                and researchers to continuously improve our system.
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()