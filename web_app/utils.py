import os
import torch
import pickle
import streamlit as st
import numpy as np

from src.models.nn_model import OperatorStrengthNN
from src.features.feature_extractor import OperatorFeatureExtractor

# Operator metadata containing exact subclass definitions and expanded limits (up to 350,000 damage, 6000 dps)
PROFESSION_METADATA = {
    "Vanguard": {
        "label": "先锋",
        "desc": "战术支点与部署费用（DP）回复核心，追求落地极速点火",
        "branches": ["尖兵", "冲锋手", "战术家", "情报官", "执旗手"],
        "avg_stats": {
            "atk": 500, "def": 375, "res": 0, "dps": 500, "dph": 600,
            "control_coverage": 0.10, "buff_amp": 1.10, "total_damage": 15000, "sp_cost": 30, "init_sp": 10
        },
        "limits": {
            "atk": (300, 1000), "def": (250, 550), "res": (0, 15), "dps": (100, 2000), "dph": (200, 8000),
            "control_coverage": (0.0, 0.5), "buff_amp": (1.0, 1.5), "total_damage": (0, 100000), "sp_cost": (5, 60), "init_sp": (0, 45)
        }
    },
    "Guard": {
        "label": "近卫",
        "desc": "核心输出手，具有极高单兵爆发或站场技能总伤",
        "branches": ["强攻手", "领主", "无畏者", "剑豪", "术战者", "收割者", "解放者"],
        "avg_stats": {
            "atk": 1100, "def": 400, "res": 5, "dps": 1500, "dph": 1800,
            "control_coverage": 0.08, "buff_amp": 1.10, "total_damage": 75000, "sp_cost": 45, "init_sp": 15
        },
        "limits": {
            "atk": (500, 2000), "def": (200, 650), "res": (0, 25), "dps": (300, 5000), "dph": (400, 15000),
            "control_coverage": (0.0, 0.6), "buff_amp": (1.0, 1.5), "total_damage": (10000, 300000), "sp_cost": (5, 90), "init_sp": (0, 70)
        }
    },
    "Defender": {
        "label": "重装",
        "desc": "阵线铁壁，依靠防御阻挡，部分分支带有强力控场技能",
        "branches": ["守护者", "决战铁卫", "驭法铁卫", "不屈者", "要塞"],
        "avg_stats": {
            "atk": 650, "def": 900, "res": 10, "dps": 650, "dph": 850,
            "control_coverage": 0.35, "buff_amp": 1.15, "total_damage": 25000, "sp_cost": 35, "init_sp": 10
        },
        "limits": {
            "atk": (300, 1500), "def": (500, 1400), "res": (0, 35), "dps": (50, 3000), "dph": (100, 10000),
            "control_coverage": (0.0, 1.0), "buff_amp": (1.0, 2.0), "total_damage": (0, 150000), "sp_cost": (5, 75), "init_sp": (0, 50)
        }
    },
    "Sniper": {
        "label": "狙击",
        "desc": "高频或重炮远程物理输出者，以单次循环总伤见长",
        "branches": ["速射手", "轰击手", "投掷手", "重射手", "战术射手", "散布手"],
        "avg_stats": {
            "atk": 950, "def": 170, "res": 5, "dps": 1400, "dph": 1500,
            "control_coverage": 0.05, "buff_amp": 1.05, "total_damage": 60000, "sp_cost": 40, "init_sp": 15
        },
        "limits": {
            "atk": (400, 2200), "def": (80, 300), "res": (0, 20), "dps": (400, 6000), "dph": (300, 15000),
            "control_coverage": (0.0, 0.4), "buff_amp": (1.0, 1.3), "total_damage": (10000, 350000), "sp_cost": (5, 80), "init_sp": (0, 60)
        }
    },
    "Caster": {
        "label": "术师",
        "desc": "法术爆发伤害中枢，完全忽略 DPH 的物理抗性稀释",
        "branches": ["中坚术师", "扩散术师", "阵法术师", "驭械术师", "轰击术师", "秘术师"],
        "avg_stats": {
            "atk": 1050, "def": 150, "res": 22, "dps": 1100, "dph": 1750,
            "control_coverage": 0.20, "buff_amp": 1.10, "total_damage": 60000, "sp_cost": 50, "init_sp": 20
        },
        "limits": {
            "atk": (500, 2200), "def": (80, 250), "res": (10, 40), "dps": (300, 5000), "dph": (500, 15000),
            "control_coverage": (0.0, 0.8), "buff_amp": (1.0, 1.4), "total_damage": (10000, 300000), "sp_cost": (10, 90), "init_sp": (0, 70)
        }
    },
    "Medic": {
        "label": "医疗",
        "desc": "生存保障核心，部分强力干员带有强大的属性增伤或元素治疗",
        "branches": ["医师", "群愈师", "疗养师", "行医", "咒愈师"],
        "avg_stats": {
            "atk": 625, "def": 190, "res": 10, "dps": 600, "dph": 750,
            "control_coverage": 0.05, "buff_amp": 1.30, "total_damage": 37500, "sp_cost": 30, "init_sp": 10
        },
        "limits": {
            "atk": (300, 1500), "def": (80, 350), "res": (0, 30), "dps": (100, 2500), "dph": (150, 10000),
            "control_coverage": (0.0, 0.3), "buff_amp": (1.0, 2.0), "total_damage": (0, 150000), "sp_cost": (5, 70), "init_sp": (0, 50)
        }
    },
    "Supporter": {
        "label": "辅助",
        "desc": "战术增益核心，强度完全依赖增伤幅度 (Buff Amp) 和控场覆率 (CC)",
        "branches": ["凝滞师", "削弱者", "吟游诗人", "召唤师", "护佑者", "工匠"],
        "avg_stats": {
            "atk": 625, "def": 200, "res": 20, "dps": 700, "dph": 900,
            "control_coverage": 0.55, "buff_amp": 1.70, "total_damage": 25000, "sp_cost": 35, "init_sp": 15
        },
        "limits": {
            "atk": (300, 1500), "def": (80, 350), "res": (10, 35), "dps": (100, 3000), "dph": (200, 10000),
            "control_coverage": (0.0, 1.0), "buff_amp": (1.0, 2.5), "total_damage": (0, 150000), "sp_cost": (5, 80), "init_sp": (0, 60)
        }
    },
    "Specialist": {
        "label": "特种",
        "desc": "战术多面手，依靠极短 of SP 消耗或超强的落地爆发控场取胜",
        "branches": ["处决者", "推击手", "钩手", "行商", "陷阱师", "傀儡师", "伏击客"],
        "avg_stats": {
            "atk": 825, "def": 325, "res": 10, "dps": 1400, "dph": 1550,
            "control_coverage": 0.45, "buff_amp": 1.20, "total_damage": 50000, "sp_cost": 20, "init_sp": 8
        },
        "limits": {
            "atk": (400, 1800), "def": (150, 550), "res": (0, 30), "dps": (200, 5000), "dph": (300, 15000),
            "control_coverage": (0.0, 1.0), "buff_amp": (1.0, 1.8), "total_damage": (5000, 250000), "sp_cost": (1, 50), "init_sp": (0, 35)
        }
    }
}

@st.cache_resource
def load_model_and_preprocessor(profession):
    """
    Loads and caches the PyTorch model and preprocessor (Scaler/Encoder) 
    for the specified profession.
    """
    model_path = f"models/{profession.lower()}_model.pth"
    preprocessor_path = f"models/{profession.lower()}_preprocessor.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        return None, None
        
    try:
        # Load Preprocessor
        extractor = OperatorFeatureExtractor.load(preprocessor_path)
        
        # Load PyTorch model
        model = OperatorStrengthNN(input_dim=extractor.get_input_dim())
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        
        return model, extractor
    except Exception as e:
        st.error(f"Error loading model/preprocessor for {profession}: {str(e)}")
        return None, None

def inject_arknights_style():
    """
    Injects custom CSS to give the Streamlit app a premium techy dark Arknights (Rhodes Island HUD) style.
    """
    st.markdown("""
        <style>
        /* Base styles */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Noto+Sans+SC:wght@300;400;700;900&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', 'Noto Sans SC', sans-serif;
            background-color: #0b0e14;
            color: #d1d5db;
        }
        
        /* App background override */
        .stApp {
            background: linear-gradient(135deg, #0b0e14 0%, #111622 100%);
        }
        
        /* Glassmorphic Cards */
        .glass-card {
            background: rgba(25, 33, 49, 0.45);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-left: 4px solid #ff9f1c; /* Rhodes Amber indicator */
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
        }
        
        /* Glowing badges */
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .badge-s {
            background-color: rgba(239, 68, 68, 0.2);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.4);
            box-shadow: 0 0 15px rgba(239, 68, 68, 0.2);
            text-shadow: 0 0 5px rgba(239, 68, 68, 0.5);
        }
        .badge-a {
            background-color: rgba(245, 158, 11, 0.2);
            color: #f59e0b;
            border: 1px solid rgba(245, 158, 11, 0.4);
            box-shadow: 0 0 15px rgba(245, 158, 11, 0.2);
        }
        .badge-b {
            background-color: rgba(59, 130, 246, 0.2);
            color: #3b82f6;
            border: 1px solid rgba(59, 130, 246, 0.4);
        }
        .badge-c {
            background-color: rgba(156, 163, 175, 0.2);
            color: #9ca3af;
            border: 1px solid rgba(156, 163, 175, 0.4);
        }
        
        /* Interactive element adjustments */
        div[data-baseweb="select"] {
            background-color: #121824 !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
        }
        
        /* Style Slider Label and values */
        .stSlider label {
            color: #9ca3af !important;
            font-weight: 600;
        }
        
        /* Custom title styling */
        .rhodes-title {
            font-weight: 900;
            background: linear-gradient(45deg, #ff9f1c 0%, #f77f00 50%, #ffffff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
            text-transform: uppercase;
            border-bottom: 2px solid rgba(255, 159, 28, 0.3);
            padding-bottom: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .rhodes-subtitle {
            color: #9ca3af;
            font-size: 14px;
            text-align: center;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            margin-top: -15px;
            margin-bottom: 30px;
        }
        </style>
    """, unsafe_allow_html=True)
