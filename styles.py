import streamlit as st

_CSS = """
<style>
    .stApp { background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #0a0a1a 100%); }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d1f 0%, #111128 60%, #0d1030 100%);
        border-right: 1px solid rgba(233,69,96,0.2);
    }
    [data-testid="stSidebar"] .stRadio label { color: #c8c8e0; font-size: 15px; }

    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(30,30,50,0.85), rgba(45,45,70,0.85));
        border: 1px solid rgba(233,69,96,0.25);
        border-radius: 14px; padding: 18px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
    }
    [data-testid="stMetric"] label { color: #9090b8 !important; font-size: 14px !important; letter-spacing: 0.5px; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 28px !important; font-weight: 800 !important; color: white !important; }

    h1 {
        background: linear-gradient(90deg, #e94560 0%, #f97316 50%, #e94560 100%);
        background-size: 200% auto;
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 900 !important; font-size: 2.2rem !important;
        animation: shimmer 3s linear infinite;
    }
    @keyframes shimmer { to { background-position: 200% center; } }
    h2 { color: #e94560 !important; border-bottom: 1px solid rgba(233,69,96,0.25); padding-bottom: 8px; }
    h3 { color: #f0a0b0 !important; }

    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #e94560, #c23152);
        border: none; border-radius: 10px; padding: 12px 30px;
        font-weight: 700; font-size: 16px;
        box-shadow: 0 4px 15px rgba(233,69,96,0.3);
        transition: all 0.3s;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(233,69,96,0.5);
    }

    .glass-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px; padding: 22px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin: 8px 0;
    }
    .metric-explain {
        background: linear-gradient(135deg, rgba(20,20,40,0.9), rgba(30,30,55,0.9));
        border: 1px solid rgba(233,69,96,0.22);
        border-radius: 14px; padding: 18px 20px; margin: 6px 0;
    }
    .metric-explain .mt { color: #e94560; font-weight: 700; font-size: 13px; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 8px; }
    .metric-explain .mv { color: white; font-size: 32px; font-weight: 900; margin-bottom: 6px; }
    .metric-explain .md { color: #9898b8; font-size: 14px; line-height: 1.6; }
    .metric-explain .bar-bg { height: 4px; border-radius: 2px; background: rgba(255,255,255,0.08); margin-top: 10px; }
    .metric-explain .bar-fill { height: 100%; border-radius: 2px; background: linear-gradient(90deg, #e94560, #f97316); }

    .step-card {
        background: linear-gradient(135deg, rgba(25,25,50,0.9), rgba(35,35,65,0.9));
        border: 1px solid rgba(233,69,96,0.18);
        border-radius: 14px; padding: 18px; text-align: center;
    }
    .step-num {
        background: linear-gradient(135deg, #e94560, #c23152);
        color: white; width: 34px; height: 34px; border-radius: 50%;
        display: inline-flex; align-items: center; justify-content: center;
        font-weight: 900; font-size: 15px; margin-bottom: 10px;
    }

    .risk-high {
        background: linear-gradient(135deg, rgba(45,20,20,0.9), rgba(60,25,25,0.9));
        border-left: 4px solid #e74c3c; border-radius: 14px; padding: 20px; margin: 10px 0;
        box-shadow: 0 4px 20px rgba(231,76,60,0.12);
    }
    .risk-medium {
        background: linear-gradient(135deg, rgba(45,40,20,0.9), rgba(60,55,25,0.9));
        border-left: 4px solid #f39c12; border-radius: 14px; padding: 20px; margin: 10px 0;
        box-shadow: 0 4px 20px rgba(243,156,18,0.12);
    }
    .risk-low {
        background: linear-gradient(135deg, rgba(20,40,20,0.9), rgba(25,55,25,0.9));
        border-left: 4px solid #2ecc71; border-radius: 14px; padding: 20px; margin: 10px 0;
        box-shadow: 0 4px 20px rgba(46,204,113,0.12);
    }
    .info-card {
        background: linear-gradient(135deg, rgba(20,20,40,0.9), rgba(30,30,55,0.9));
        border-radius: 14px; padding: 20px; border-left: 4px solid #e94560; margin: 10px 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }

    hr { border-color: rgba(233,69,96,0.18) !important; }
    .streamlit-expanderHeader { color: #e94560 !important; font-weight: 600 !important; font-size: 15px !important; }

    .stMarkdown p { font-size: 15px; line-height: 1.7; }
    .stMarkdown li { font-size: 15px; }
    div[data-testid="stCaptionContainer"] p { font-size: 14px !important; }
</style>
"""


def aplicar_estilos():
    st.markdown(_CSS, unsafe_allow_html=True)
