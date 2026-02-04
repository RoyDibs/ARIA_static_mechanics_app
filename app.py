import streamlit as st
import sys
from pathlib import Path
import os
import json
from datetime import datetime
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from scripts.teaching_assistant import StaticsMechanicsTA
from scripts.database.feedback_storage import feedback_storage
from scripts.database.conversation_storage import conversation_storage
from scripts.database.supabase_config import supabase_config

# Get API key from environment variables or Streamlit secrets
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        st.error("OpenAI API key not found. Please set it in Streamlit secrets or environment variables.")
        st.stop()

# Page configuration
st.set_page_config(
    page_title="ARIA: Teaching Assistant for Statics and Mechanics of Materials (EN.560.201)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Make the toggle a clean icon button and hide any inner text */
button[data-testid="collapsedControl"]{
  position: relative;
  width: 36px !important;
  height: 36px !important;
  padding: 4px !important;
  border: 1px solid #26272b;
  border-radius: 10px;
  background: #0e0f12;
  cursor: pointer;
}

/* Hide whatever Streamlit puts inside (e.g., 'keyboard_double_arrow_right') */
button[data-testid="collapsedControl"] *{
  opacity: 0 !important;
  font-size: 0 !important;
  line-height: 0 !important;
}

/* Draw a crisp double-chevron icon with an inline SVG mask */
button[data-testid="collapsedControl"]::after{
  content: "";
  position: absolute;
  inset: 0;
  margin: auto;
  width: 22px;
  height: 22px;

  background-color: #d4d4d6;
  -webkit-mask: url("data:image/svg+xml;utf8,\
<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>\
  <path d='M8.59 16.59 13.17 12 8.59 7.41 10 6l6 6-6 6z'/>\
  <path d='M4.59 16.59 9.17 12 4.59 7.41 6 6l6 6-6 6z'/>\
</svg>") no-repeat center / contain;
          mask: url("data:image/svg+xml;utf8,\
<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>\
  <path d='M8.59 16.59 13.17 12 8.59 7.41 10 6l6 6-6 6z'/>\
  <path d='M4.59 16.59 9.17 12 4.59 7.41 6 6l6 6-6 6z'/>\
</svg>") no-repeat center / contain;
}

/* Hover and focus states */
button[data-testid="collapsedControl"]:hover::after{ background-color: #ffffff; }
button[data-testid="collapsedControl"]:focus{ outline: 2px solid #6aa6ff; outline-offset: 2px; }
</style>
""", unsafe_allow_html=True)

# Design CSS with Cambria font and yellowish-brown color scheme
st.markdown("""
<style>
:root{
  --bg:#F5F1E8 !important; --panel:#FEFCF7 !important; --panel-2:#F9F5EC !important;
  --text:#3C2E1E !important; --muted:#8B7355 !important; --border:#D4A574 !important;
  --accent:#B8956A !important; --accent-2:#A0845C;
  --accent-hover:#8B7355 !important; --callout-bg:#F0E6D2 !important;
  --claude-purple:#a47aff !important;
  
  /* Override button colors - #D68C45 for all modes */
  --claude-btn-primary:#D68C45 !important;
  --claude-btn-primary-hover:#C17B3A !important;
  --claude-btn-primary-active:#A66A30 !important;
  --claude-btn-text:#FFFFFF !important;
  --claude-btn-border:#D68C45 !important;
  --claude-btn-focus:#8B7355 !important;
}

/* Override button colors - #D68C45 for dark mode */
[data-theme="dark"] {
  --claude-btn-primary:#D68C45 !important;
  --claude-btn-primary-hover:#C17B3A !important;
  --claude-btn-primary-active:#A66A30 !important;
  --claude-btn-text:#FFFFFF !important;
  --claude-btn-border:#D68C45 !important;
  --claude-btn-focus:#8B7355 !important;
}

/* Apply Cambria globally, including code blocks */
html, body, .stApp, .main, .block-container,
h1,h2,h3,h4,h5,h6,
p,div,span,label,li,small,em,strong,
button, input, textarea, select,
code, pre, kbd, samp {
  font-family: "Cambria", "Times New Roman", serif !important;
  -webkit-font-smoothing: antialiased !important;
  -moz-osx-font-smoothing: grayscale !important;
  color: var(--text);
}

.stApp{ background-color:var(--bg); }

.main .block-container{ max-width:960px; padding:2rem 1.5rem; background:var(--bg); }

/* Centered larger app title */
.app-title{
  text-align:center;
  font-weight:700;
  line-height:1.2;
  margin:0 0 1rem 0;
  letter-spacing:.2px;
  font-size:clamp(3.5rem, 4vw + 2.5rem, 5rem);
}

/* Headings scale */
h2{font-size:1.8rem !important}
h3{font-size:1.4rem !important}

/* Sidebar */
[data-testid="stSidebar"]{
  background:var(--panel); border-right:1px solid var(--border); box-shadow:2px 0 8px rgba(0,0,0,.25); padding-top:.5rem;
}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{
  border-bottom:1px solid var(--border); padding-bottom:.5rem; margin-bottom:.75rem;
}

/* Inputs */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea{
  background:var(--panel-2); color:var(--text) !important; border:1px solid var(--border) !important;
  border-radius:12px; padding:.75rem .9rem; transition:border .15s ease, box-shadow .15s ease;
}

/* Clean selectbox button - No styling */
.stSelectbox > div > div > select{
  background: transparent !important; 
  color: var(--text) !important; 
  border: none !important;
  border-radius: 0 !important;
  padding: 0.5rem 0 !important; 
  box-shadow: none !important;
  -webkit-appearance: none !important;
  -moz-appearance: none !important;
  appearance: none !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus{
  border-color:var(--accent) !important; box-shadow:0 0 0 3px rgba(106,166,255,.18) !important; outline:none;
}

/* Clean selectbox focus - No styling */
.stSelectbox > div > div > select:focus{
  border: none !important; 
  box-shadow: none !important; 
  outline: none !important;
}

/* General Buttons - Enhanced visibility */
.stButton > button{
  background:var(--panel-2) !important; 
  color:var(--text) !important; 
  border:2px solid var(--border) !important;
  border-radius:8px !important; 
  padding:.7rem 1.1rem !important; 
  font-weight:700 !important;
  font-size:0.95rem !important;
  transition:all .2s ease !important;
  box-shadow:0 1px 3px rgba(0,0,0,0.1) !important;
}
.stButton > button:hover{ 
  background:var(--accent-hover) !important; 
  border-color:var(--accent) !important; 
  transform:translateY(-1px) !important; 
  color:white !important;
  box-shadow:0 2px 6px rgba(0,0,0,0.15) !important;
}

/* Clear Conversation button specific styling */
[data-testid="stSidebar"] .stButton > button {
  background:var(--accent-2) !important;
  color:var(--text) !important;
  border:2px solid var(--accent) !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background:var(--accent) !important;
  color:white !important;
}

/* Focus Area Selectbox - Clean, borderless design */
.stSelectbox > div > div > select{
  background: transparent !important; 
  color: var(--text) !important; 
  border: none !important;
  border-radius: 0 !important;
  font-weight: 500 !important;
  font-size: 1rem !important;
  padding: 0.5rem 0 !important;
  -webkit-appearance: none !important;
  -moz-appearance: none !important;
  appearance: none !important;
  box-shadow: none !important;
}
.stSelectbox > div > div > select:focus{
  border: none !important; 
  box-shadow: none !important;
  outline: none !important;
}

/* Remove custom dropdown arrow */
.stSelectbox > div > div {
  position: relative !important;
}
.stSelectbox > div > div::after {
  display: none !important;
}

/* Dropdown menu options styling */
.stSelectbox > div > div > select option {
  background:var(--panel) !important;
  color:var(--text) !important;
  padding:8px 12px !important;
  font-weight:500 !important;
  border:none !important;
}
.stSelectbox > div > div > select option:hover,
.stSelectbox > div > div > select option:focus,
.stSelectbox > div > div > select option:checked {
  background:var(--accent) !important;
  color:white !important;
}

/* Clean selectbox widget styling - No borders or backgrounds */
[data-baseweb="select"] {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
[data-baseweb="select"] > div {
  background: transparent !important;
  color: var(--text) !important;
  border: none !important;
  border-radius: 0 !important;
  box-shadow: none !important;
}

/* Clean dropdown popover container - Single border design */
[data-baseweb="popover"] {
  background: #FFFFFF !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  box-shadow: 0 4px 16px rgba(0,0,0,0.15) !important;
  z-index: 99999 !important;
  max-height: 300px !important;
  overflow-y: auto !important;
  padding: 0 !important;
}

/* Remove borders from child elements */
[data-baseweb="popover"] > div,
[data-baseweb="popover"] ul {
  background: transparent !important;
  border: none !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  padding: 0 !important;
  margin: 0 !important;
}

/* Clean dropdown menu container */
[data-baseweb="menu"],
[data-baseweb="menu"] > div,
[data-baseweb="menu"] ul {
  background: transparent !important;
  border: none !important;
  border-radius: 0 !important;
  padding: 8px !important;
  margin: 0 !important;
}

/* Clean menu items - No individual borders */
[data-baseweb="menu"] li,
[data-baseweb="menu"] li > div,
[data-baseweb="menu"] li span,
[data-baseweb="menu"] [role="option"],
[data-baseweb="menu"] [role="option"] > div,
[data-baseweb="menu"] [role="option"] span {
  background: transparent !important;
  color: #2C1810 !important;
  padding: 12px 16px !important;
  margin: 0 !important;
  border-radius: 8px !important;
  font-weight: 500 !important;
  font-size: 1rem !important;
  border: none !important;
  transition: all 0.15s ease !important;
  line-height: 1.4 !important;
  min-height: 44px !important;
  display: flex !important;
  align-items: center !important;
}

/* Clean menu item hover state */
[data-baseweb="menu"] li:hover,
[data-baseweb="menu"] li:hover > div,
[data-baseweb="menu"] li:hover span,
[data-baseweb="menu"] [role="option"]:hover,
[data-baseweb="menu"] [role="option"]:hover > div,
[data-baseweb="menu"] [role="option"]:hover span {
  background: var(--accent) !important;
  color: #FFFFFF !important;
  transform: none !important;
  box-shadow: none !important;
}

/* Clean menu item selected/active state */
[data-baseweb="menu"] li[aria-selected="true"],
[data-baseweb="menu"] li[aria-selected="true"] > div,
[data-baseweb="menu"] li[aria-selected="true"] span,
[data-baseweb="menu"] [role="option"][aria-selected="true"],
[data-baseweb="menu"] [role="option"][aria-selected="true"] > div,
[data-baseweb="menu"] [role="option"][aria-selected="true"] span {
  background: var(--accent-2) !important;
  color: #2C1810 !important;
  font-weight: 600 !important;
  box-shadow: none !important;
}

/* Clean fallback styling for compatibility - single border */
.stSelectbox [role="listbox"] {
  background: #FFFFFF !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  box-shadow: 0 4px 16px rgba(0,0,0,0.15) !important;
  z-index: 99999 !important;
  position: relative !important;
  padding: 8px !important;
}

/* Remove borders from fallback child elements */
.stSelectbox [role="listbox"] > div {
  background: transparent !important;
  border: none !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  padding: 0 !important;
}

.stSelectbox [role="option"],
.stSelectbox [role="option"] > div,
.stSelectbox [role="option"] span {
  background: transparent !important;
  color: #2C1810 !important;
  padding: 12px 16px !important;
  border-radius: 8px !important;
  margin: 0 !important;
  font-weight: 500 !important;
  border: none !important;
  min-height: 44px !important;
  display: flex !important;
  align-items: center !important;
  transition: all 0.15s ease !important;
}

.stSelectbox [role="option"]:hover,
.stSelectbox [role="option"]:hover > div,
.stSelectbox [role="option"]:hover span,
.stSelectbox [role="option"][aria-selected="true"],
.stSelectbox [role="option"][aria-selected="true"] > div,
.stSelectbox [role="option"][aria-selected="true"] span {
  background: var(--accent) !important;
  color: #FFFFFF !important;
}

/* Comprehensive text visibility enforcement */
.stSelectbox [data-baseweb] *,
.stSelectbox [role="option"] *,
.stSelectbox [role="listbox"] * {
  color: inherit !important;
}

/* Positioning and layout fixes */
[data-baseweb="popover"] {
  position: fixed !important;
  top: auto !important;
  left: auto !important;
  right: auto !important;
  bottom: auto !important;
  z-index: 99999 !important;
}

/* Ensure dropdown appears below the selectbox */
.stSelectbox > div > div[data-baseweb="select"] {
  position: relative !important;
}

/* Force dropdown to be visible and properly positioned */
.stSelectbox [aria-expanded="true"] + [data-baseweb="popover"],
.stSelectbox [aria-expanded="true"] ~ [data-baseweb="popover"] {
  display: block !important;
  visibility: visible !important;
  opacity: 1 !important;
}

/* Clean responsive design for mobile devices */
@media (max-width: 768px) {
  [data-baseweb="popover"],
  [data-baseweb="popover"] > div,
  [data-baseweb="popover"] ul {
    max-width: calc(100vw - 32px) !important;
    left: 16px !important;
    right: 16px !important;
    border-radius: 16px !important;
  }
  
  [data-baseweb="menu"] li,
  [data-baseweb="menu"] [role="option"] {
    padding: 16px 20px !important;
    font-size: 1.1rem !important;
    min-height: 52px !important;
    border-radius: 10px !important;
  }
  
  .stSelectbox > div > div > select {
    font-size: 1.1rem !important;
    padding: 0.6rem 0 !important;
  }
  
  /* Enhanced Ask ARIA button for mobile */
  .stForm .stButton > button {
    padding: 1rem 2rem !important;
    font-size: 1.2rem !important;
    min-height: 56px !important;
    border-radius: 12px !important;
    width: 100% !important;
    max-width: 300px !important;
    margin: 0 auto !important;
  }
  
  /* Mobile focus state adjustments */
  .stForm .stButton > button:focus {
    outline-offset: 2px !important;
    box-shadow: 0 0 0 2px #FFFFFF, 0 0 0 4px #F0C896, 0 4px 12px rgba(0,0,0,0.25) !important;
  }
}

/* Tablet and medium screen optimizations */
@media (min-width: 769px) and (max-width: 1024px) {
  .stForm .stButton > button {
    padding: 0.95rem 1.7rem !important;
    font-size: 1.05rem !important;
    min-height: 50px !important;
  }
}

/* Large screen enhancements */
@media (min-width: 1200px) {
  .stForm .stButton > button {
    padding: 1rem 2rem !important;
    font-size: 1.15rem !important;
    min-height: 54px !important;
  }
}

/* Visual hierarchy improvements */
[data-baseweb="menu"] li:first-child,
[data-baseweb="menu"] [role="option"]:first-child {
  margin-top: 4px !important;
}

[data-baseweb="menu"] li:last-child,
[data-baseweb="menu"] [role="option"]:last-child {
  margin-bottom: 4px !important;
}

/* Subtle separator for better readability */
[data-baseweb="menu"] li:not(:last-child)::after,
[data-baseweb="menu"] [role="option"]:not(:last-child)::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 16px;
  right: 16px;
  height: 1px;
  background: rgba(0,0,0,0.05);
  pointer-events: none;
}

[data-baseweb="menu"] li:hover::after,
[data-baseweb="menu"] [role="option"]:hover::after {
  display: none;
}

/* High contrast mode support */
@media (prefers-contrast: high) {
  [data-baseweb="popover"],
  [data-baseweb="menu"] {
    border-width: 4px !important;
  }
  
  [data-baseweb="menu"] li,
  [data-baseweb="menu"] [role="option"] {
    border-width: 2px !important;
    font-weight: 700 !important;
  }
  
  /* Ask ARIA button matching Clear Conversation in high contrast */
  .stForm .stButton > button {
    border-width: 2px !important;
    font-weight: normal !important;
    background: #C4AA7A !important;
    color: var(--text) !important;
    border-color: #D4A574 !important;
  }
  
  .stForm .stButton > button:hover,
  .stForm .stButton > button:focus {
    background: #D4A574 !important;
    color: white !important;
    border-color: #D4A574 !important;
    outline: none !important;
    outline-offset: 0 !important;
  }
}

/* Reduced motion support */
@media (prefers-reduced-motion: reduce) {
  [data-baseweb="menu"] li,
  [data-baseweb="menu"] [role="option"],
  .stSelectbox [role="option"] {
    transition: none !important;
    transform: none !important;
  }
  
  [data-baseweb="menu"] li:hover,
  [data-baseweb="menu"] [role="option"]:hover {
    transform: none !important;
  }
  
  /* Reduced motion for Ask ARIA button */
  .stForm .stButton > button {
    transition: none !important;
    transform: none !important;
  }
  
  .stForm .stButton > button:hover,
  .stForm .stButton > button:focus,
  .stForm .stButton > button:active {
    transform: none !important;
    transition: none !important;
  }
}

/* Accessibility enhancements for keyboard navigation */
[data-baseweb="menu"] li:focus,
[data-baseweb="menu"] [role="option"]:focus,
.stSelectbox [role="option"]:focus {
  outline: 3px solid var(--accent) !important;
  outline-offset: 2px !important;
  background: var(--accent-2) !important;
  color: #2C1810 !important;
  border-color: var(--accent) !important;
}

/* Clean focus styling - minimal visual indication */
[data-baseweb="select"]:focus-within,
.stSelectbox > div > div > select:focus {
  outline: none !important;
  border: none !important;
  box-shadow: none !important;
}

/* Screen reader and accessibility improvements */
[data-baseweb="menu"] li,
[data-baseweb="menu"] [role="option"] {
  cursor: pointer !important;
  user-select: none !important;
  -webkit-tap-highlight-color: transparent !important;
}

/* Ensure proper contrast ratios for WCAG compliance */
[data-baseweb="menu"] li,
[data-baseweb="menu"] [role="option"],
.stSelectbox [role="option"] {
  color: #2C1810 !important;
  background: #FFFFFF !important;
}

[data-baseweb="menu"] li:hover,
[data-baseweb="menu"] li:focus,
[data-baseweb="menu"] [role="option"]:hover,
[data-baseweb="menu"] [role="option"]:focus,
.stSelectbox [role="option"]:hover,
.stSelectbox [role="option"]:focus {
  color: #FFFFFF !important;
  background: var(--accent) !important;
}

/* Force visibility for screen readers */
[data-baseweb="popover"][aria-hidden="true"] {
  display: none !important;
}

[data-baseweb="popover"][aria-hidden="false"] {
  display: block !important;
}

/* Ask ARIA button - Theme-adaptive colors for optimal contrast */
.stForm .stButton > button{
  background: #E6B885 !important;
  color: #2C1810 !important;
  border: 2px solid #D4A574 !important;
  font-weight: 700 !important;
  font-size: 0.95rem !important;
  border-radius: 8px !important;
  padding: 0.7rem 1.1rem !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
  transition: all 0.2s ease !important;
  text-transform: none !important;
  letter-spacing: normal !important;
  min-height: auto !important;
  cursor: pointer !important;
  position: relative !important;
  overflow: visible !important;
}

/* Hover state with theme-adaptive colors */
.stForm .stButton > button:hover{
  background: #D4A574 !important;
  color: white !important;
  border-color: #D4A574 !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 2px 6px rgba(0,0,0,0.15) !important;
}

/* Focus state with theme-adaptive colors */
.stForm .stButton > button:focus{
  outline: none !important;
  outline-offset: 0 !important;
  background: #D4A574 !important;
  color: white !important;
  border-color: #D4A574 !important;
  box-shadow: 0 2px 6px rgba(0,0,0,0.15) !important;
}

/* Active state with theme-adaptive colors */
.stForm .stButton > button:active{
  transform: translateY(-1px) !important;
  box-shadow: 0 2px 6px rgba(0,0,0,0.15) !important;
  background: #D4A574 !important;
  color: white !important;
}

/* Dark mode adaptation for Ask ARIA button */
@media (prefers-color-scheme: dark) {
  .stForm .stButton > button {
    background: #8B6914 !important;
    color: #F5F1E8 !important;
    border-color: #A67C00 !important;
  }
  
  .stForm .stButton > button:hover,
  .stForm .stButton > button:focus,
  .stForm .stButton > button:active {
    background: #A67C00 !important;
    color: #FFFFFF !important;
    border-color: #A67C00 !important;
  }
}

/* Disabled state with proper contrast */
.stForm .stButton > button:disabled{
  background: #D4D4D4 !important;
  color: #757575 !important;
  border-color: #D4D4D4 !important;
  cursor: not-allowed !important;
  transform: none !important;
  box-shadow: none !important;
}

/* Enhanced text area contrast for better visibility */
.stTextArea > div > div > textarea::placeholder{
  color:#2C1810 !important; opacity:0.8 !important;
  font-weight:500 !important;
}
.stTextArea > div > div > textarea{
  color:#2C1810 !important;
  font-weight:500 !important;
  background:#FFFFFF !important;
  border:2px solid var(--border) !important;
}

/* Single rail callout with yellowish-brown styling */
.aria-callout{
  position:relative; background:var(--callout-bg);
  border-radius:14px; padding:1.5rem 1.25rem; box-shadow:0 4px 12px rgba(180, 149, 106, 0.15), inset 0 0 0 1px rgba(180, 149, 106, 0.2);
  text-align: center; border: 1px solid var(--border);
}
.aria-callout::before{
  content:""; position:absolute; left:0; top:0; bottom:0; width:6px;
  background:var(--accent); border-top-left-radius:14px; border-bottom-left-radius:14px;
}
.aria-callout .welcome-main {
  font-size: 1.1rem; line-height: 1.4; margin-bottom: 0.75rem; color: var(--text);
}
.aria-callout .welcome-sub {
  font-size: 1rem; color: var(--muted); font-style: italic;
}

/* Messages */
.chat-message{
  padding:1.1rem 1.2rem; border-radius:12px; margin:1rem 0; border:1px solid var(--border);
  background:var(--panel); box-shadow:0 1px 6px rgba(0,0,0,.18); color:var(--text);
}
.student-message{ 
  background:var(--panel-2); border-left:4px solid var(--accent); 
}
.ta-message{ 
  background:var(--panel); border-left:4px solid var(--claude-purple); 
}

/* Enhanced Feedback buttons */
.feedback-container{
  display:flex; align-items:center; gap:0.5rem; margin-top:0.75rem; padding-top:0.75rem;
  border-top:1px solid var(--border); opacity:0.9;
}
.feedback-text{
  font-size:0.9rem; color:var(--text); margin-right:0.5rem; font-weight:600;
}
.feedback-button{
  background:#FFFFFF !important; 
  border:2px solid #D4A574 !important; 
  border-radius:8px !important;
  padding:0.5rem 1rem !important; 
  cursor:pointer !important; 
  transition:all 0.2s ease !important;
  color:#2C1810 !important; 
  font-size:0.85rem !important; 
  display:flex !important; 
  align-items:center !important; 
  gap:0.4rem !important;
  min-width:80px !important; 
  white-space:nowrap !important;
  font-weight:600 !important;
  box-shadow:0 2px 4px rgba(0,0,0,0.1) !important;
}
.feedback-button:hover{
  background:#F0E6D2 !important; 
  border-color:#D4A574 !important; 
  color:#2C1810 !important;
  transform:translateY(-1px) !important;
  box-shadow:0 4px 8px rgba(0,0,0,0.15) !important;
}
.feedback-button.selected{
  background:#D4A574 !important; 
  border-color:#D4A574 !important; 
  color:white !important;
  transform:scale(1.05) !important;
}
.feedback-button.selected.negative{
  background:#dc3545 !important; 
  border-color:#dc3545 !important;
}
.feedback-button.selected.partial{
  background:#ffa500 !important; 
  border-color:#ffa500 !important;
}

/* Dark mode adaptation for feedback buttons */
@media (prefers-color-scheme: dark) {
  .feedback-button {
    background: #3C2E1E !important;
    color: #F5F1E8 !important;
    border-color: #8B6914 !important;
  }
  
  .feedback-button:hover {
    background: #4A3626 !important;
    color: #F5F1E8 !important;
    border-color: #A67C00 !important;
  }
  
  .feedback-button.selected {
    background: #8B6914 !important;
    color: #FFFFFF !important;
    border-color: #8B6914 !important;
  }
}

/* Footer centered */
.app-footer{
  text-align:center;
  color:var(--muted);
  font-size:1rem;
  margin-top:1rem;
}

/* Code blocks keep CMU Serif but retain block styling */
pre{
  background:#0f1012 !important;
  border:1px solid var(--border) !important;
  border-radius:10px !important;
  padding:1rem !important;
  overflow-x:auto !important;
}

/* Navigation bar icons - White Share option for better visibility */
[data-testid="stHeader"] {
  background-color: rgba(0, 0, 0, 0.9) !important;
}
[data-testid="stHeader"] button {
  color: #FFFFFF !important;
}
[data-testid="stHeader"] svg {
  fill: #FFFFFF !important;
  color: #FFFFFF !important;
}
[data-testid="stHeader"] [data-testid="stHeaderActionElements"] button {
  color: #FFFFFF !important;
}
[data-testid="stHeader"] [data-testid="stHeaderActionElements"] svg {
  fill: #FFFFFF !important;
}

/* Dark mode preference - maintain yellowish-brown theme */
@media (prefers-color-scheme: dark){
  :root{
    --bg:#2A1F15; --panel:#3C2E1E; --panel-2:#342619;
    --text:#F5F1E8; --muted:#B8956A; --border:#5D4A37;
    --accent:#D4A574; --accent-2:#E6B885;
    --accent-hover:#F0C896; --callout-bg:#3C2E1E;
    --claude-purple:#a47aff;
    
    /* Override button colors - #D68C45 for dark mode */
    --claude-btn-primary:#D68C45;
    --claude-btn-primary-hover:#C17B3A;
    --claude-btn-primary-active:#A66A30;
    --claude-btn-text:#FFFFFF;
    --claude-btn-border:#D68C45;
    --claude-btn-focus:#8B7355;
  }
  .stApp{ background:var(--bg); color:var(--text); }
  .main .block-container{ background:var(--bg); }
  .stButton > button{ background:var(--panel-2); color:var(--text); }
  .aria-callout{ background:var(--callout-bg); }
  .chat-message{ background:var(--panel); color:var(--text); }
  .student-message{ background:var(--panel-2); }
  .ta-message{ background:var(--panel); }
  
  /* Ensure all text elements are visible in dark mode */
  h1, h2, h3, h4, h5, h6, p, div, span, label, li, small, em, strong {
    color: var(--text) !important;
  }
  
  /* Sidebar styling in dark mode */
  [data-testid="stSidebar"] {
    background: var(--panel) !important;
    color: var(--text) !important;
  }
  [data-testid="stSidebar"] * {
    color: var(--text) !important;
  }
  
  /* Form elements in dark mode */
  .stTextInput > div > div > input,
  .stTextArea > div > div > textarea {
    background: var(--panel-2) !important;
    color: var(--text) !important;
    border: 2px solid var(--border) !important;
    font-weight: 500 !important;
  }
  
  /* Enhanced text area placeholder in dark mode */
  .stTextArea > div > div > textarea::placeholder {
    color: var(--text) !important;
    opacity: 0.9 !important;
    font-weight: 500 !important;
  }
  
  /* Clean selectbox in dark mode */
  .stSelectbox > div > div > select {
    background: transparent !important;
    color: var(--text) !important;
    border: none !important;
  }
  
  /* Enhanced button styling for dark mode */
  .stButton > button {
    background: var(--panel-2) !important;
    color: var(--text) !important;
    border: 2px solid var(--border) !important;
  }
  
  /* Enhanced Ask ARIA button for dark mode with WCAG compliance */
  /* Claude button colors automatically handled by CSS variables in dark mode */
  
  /* Feedback buttons in dark mode */
  .feedback-button {
    background: #F5F1E8 !important;
    color: #2C1810 !important;
    border: 2px solid var(--border) !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
  }
  
  .feedback-button:hover {
    background: #E6D7C3 !important;
    color: #2C1810 !important;
    border-color: var(--accent) !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.25) !important;
  }
  
  /* Selectbox dropdown and options in dark mode */
  .stSelectbox > div > div::after {
    color: var(--text) !important;
  }
  
  /* Clean selectbox widget in dark mode - No styling */
  [data-baseweb="select"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
  }
  [data-baseweb="select"] > div {
    background: transparent !important;
    color: var(--text) !important;
    border: none !important;
    box-shadow: none !important;
  }
  
  /* Clean dark mode dropdown styling - single border */
  [data-baseweb="popover"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 6px 20px rgba(0,0,0,0.5) !important;
  }
  
  /* Remove borders from child elements in dark mode */
  [data-baseweb="popover"] > div,
  [data-baseweb="popover"] ul {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
  }
  
  /* Dark mode dropdown menu */
  [data-baseweb="menu"],
  [data-baseweb="menu"] > div,
  [data-baseweb="menu"] ul {
    background: transparent !important;
    border: none !important;
  }
  
  /* Clean dark mode menu items */
  [data-baseweb="menu"] li,
  [data-baseweb="menu"] li > div,
  [data-baseweb="menu"] li span,
  [data-baseweb="menu"] [role="option"],
  [data-baseweb="menu"] [role="option"] > div,
  [data-baseweb="menu"] [role="option"] span {
    background: transparent !important;
    color: var(--text) !important;
    border: none !important;
    font-weight: 500 !important;
  }
  
  /* Dark mode hover and focus states */
  [data-baseweb="menu"] li:hover,
  [data-baseweb="menu"] li:hover > div,
  [data-baseweb="menu"] li:hover span,
  [data-baseweb="menu"] li:focus,
  [data-baseweb="menu"] li:focus > div,
  [data-baseweb="menu"] li:focus span,
  [data-baseweb="menu"] [role="option"]:hover,
  [data-baseweb="menu"] [role="option"]:hover > div,
  [data-baseweb="menu"] [role="option"]:hover span,
  [data-baseweb="menu"] [role="option"]:focus,
  [data-baseweb="menu"] [role="option"]:focus > div,
  [data-baseweb="menu"] [role="option"]:focus span {
    background: var(--accent) !important;
    color: white !important;
  }
  
  /* Dark mode selected states */
  [data-baseweb="menu"] li[aria-selected="true"],
  [data-baseweb="menu"] li[aria-selected="true"] > div,
  [data-baseweb="menu"] li[aria-selected="true"] span,
  [data-baseweb="menu"] [role="option"][aria-selected="true"],
  [data-baseweb="menu"] [role="option"][aria-selected="true"] > div,
  [data-baseweb="menu"] [role="option"][aria-selected="true"] span {
    background: var(--accent-2) !important;
    color: var(--text) !important;
    font-weight: 600 !important;
  }
  
  /* Clean dark mode fallback styling - single border */
  .stSelectbox [role="listbox"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
  }
  
  /* Remove borders from fallback child elements in dark mode */
  .stSelectbox [role="listbox"] > div {
    background: transparent !important;
    border: none !important;
  }
  
  .stSelectbox [role="option"],
  .stSelectbox [role="option"] > div,
  .stSelectbox [role="option"] span {
    background: transparent !important;
    color: var(--text) !important;
    border: none !important;
  }
  
  /* Dark mode separator styling */
  [data-baseweb="menu"] li:not(:last-child)::after,
  [data-baseweb="menu"] [role="option"]:not(:last-child)::after {
    background: rgba(255,255,255,0.1) !important;
  }
  
  /* Ensure proper contrast for labels */
  [data-testid="stSidebar"] label,
  .stSelectbox label,
  .stTextInput label,
  .stTextArea label {
    color: var(--text) !important;
    font-weight: 600 !important;
  }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
:root { --app-font-size: 1.15rem; }

/* make everything bigger except the main title */
body, .stApp, .main .block-container,
p, div, span, label, li, small, em, strong,
.stMarkdown, .stTextInput *, .stTextArea *, .stSelectbox *,
.stButton > button, code, pre, kbd, samp {
  font-size: var(--app-font-size) !important;
  line-height: 1.35 !important;
}

/* keep your custom title size */
h1.app-title{
  font-size: clamp(3.5rem, 4vw + 2.5rem, 5rem) !important;
}

/* ARIA text styling in title - maintains visual consistency */
h1.app-title em {
  font-size: inherit !important;
  font-style: italic !important;
  font-weight: inherit !important;
  color: inherit !important;
  letter-spacing: inherit !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Feedback buttons: keep text on one line and give room */
.stButton > button{
  white-space: nowrap !important;       /* prevents Helpf / ul split */
  min-width: 10.5rem !important;        /* ~168px; adjust if you want wider */
  font-size: 0.95rem !important;        /* slightly smaller so it fits */
  padding: 0.55rem 1rem !important;
}
</style>
""", unsafe_allow_html=True)

# Session state
if "ta_system" not in st.session_state:
    st.session_state.ta_system = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "system_initialized" not in st.session_state:
    st.session_state.system_initialized = False
if "feedback_data" not in st.session_state:
    st.session_state.feedback_data = []
if "message_feedback" not in st.session_state:
    st.session_state.message_feedback = {}
if "last_question_count" not in st.session_state:
    st.session_state.last_question_count = 0

def initialize_ta_system():
    """Initialize TA system with comprehensive error handling and validation"""
    try:
        # Validate API key
        if not OPENAI_API_KEY or OPENAI_API_KEY.strip() == "":
            st.error("OpenAI API key is empty or not provided. Please check your configuration.")
            return None
        
        if not OPENAI_API_KEY.startswith("sk-"):
            st.error("Invalid OpenAI API key format. API key should start with 'sk-'.")
            return None
        
        # Initialize TA system
        base_path = str(Path(__file__).parent)
        ta_system = StaticsMechanicsTA(base_path, OPENAI_API_KEY)
        
        # Test basic functionality (silent check)
        if not (hasattr(ta_system, 'rag') and ta_system.rag):
            # Only show warning if there are actual issues
            pass
        
        return ta_system
        
    except Exception as e:
        error_msg = str(e)
        
        # Provide specific error messages for common issues
        if "Cannot copy out of meta tensor" in error_msg:
            st.error("PyTorch model loading error detected. The system will attempt to use fallback mode.")
        elif "API key" in error_msg.lower() or "authentication" in error_msg.lower():
            st.error(f"API key validation failed: {error_msg}")
        elif "torch" in error_msg.lower() or "cuda" in error_msg.lower():
            st.error(f"PyTorch/CUDA error: {error_msg}. Attempting CPU fallback.")
        else:
            st.error(f"Failed to initialize TA system: {error_msg}")
        
        # Log the full error for debugging
        st.write(f"Debug info: {error_msg}")
        return None

@st.cache_data(ttl=3600)
def get_course_topics():
    return [
        "Rigid Body Equilibrium", "Force Systems", "Moments", "Trusses",
        "Frames and Machines", "Stress and Strain", "Axial Force Members",
        "Torsion", "Bending", "Shear", "Deflections", "Centroids",
        "Moment of Inertia", "Stress Transformation", "Principal Stresses"
    ]

def update_conversation_feedback(conversation_id, feedback_type):
    """Update feedback for a conversation"""
    try:
        success = feedback_storage.update_feedback(
            conversation_id=conversation_id,
            feedback_type=feedback_type
        )
        if not success:
            st.error("Failed to update feedback")
        return success
    except Exception as e:
        st.error(f"Error updating feedback: {e}")
        return False

def get_conversation_feedback(conversation_id):
    """Get feedback for a conversation"""
    try:
        return feedback_storage.get_conversation_feedback(conversation_id)
    except Exception as e:
        st.error(f"Error getting feedback: {e}")
        return None

def clear_conversation_feedback(conversation_id):
    """Clear feedback for a conversation"""
    try:
        success = feedback_storage.clear_conversation_feedback(conversation_id)
        return success
    except Exception as e:
        st.error(f"Error clearing feedback: {e}")
        return False

def handle_feedback(message_index, feedback_type):
    """Handle user feedback for a specific message"""
    if message_index < len(st.session_state.conversation_history):
        message = st.session_state.conversation_history[message_index]
        if message["role"] == "assistant":
            # Get conversation ID from the message
            conversation_id = message.get("conversation_id")
            if not conversation_id:
                st.error("No conversation ID found for this message")
                return
            
            current_feedback = st.session_state.message_feedback.get(message_index)
            
            # If clicking the same feedback type, remove it (toggle off)
            if current_feedback == feedback_type:
                # Clear feedback from database
                success = clear_conversation_feedback(conversation_id)
                if success:
                    # Remove from session state
                    if message_index in st.session_state.message_feedback:
                        del st.session_state.message_feedback[message_index]
                    
                    # Remove from feedback_data list (for backward compatibility)
                    st.session_state.feedback_data = [
                        entry for entry in st.session_state.feedback_data 
                        if entry.get("message_index") != message_index
                    ]
            else:
                # Update feedback in database
                success = update_conversation_feedback(conversation_id, feedback_type)
                
                if success:
                    # Update session state
                    st.session_state.message_feedback[message_index] = feedback_type
                    
                    # Update feedback_data list (for backward compatibility)
                    user_question = st.session_state.conversation_history[message_index-1]["content"] if message_index > 0 else ""
                    ai_response = message["content"]
                    concepts_covered = message.get("concepts", [])
                    response_time = message.get("response_time", 0)
                    
                    feedback_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "message_index": message_index,
                        "conversation_id": conversation_id,
                        "user_question": user_question,
                        "ta_response": ai_response,
                        "feedback": feedback_type,
                        "concepts_covered": concepts_covered,
                        "response_time": response_time
                    }
                    
                    # Remove any existing feedback for this message
                    st.session_state.feedback_data = [
                        entry for entry in st.session_state.feedback_data 
                        if entry.get("message_index") != message_index
                    ]
                    
                    # Add new feedback
                    st.session_state.feedback_data.append(feedback_entry)
            
            st.rerun()

def main():
    if not st.session_state.system_initialized:
        with st.spinner("Initializing TA system"):
            ta_system = initialize_ta_system()
            if ta_system:
                st.session_state.ta_system = ta_system
                st.session_state.system_initialized = True
            else:
                st.error("Failed to initialize TA system. Check the API key.")
                return
    
    st.markdown(
        '<h1 class="app-title" style="font-size: 3rem !important;"><em>ARIA</em>: Teaching Assistant for Statics and Mechanics of Materials (EN.560.201)</h1>',
        unsafe_allow_html=True
    )
    
    if st.session_state.system_initialized:
        st.markdown(
            """
            <div class="aria-callout">
              <div class="welcome-main">I am ARIA, your teaching assistant for Statics and Mechanics of Materials. I will guide you through problem solving steps and help you understand key concepts.</div>
              <div class="welcome-sub">How can I help you today?</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with st.sidebar:
        st.header("Configuration")
        if st.session_state.system_initialized:
            st.success("TA System Ready")
        else:
            st.error("System Not Ready")
        
        st.subheader("Focus Area")
        topics = get_course_topics()
        selected_topic = st.selectbox(
            "Select a topic to focus on (optional)",
            ["All Topics"] + topics
        )
        
        st.subheader("Conversation")
        if st.button("Clear Conversation"):
            st.session_state.conversation_history = []
            st.rerun()
        
        st.subheader("Tips for Better Learning")
        st.markdown("""
        * Ask specific questions about concepts
        * Describe your problem step by step
        * Ask for guidance, not direct answers
        * Request examples or analogies
        * Ask about common mistakes to avoid
        """)
    
    if st.session_state.system_initialized and st.session_state.ta_system:
        for i, msg in enumerate(st.session_state.conversation_history):
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-message student-message"><strong>You</strong><br>{msg["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                # Display TA message
                st.markdown(
                    f'<div class="chat-message ta-message"><strong>ARIA</strong><br>{msg["content"]}</div>',
                    unsafe_allow_html=True
                )
                
                # Add feedback buttons
                current_feedback = st.session_state.message_feedback.get(i, None)
                
                # Check if this is the most recent TA response (feedback is changeable)
                user_messages_count = len([msg for msg in st.session_state.conversation_history if msg["role"] == "user"])
                message_question_index = len([msg for msg in st.session_state.conversation_history[:i+1] if msg["role"] == "user"])
                is_changeable = (message_question_index == user_messages_count)  # This is the response to the latest question
                
                feedback_html = f"""
                <div class="feedback-container">
                    <span class="feedback-text">Was this response helpful?</span>
                </div>
                """
                st.markdown(feedback_html, unsafe_allow_html=True)
                
                # columns: make the first three wider; last one is spacer
                col1, col2, col3, _spacer = st.columns([2.8, 2.8, 2.8, 3.6])

                with col1:
                    button_type = "primary" if current_feedback == "helpful" else "secondary"
                    if st.button("👍 Helpful", key=f"helpful_{i}",
                                help="This response was helpful" if is_changeable else "Feedback locked after new question",
                                type=button_type, disabled=not is_changeable, use_container_width=True):
                        handle_feedback(i, "helpful")

                with col2:
                    button_type = "primary" if current_feedback == "not_helpful" else "secondary"
                    if st.button("👎 Not Helpful", key=f"not_helpful_{i}",
                                help="This response was not helpful" if is_changeable else "Feedback locked after new question",
                                type=button_type, disabled=not is_changeable, use_container_width=True):
                        handle_feedback(i, "not_helpful")

                with col3:
                    button_type = "primary" if current_feedback == "partially_helpful" else "secondary"
                    if st.button("🤔 Partially", key=f"partial_{i}",
                                help="This response was partially helpful" if is_changeable else "Feedback locked after new question",
                                type=button_type, disabled=not is_changeable, use_container_width=True):
                        handle_feedback(i, "partially_helpful")

                
                # Show feedback status if already given
                if current_feedback:
                    if is_changeable:
                        feedback_text = {
                            "helpful": "✅ You found this helpful (click to change)",
                            "not_helpful": "❌ You found this not helpful (click to change)", 
                            "partially_helpful": "🤔 You found this partially helpful (click to change)"
                        }
                    else:
                        feedback_text = {
                            "helpful": "✅ Marked as helpful",
                            "not_helpful": "❌ Marked as not helpful", 
                            "partially_helpful": "🤔 Marked as partially helpful"
                        }
                    st.markdown(f"<small style='color: var(--muted);'>{feedback_text[current_feedback]}</small>", 
                              unsafe_allow_html=True)
                
                st.divider()
        
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Ask your question",
                placeholder="For example, how to calculate the moment about point A in a beam problem",
                height=100
            )
            col1, col2 = st.columns([1, 4])
            with col1:
                submit_button = st.form_submit_button("Ask ARIA", use_container_width=True)
            
            if submit_button and user_input:
                # Track new question - this will reset feedback changeability for previous responses
                current_question_count = len([msg for msg in st.session_state.conversation_history if msg["role"] == "user"])
                st.session_state.last_question_count = current_question_count + 1
                
                st.session_state.conversation_history.append({
                    "role": "user",
                    "content": user_input
                })
                with st.spinner("ARIA is thinking"):
                    try:
                        # Get session ID (create one if it doesn't exist)
                        if "session_id" not in st.session_state:
                            st.session_state.session_id = conversation_storage.create_session_id()
                        
                        start_time = time.time()
                        response_data = st.session_state.ta_system.generate_response(
                            user_input,
                            st.session_state.conversation_history[-10:]
                        )
                        response_time = time.time() - start_time
                        
                        # Store conversation in database and get conversation ID
                        conversation_id = None
                        if supabase_config.is_connected():
                            conversation_id = conversation_storage.store_conversation(
                                session_id=st.session_state.session_id,
                                user_question=user_input,
                                ta_response=response_data["response"],
                                context_sources=response_data.get("context_sources", []),
                                concepts_used=response_data.get("concepts_covered", []),
                                response_time=response_time
                            )
                        
                        ta_message = {
                            "role": "assistant",
                            "content": response_data["response"],
                            "concepts": response_data.get("concepts_covered", []),
                            "response_time": response_time,
                            "conversation_id": conversation_id
                        }
                        st.session_state.conversation_history.append(ta_message)
                        with st.sidebar:
                            st.metric("Response Time", f"{response_time:.2f}s")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating response: {e}")
    else:
        st.markdown("""
        ## Welcome to your Statics and Mechanics Teaching Assistant
        
        This assistant helps you learn by  
        Guiding you through problem solving steps  
        Explaining key concepts and formulas  
        Providing hints and examples  
        Asking questions to check your understanding
        
        Important  
        This assistant will not give direct answers. It will help you develop problem solving skills.
        
        The system initializes automatically.
        """)
        
        st.subheader("Example Questions You Can Ask")
        examples = [
            "How do I start analyzing a truss structure",
            "What is the difference between stress and strain",
            "Can you guide me through setting up equilibrium equations",
            "What are the key steps for calculating beam deflections",
            "How do I determine if a structure is statically determinate"
        ]
        for example in examples:
            st.markdown(f"• {example}")
    
    st.divider()
    st.markdown(
        '<div class="app-footer"><div>Built by Dibakar Roy Sarkar and Yue Luo, Somdatta Goswami (PI),</div><div> © Centrum Intelliphysics, Civil and System Engineering, Johns Hopkins University</div></div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
