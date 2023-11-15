import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

def member_page():
    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
    width: 300px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
    width: 300px;
    margin-left: -300px;
    }
    </style>
    """,
        unsafe_allow_html=True
    )


    st.write("---")
    st.markdown("""
     ## [姓名]
    <p style="font-size: 22px;">
    [内容]
    </p>
    """, unsafe_allow_html=True
    )
