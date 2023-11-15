import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from ..functions.gui import create_st_button



def home_page():
    st.markdown('<center><img src="https://s3.bmp.ovh/imgs/2022/07/13/ea94a09608d23d21.gif" width=900 height=600></center>', unsafe_allow_html=True)
    # change sidebar width
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

    left_col, right_col = st.columns([1,2])

    logoimage = Image.open('Logo 450 I.png')
    left_col.image(logoimage, width=300,output_format="PNG")


    right_col.markdown('# 面向未来低碳建筑的实时碳排放评估')

    right_col.markdown('## 基于强化学习和EnergyPlus的智能碳建筑控制工具')


    paper_link_dict = {
        "IBM Paper": "https://github.com/IBM/rl-testbed-for-energyplus",
    }
    st.sidebar.subheader("相关数据库链接")
    for link_text, link_url in paper_link_dict.items():
        create_st_button(link_text, link_url, st_col=st.sidebar)
    

    software_link_dict = {
        "EnergyPlus": "https://energyplus.net/",}
    software_link_dict2 = {
        "DesignBuilder": "https://designbuilder.co.uk/",
    }    
    software_link_dict3 = {
        "OpenAI_Baselines": "https://github.com/openai/baselines"
    }


    st.sidebar.subheader("相关软件链接")
    for link_text, link_url in software_link_dict.items():
        create_st_button(link_text, link_url, st_col=st.sidebar)
    for link_text, link_url in software_link_dict2.items():
        create_st_button(link_text, link_url, st_col=st.sidebar)
    for link_text, link_url in software_link_dict3.items():
        create_st_button(link_text, link_url, st_col=st.sidebar)


    st.markdown("---")

    st.markdown(
    """
    ## 概要
    <p style="font-size: 22px;">
    如今，住宅和办公楼消耗的能源占据了碳排放的很大一部分。了解建筑物中碳排放的实时状态，可以帮助建筑运营商优化能源使用并控制碳排放。此外，建立实时的碳排放监测系统，作为评估未来低碳建筑可再生能源解决方案的基础，也是非常重要的。作为一个长期的发展方向，项目旨在为乡村振兴设计和建造未来的低碳建筑。
    """,unsafe_allow_html=True)
    st.write('---')

    left_col, right_col = st.columns(2)
    
    left_col.markdown(
        
    """
    ## 使用
     <p style="font-size: 22px;">
    通过左侧主菜单，可进入此产品各界面：
    </p>
    <p style="font-size: 20px;">-&nbsp <b>
    首页</b>：本页面。此页面是“面向未来低碳建筑的实时碳排放评估”项目的概况，囊括该项目的基本信息。本项目旨在降低建筑物碳排放，同时保障屋内人员的舒适。</p>
    
    <p style="font-size: 20px;">-&nbsp <b>
    建筑数据</b>：模拟用建筑的关键数据。</p>

    <p style="font-size: 20px;">-&nbsp <b>
    结果可视化</b>：数据页面。包括室内外温度、碳排放率的折线图。通过调整时间尺度查看不同时间段静态或动态的变化情况。</p>
    """, unsafe_allow_html=True)

    cnptimg = Image.open('util/data/diag.png')
    right_col.image(cnptimg,width=450, output_format="PNG")
    

#<p style="font-size: 20px;">-&nbsp <b>
#    团队成员：</b>团队成员页面展示了团队各成员的背景和感兴趣的领域。有任何疑问欢迎联系我们。</p>