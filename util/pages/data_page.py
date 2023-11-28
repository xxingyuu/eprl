from cgitb import reset
import os
from tkinter.tix import Tree
import requests
import plotly.figure_factory as ff
import plotly.graph_objs as go
from matplotlib.backends.backend_agg import RendererAgg
import matplotlib as mpl
from streamlit import caching
import wavfile
from util.functions.gui import write_st_end
import io
import base64
from copy import deepcopy
from gwosc.api import fetch_event_json
from gwosc import datasets
from gwosc.locate import get_urls
from gwpy.timeseries import TimeSeries
import plotly.express as px
from sqlite3 import Time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import plotly.offline as py                    
py.init_notebook_mode(connected=True)         
import plotly.graph_objs as go                
import plotly.figure_factory as ff   
import plotly.express as px
import requests, os
from gwpy.timeseries import TimeSeries
from gwosc.locate import get_urls
from gwosc import datasets
from gwosc.api import fetch_event_json

from copy import deepcopy
import base64
import io
from util.functions.gui import write_st_end
import wavfile
from streamlit import caching

# Use the non-interactive Agg backend, which is recommended as a
# thread-safe backend.
# See https://matplotlib.org/3.3.2/faq/howto_faq.html#working-with-threads.
mpl.use("agg")


_lock = RendererAgg.lock

datalength = 7
T = 52
format = "%m-%d %H:%M:%S"

# df = pd.read_csv("util/pages/RL_new.csv", parse_dates=['Date/Time'],infer_datetime_format=format)
# easydf = pd.read_csv("util/pages/easy_agent_data.csv", parse_dates=['Date/Time'], infer_datetime_format=format)
df = pd.read_csv("RL_final_v6.csv", parse_dates=['Date/Time'],infer_datetime_format=format)
cur_var = 1

if 'cur' not in st.session_state:
    st.session_state['cur'] = 1

def getGas(gas, df):
    if gas == "CO2":
        carbondf = df[["Site:Environmental Impact Total CO2 Emissions Carbon Equivalent Mass [kg](Hourly)_our",
        "Site:Environmental Impact Total CO2 Emissions Carbon Equivalent Mass [kg](Hourly)_easy","Date/Time"]]
        carbondf.rename(columns={"Site:Environmental Impact Total CO2 Emissions Carbon Equivalent Mass [kg](Hourly)_our":
                        "TRPO MPI算法下CO2排放量(kg/h)",
                        "Site:Environmental Impact Total CO2 Emissions Carbon Equivalent Mass [kg](Hourly)_easy":
                        "Baseline算法下CO2排放量(kg/h)"
                        }, inplace = True)
    elif gas == "CO":
        carbondf = df[["Site:Environmental Impact Electricity CO Emissions Mass [kg](Hourly)_our",
        "Site:Environmental Impact Electricity CO Emissions Mass [kg](Hourly)_easy","Date/Time"]]
        carbondf.rename(columns={"Site:Environmental Impact Electricity CO Emissions Mass [kg](Hourly)_our":
                        "TRPO MPI算法下CO排放量(kg/h)",
                        "Site:Environmental Impact Electricity CO Emissions Mass [kg](Hourly)_easy":
                        "Baseline算法下CO排放量(kg/h)"
                        }, inplace = True)
    elif gas == "CH4":
        # col1 = str(gas) + ":Facility [kg](Hourly)_our"
        # col2 = str(gas) + ":Facility [kg](Hourly)_our"
        carbondf = df[['CH4:Facility [kg](Hourly)_our', 'CH4:Facility [kg](Hourly)_easy', 'Date/Time']]
        carbondf.rename(columns = {"CH4:Facility [kg](Hourly)_our:col3":"TRPO MPI算法下CH4排放量(kg/h)"
                    , "CH4:Facility [kg](Hourly)_easy":"Baseline算法下CH4排放量(kg/h)"}, inplace = True)
    elif gas == "PM2.5":
        carbondf = df[['PM2.5:Facility [kg](Hourly)_our', 'PM2.5:Facility [kg](Hourly)_easy', 'Date/Time']]
        carbondf.rename(columns = {"PM2.5:Facility [kg](Hourly)_our":"TRPO MPI算法下PM2.5排放量(kg/h)"
                    , "PM2.5:Facility [kg](Hourly)_easy":"Baseline算法下PM2.5排放量(kg/h)"}, inplace = True)
    elif gas == "SO2":
        carbondf = df[['SO2:Facility [kg](Hourly)_our', 'SO2:Facility [kg](Hourly)_easy', 'Date/Time']]
        carbondf.rename(columns = {"SO2:Facility [kg](Hourly)_our":"TRPO MPI算法下SO2排放量(kg/h)"
                    , "SO2:Facility [kg](Hourly)_easy":"Baseline算法下SO2排放量(kg/h)"}, inplace = True)
    elif gas == "NOx":
        carbondf = df[['NOx:Facility [kg](Hourly)_our', 'NOx:Facility [kg](Hourly)_easy', 'Date/Time']]
        carbondf.rename(columns = {"NOx:Facility [kg](Hourly)_our":"TRPO MPI算法下NOx排放量(kg/h)"
                    , "NOx:Facility [kg](Hourly)_easy":"Baseline算法下NOx排放量(kg/h)"}, inplace = True)
    return carbondf

def staticPlot(weeks):
    st.title("静态数据")    #Static HVAC Dashboard
    st.markdown("""
        使用以下算法与选项选择您想演示的数据。
    """)    #Use Checkbox and Select menu to select the data you want to display.
    Parameters = ['西部区域HVAC温度', '东部区域HVAC温度'] #HVAC West Temperature  HVAC East Temperature
    Fanmeters = ['西部区域送风速率', '东部区域送风速率']   #West Zone Supply Fan Rate  East Zone Supply Fan Rate
    Outmeters = ['西部区域室温', '东部区域室温']
    check1 = st.checkbox("Baseline")
    check2 = st.checkbox("TRPO MPI")
    st.markdown("""
        - Baseline是优于虚拟代理性能的线性非平凡控制算法。        
        - TRPO MPI是基于强化学习的控制算法。
    """)
        #- Baseline is a linear non-trivial control algorithm which surpassed dummy agent performance. 
        #- TRPO MPI is our Reinforcement Learning based control algorithm.
        
    # selectzone = st.multiselect('Evaluation Parameters', eval, default=['CO2'])
    gas = st.selectbox('评估气体对象', ('CO2', 'CO', 'CH4', 'PM2.5', 'SO2', 'NOx')) #Evaluation Gas
    selectzone = st.multiselect('行为温度参数', Parameters, default=['西部区域HVAC温度'])    #Action Temperature Parameters
    fanmass = st.multiselect('HVAC送风速率', Fanmeters, default = ['西部区域送风速率'])    #HVAC Supply Fan Rate


    outtemperatre = st.multiselect('选择输出室温类型 ', Outmeters, default=['西部区域室温'])    #Select Output Temperatures

    length = len(df)
    X = np.linspace(0, 1, len(df))
    start = weeks[0]
    end = weeks[1]


    carbondf = getGas(gas, df)
    carbondf = carbondf[int(start*length/T):int(end*length/T)]

    zonedf= df[[
                'WEST ZONE DEC OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)_our',
                'WEST ZONE DEC OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)_easy',
                'EAST AIR LOOP OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)_our',
                'EAST AIR LOOP OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)_easy',
                'Date/Time'
            ]]
    zonedf.rename(columns={
        "WEST ZONE DEC OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)_our":
        "TRPO MPI算法下西部区域HVAC排放温度(C)",
        "WEST ZONE DEC OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)_easy":
        "Baseline算法下西部区域HVAC排放温度(C)",
        "EAST AIR LOOP OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)_our":
        "TRPO MPI算法下东部区域HVAC排放温度(C)",
        "EAST AIR LOOP OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)_easy":
        "Baseline算法下东部区域HVAC排放温度(C)"
    }, inplace=True)
    outdf = df[[
                "WEST ZONE:Zone Air Temperature [C](TimeStep)_our",
                "WEST ZONE:Zone Air Temperature [C](TimeStep)_easy",
                "EAST ZONE:Zone Air Temperature [C](TimeStep)_our",
                "EAST ZONE:Zone Air Temperature [C](TimeStep)_easy",
                "Date/Time"
            ]]
    outdf.rename(columns={
                "WEST ZONE:Zone Air Temperature [C](TimeStep)_our":
                "TRPO MPI算法下西部区域实时温度",
                "WEST ZONE:Zone Air Temperature [C](TimeStep)_easy":
                "Baseline算法下西部区域实时温度",
                "EAST ZONE:Zone Air Temperature [C](TimeStep)_our":
                "TRPO MPI算法下东部区域实时温度",
                "EAST ZONE:Zone Air Temperature [C](TimeStep)_easy":
                "Baseline算法下东部区域实时温度"
            }, inplace=True)
    fandf = df[[
        "WEST ZONE SUPPLY FAN:Fan Air Mass Flow Rate [kg/s](Hourly)_our",
        "WEST ZONE SUPPLY FAN:Fan Air Mass Flow Rate [kg/s](Hourly)_easy",
        "EAST ZONE SUPPLY FAN:Fan Air Mass Flow Rate [kg/s](Hourly)_our",
        "EAST ZONE SUPPLY FAN:Fan Air Mass Flow Rate [kg/s](Hourly)_easy",
        "Date/Time"
    ]]
    fandf.rename(columns={
        "WEST ZONE SUPPLY FAN:Fan Air Mass Flow Rate [kg/s](Hourly)_our":
        "TRPO MPI算法下西部区域送风速率",
        "WEST ZONE SUPPLY FAN:Fan Air Mass Flow Rate [kg/s](Hourly)_easy":
        "Baseline算法下西部区域送风速率",
        "EAST ZONE SUPPLY FAN:Fan Air Mass Flow Rate [kg/s](Hourly)_our":
        "TRPO MPI算法下东部区域送风速率",
        "EAST ZONE SUPPLY FAN:Fan Air Mass Flow Rate [kg/s](Hourly)_easy":
        "Baseline算法下东部区域送风速率",
    }, inplace=True)

    zonedf = zonedf[int(start*length/T):int(end*length/T)]
    outdf = outdf[int(start*length/T):int(end*length/T)]
    fandf = fandf[int(start*length/T):int(end*length/T)]
    # print(plotdf.columns[3:5])
    if check1 and check2:
        y = carbondf.columns[0:2]
    elif not check1 and not check2:
        y = None
    elif not check1:
        y = carbondf.columns[0:1]
    elif not check2:
        y = carbondf.columns[1:2]
    placeholder = st.empty()
    
    with placeholder.container():
        if y is not None:
            fig = px.line(carbondf, x='Date/Time', y=y)
            # labels = ['Outer Temprature', ]
            fig.update_xaxes(
                tickangle=45,
                tickformat=format,
                title = "气体排放监测"  #Carbon Emisson with Time Monitor
                )

            st.plotly_chart(fig, use_container_width=True)

        y2 = gety2index(check1, check2, selectzone, zonedf)
        if y2 is not None:
            fig2 = px.line(zonedf, x='Date/Time', y=y2)
            fig2.update_xaxes(
                tickangle=45,
                tickformat=format,
                title = "西部/东部区域(HVAC行为)温度监测"  #West/East Zone Temperature (Action of HVAC) Monitor
                )
            st.plotly_chart(fig2, use_container_width=True)
        y4 = gety4index(check1, check2,fanmass , fandf)
        if y4 is not None:
            fig4 = px.line(fandf, x='Date/Time', y=y4)
            fig4.update_xaxes(
                tickangle=45,
                tickformat=format,
                title = "实时西部/东部每小时送风速率"  #Real-time West/East Supply Fan Air Mass Flow Rate[kg/s] Hourly
                )

            fig4.update_yaxes(
                range=[0,8]
            )
            st.plotly_chart(fig4, use_container_width=True)

        y3 = gety3index(check1, check2, outtemperatre, outdf)
        if y3 is not None:
            fig3 = px.line(outdf, x='Date/Time', y=y3)
            fig3.update_xaxes(
                tickangle=45,
                tickformat=format,
                title = "实时西部/东部室温监测" #Real-time West/East Zone Real Temperature Monitor
                )
            st.plotly_chart(fig3, use_container_width=True)       
        
    if 'cur' not in st.session_state:
        st.session_state['cur'] = 1
    st.session_state['cur'] = 1
    # st.caption("Date/Time vs Controled Air ")
    # fig2 = px.line(plotdf, x='Date/Time', y=plotdf.columns[4:6])
    # st.plotly_chart(fig2, use_container_width=True)

    # # st.caption("Date/Time vs Evaluation Data")
    # fig3 = px.line(plotdf, x='Date/Time', y=plotdf.columns[7:10])
    # st.plotly_chart(fig3, use_container_width=True)

Medudict = {'West': ['WEST ZONE DEC OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)_easy.',
                    'WEST ZONE DEC OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)_our'
                    ],
            'East': [
                    'EAST ZONE:Zone Air Temperature [C](TimeStep)_easy',
                    'EAST ZONE:Zone Air Temperature [C](TimeStep)_our'
                    ]
            }

def gety2index(check1, check2, selectzone, zonedf):
    west = False
    east = False
    if "西部区域HVAC温度" in selectzone:    #HVAC West Temperature
        west = True
    if "东部区域HVAC温度" in selectzone:    #HVAC East Temperature
        east = True
    if not check1 and not check2:
        return None
    if check1 and check2:
        if west and east:
            return zonedf.columns[0:4]
        elif west and not east:
            return zonedf.columns[0:2]
        elif east and not west:
            return zonedf.columns[2:4]
        else:
            return None

    if not check2:
        if west and east:
            newdf = zonedf[["East Out Temperature (C) of Baseline","West Out Temperature (C) of Baseline"]] #  West Out Temperature (C) of Baseline
            return newdf.columns[0:2]
        elif west:
            return zonedf.columns[1:2]
        elif east:
            return zonedf.columns[3:4]
    if not check1:
        if west and east:
            newdf = zonedf[["East Out Temperature (C) of TRPO MPI","West Out Temperature (C) of TRPO MPI"]] #East Out Temperature (C) of TRPO MPI  West Out Temperature (C) of TRPO MPI
            return newdf.columns[0:2]
        elif west:
            return zonedf.columns[0:1]
        elif east:
            return zonedf.columns[2:3]


def gety3index(check1, check2, selectzone, zonedf):
    west = False
    east = False
    if "西部区域室温" in selectzone:    #East Zone Supply Fan Rate
        west = True
    if "东部区域室温" in selectzone:    #East Temperature
        east = True
    if not check1 and not check2:
        return None
    if check1 and check2:
        if west and east:
            return zonedf.columns[0:4]
        elif west and not east:
            return zonedf.columns[0:2]
        elif east and not west:
            return zonedf.columns[2:4]
        else:
            return None
    if not check2:
        if west and east:
            newdf = zonedf[["East Zone Real Temperature(C) of Baseline","West Zone Real Temperature(C) of Baseline"]]
            return newdf.columns[0:2]
        elif west:
            return zonedf.columns[1:2]
        elif east:
            return zonedf.columns[3:4]
    if not check1:
        if west and east:
            newdf = zonedf[["East Zone Real Temperature(C) of TRPO MPI","West Zone Real Temperature(C) of TRPO MPI"]]
            return newdf.columns[0:2]
        elif west:
            return zonedf.columns[0:1]
        elif east:
            return zonedf.columns[2:3]

def gety4index(check1, check2, selectzone, zonedf):
    west = False
    east = False
    if "西部区域送风速率" in selectzone:   #West Zone Supply Fan Rate
        west = True
    if "东部区域送风速率" in selectzone:    #East Zone Supply Fan Rate
        east = True
    if not check1 and not check2:
        return None
    if check1 and check2:
        if west and east:
            return zonedf.columns[0:4]
        elif west and not east:
            return zonedf.columns[0:2]
        elif east and not west:
            return zonedf.columns[2:4]
        else:
            return None
    if not check2:
        if west and east:
            newdf = zonedf[["East Zone Supply Fan Rate of Baseline","West Zone Supply Fan Rate of Baseline"]]
            return newdf.columns[0:2]
        elif west:
            return zonedf.columns[1:2]
        elif east:
            return zonedf.columns[3:4]
    if not check1:
        if west and east:
            newdf = zonedf[["East Zone Supply Fan Rate of TRPO MPI","West Zone Supply Fan Rate of TRPO MPI"]]
            return newdf.columns[0:2]
        elif west:
            return zonedf.columns[0:1]
        elif east:
            return zonedf.columns[2:3]



def dynamicPlot(weeks):
    st.title("实时数据")    #Real-Time HVAC Dashboard
    st.markdown("""
        使用以下算法与选项选择您想演示的数据。
    """)
    move_length = 160
    st.experimental_memo.clear()
    Parameters = ['西部区域HVAC温度', '东部区域HVAC温度']
    Fanmeters = ['西部区域送风速率', '东部区域送风速率']

    Outmeters = ['西部区域室温', '东部区域室温']
    check1 = st.checkbox("Baseline")
    check2 = st.checkbox("TRPO MPI")
    st.markdown("""
        - Baseline是优于虚拟代理性能的线性非平凡控制算法。        
        - TRPO MPI是基于强化学习的控制算法。
    """)

    gas = st.selectbox('评估气体对象', ('CO2', 'CO', 'CH4', 'PM2.5', 'SO2', 'NOx'))   
    selectzone = st.multiselect('选择HVAC温度参数 ', Parameters, default=['西部区域HVAC温度'])
    # selectzone = st.multiselect('Select HVAC Temperature Parameters ', Parameters)

    fanmass = st.multiselect('选择区域送风速率参数',Fanmeters, default=['西部区域送风速率'])
    # fanmass = st.multiselect('Select Zone Fan Parameters',Fanmeters)

    outtemperatre = st.multiselect('选择输出温度类型 ', Outmeters, default=['西部区域室温'])
    # outtemperatre = st.multiselect('Select Output Temperatures ', Outmeters)

    placeholder = st.empty()
    Timestep = len(df)
    # start = cur_var
    start = st.session_state['cur']

    if not check1 and not check2:
        return 
    

    for i in range(start, Timestep):
        with placeholder.container():

            length = len(df)
            kpi1, kpi2 = st.columns(2)

            if i < move_length:
                plotdf = df[0:int(i*length/Timestep)]
                # basedf = easydf[0:int(i*length/Timestep)]
            else:
                plotdf = df[int((i-move_length)*length/Timestep):int(i*length/Timestep)]

            kpi1.metric(
                label = "我们的" + str(gas) + "平均排放量 (kg/h)",
                value = round(np.average(getGas(gas, plotdf).iloc[:,0]),3)
            )
            kpi2.metric(
                label = "Baseline模式下" + str(gas) + "平均排放量 (kg/h)",
                value = round(np.average(getGas(gas, plotdf).iloc[:,0]),3)
            )

                # basedf = easydf[int((i-5)*length/Timestep):int(i*length/Timestep)]
            zonedf = plotdf[[
                'WEST ZONE DEC OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)_our',
                'WEST ZONE DEC OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)_easy',
                'EAST AIR LOOP OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)_our',
                'EAST AIR LOOP OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)_easy',
                'Date/Time'
            ]]

            zonedf.rename(columns={
                "WEST ZONE DEC OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)_our":
                "TRPO MPI算法下西部区域HVAC排放温度(C)",
                "WEST ZONE DEC OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)_easy":
                "Baseline算法下西部区域HVAC排放温度(C)",
                "EAST AIR LOOP OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)_our":
                "TRPO MPI算法下东部区域HVAC排放温度(C)",
                "EAST AIR LOOP OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)_easy":
                "Baseline算法下东部区域HVAC排放温度(C)"
            }, inplace=True)

            outdf = plotdf[[
                "WEST ZONE:Zone Air Temperature [C](TimeStep)_our",
                "WEST ZONE:Zone Air Temperature [C](TimeStep)_easy",
                "EAST ZONE:Zone Air Temperature [C](TimeStep)_our",
                "EAST ZONE:Zone Air Temperature [C](TimeStep)_easy",
                'Date/Time'
            ]]
            outdf.rename(columns={
                "WEST ZONE:Zone Air Temperature [C](TimeStep)_our":
                "TRPO MPI算法下西部区域实时温度(C)",
                "WEST ZONE:Zone Air Temperature [C](TimeStep)_easy":
                "Baseline算法下西部区域实时温度(C)",
                "EAST ZONE:Zone Air Temperature [C](TimeStep)_our":
                "TRPO MPI算法下东部区域实时温度(C)",
                "EAST ZONE:Zone Air Temperature [C](TimeStep)_easy":
                "Baseline算法下东部区域实时温度(C)"
            }, inplace=True)


            fandf = plotdf[[
                "WEST ZONE SUPPLY FAN:Fan Air Mass Flow Rate [kg/s](Hourly)_our",
                "WEST ZONE SUPPLY FAN:Fan Air Mass Flow Rate [kg/s](Hourly)_easy",
                "EAST ZONE SUPPLY FAN:Fan Air Mass Flow Rate [kg/s](Hourly)_our",
                "EAST ZONE SUPPLY FAN:Fan Air Mass Flow Rate [kg/s](Hourly)_easy",
                "Date/Time"
            ]]
            fandf.rename(columns={
                "WEST ZONE SUPPLY FAN:Fan Air Mass Flow Rate [kg/s](Hourly)_our":
                "TRPO MPI算法下西部区域送风速率",
                "WEST ZONE SUPPLY FAN:Fan Air Mass Flow Rate [kg/s](Hourly)_easy":
                "Baseline算法下西部区域送风速率",
                "EAST ZONE SUPPLY FAN:Fan Air Mass Flow Rate [kg/s](Hourly)_our":
                "TRPO MPI算法下西部区域送风速率",
                "EAST ZONE SUPPLY FAN:Fan Air Mass Flow Rate [kg/s](Hourly)_easy":
                "Baseline算法下西部区域送风速率",
            }, inplace=True)

            subdf = getGas(gas, plotdf)

            y_ = subdf.columns[0:2]
            if check1 and not check2:
                y_ = subdf.columns[1:2]
            if check2 and not check1:
                y_ = subdf.columns[0:1]
            
            fig = px.line(subdf, x='Date/Time', y=y_)
            fig.update_xaxes(
                            tickangle=45,
                            dtick=12,
                            tickformat=format,
                            range=[0,move_length],
                            title = "实时CO2排放量监测"  #Real-time CO2 Emission Equivalent Mass Monitor
                            )
            # fig.update_yaxes(range=[0,32])
            # fig.addtrace(bbasedf, x = 'Date/Time', y=basedf.columms[5:6])
            st.plotly_chart(fig, use_container_width=True)

        
            y2 = gety2index(check1, check2, selectzone, zonedf)
            fig2 = px.line(zonedf, x='Date/Time', y=y2)
            fig2.update_xaxes(
                tickangle=45,
                dtick=12,
                tickformat=format,
                range=[0,move_length],
                title = "实时西部/东部区域(HVAC行为)温度监测"  #Real-time West/East Zone Temperature (Action of HVAC) Monitor
                )


            st.plotly_chart(fig2, use_container_width=True)

            y4 = gety4index(check1, check2,fanmass , fandf)
            fig4 = px.line(fandf, x='Date/Time', y=y4)
            fig4.update_xaxes(
                tickangle=45,
                dtick=12,
                tickformat=format,
                range=[0,move_length],
                title = "实时西部/东部区域室温监测"  #Real-time West/East Zone Real Temperature Monitor
                )
            fig4.update_yaxes(
                range=[0,8]
            )
            st.plotly_chart(fig4, use_container_width=True)
            
            y3 = gety3index(check1, check2,outtemperatre, outdf)
            fig3 = px.line(outdf, x='Date/Time', y=y3)
            fig3.update_xaxes(
                tickangle=45,
                dtick=12,
                tickformat=format,
                range=[0,move_length],
                title = "实时西部/东部区域室温监测"  #Real-time West/East Zone Real Temperature Monitor
                )
            st.plotly_chart(fig3, use_container_width=True)
            

            time.sleep(0.1)
            st.session_state['cur'] += 1

    st.session_state['cur'] = 1


def data_page():

    # -- Set page config
    # apptitle = 'GW Quickview'

    # st.set_page_config(page_title=apptitle, page_icon=":eyeglasses:")

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

    # -- Default detector list
    detectorlist = ['H1', 'L1', 'V1']

    # Title the app

    if 'cur' not in st.session_state:
        st.session_state['cur'] = 1

    @st.cache(ttl=3600, max_entries=10)  # -- Magic command to cache data
    def load_gw(t0, detector, fs=4096):
        strain = TimeSeries.fetch_open_data(
            detector, t0-14, t0+14, sample_rate=fs, cache=False)
        return strain

        
    st.sidebar.markdown("## 演示类型")
    mode = st.sidebar.selectbox('您想如何演示类型？', [ '静态', '动态' ])   

    st.sidebar.markdown('## 时间（周）')
    weeks = st.sidebar.slider('周数', min_value=1,
                              max_value=52, value=(1,2), step=1)
    df = pd.read_csv('./util/pages/newmtr.csv')
    X = df['Date/Time']



    if mode == '静态':
        with _lock:
            staticPlot(weeks)
            write_st_end()

    elif mode == '动态':
        dynamicPlot(weeks)
        write_st_end()

        pass


 