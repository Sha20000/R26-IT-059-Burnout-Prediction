import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
import pickle
import math
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="BurnoutGuard AI | R26-IT-059",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 25px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .main-header h1 {
        font-size: 2.0em;
        font-weight: 700;
        letter-spacing: 1px;
        margin: 0;
    }
    .main-header p {
        font-size: 0.95em;
        opacity: 0.9;
        margin: 5px 0 0 0;
    }
    .risk-high {
        color: #ff4b4b;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffa500;
        font-weight: bold;
    }
    .risk-low {
        color: #00cc44;
        font-weight: bold;
    }
    .section-divider {
        border-top: 1px solid #2d6a9f;
        margin: 20px 0;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    anomaly = pd.read_csv(
        "../results/metrics/anomaly_scores_v2.csv"
    )
    features = pd.read_csv(
        "../results/metrics/features_v2.csv"
    )
    expected = pd.read_csv(
        "../results/metrics/expected_behavior.csv"
    )
    return anomaly, features, expected

@st.cache_resource
def load_model():
    class VAE_V2(nn.Module):
        def __init__(self, input_dim=10, latent_dim=6):
            super(VAE_V2, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64), nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU()
            )
            self.fc_mu = nn.Linear(16, latent_dim)
            self.fc_logvar = nn.Linear(16, latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 16), nn.ReLU(),
                nn.Linear(16, 32), nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 64), nn.ReLU(),
                nn.Linear(64, input_dim)
            )
        def encode(self, x):
            h = self.encoder(x)
            return self.fc_mu(h), self.fc_logvar(h)
        def decode(self, z):
            return self.decoder(z)
        def forward(self, x):
            mu, logvar = self.encode(x)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            return self.decode(z), mu, logvar

    device = torch.device('cpu')
    vae = VAE_V2(input_dim=10, latent_dim=6).to(device)
    vae.load_state_dict(
        torch.load('../models/saved/vae_v2.pt',
                   map_location=device)
    )
    vae.eval()
    with open('../models/saved/scaler_v2.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return vae, scaler

anomaly_df, features_df, expected_df = load_data()
vae_model, scaler = load_model()

FEATURE_COLS = [
    'weekly_clicks', 'active_days', 'content_diversity',
    'forum_clicks', 'resource_clicks', 'quiz_clicks',
    'avg_submission_latency', 'late_submissions',
    'curriculum_compliance', 'click_compliance'
]

academic_calendar = {
    'AAA': {'holiday_weeks': [8, 9], 'exam_weeks': [34, 35, 36, 37]},
    'BBB': {'holiday_weeks': [8, 9], 'exam_weeks': [32, 33, 34, 35]},
    'CCC': {'holiday_weeks': [8, 9], 'exam_weeks': [32, 33, 34, 35]},
    'DDD': {'holiday_weeks': [8, 9], 'exam_weeks': [32, 33, 34, 35]},
    'EEE': {'holiday_weeks': [8, 9], 'exam_weeks': [33, 34, 35, 36]},
    'FFF': {'holiday_weeks': [8, 9], 'exam_weeks': [32, 33, 34, 35]},
    'GGG': {'holiday_weeks': [8, 9], 'exam_weeks': [33, 34, 35, 36]}
}

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:15px;
    background:linear-gradient(135deg,#1e3a5f,#2d6a9f);
    border-radius:10px; color:white; margin-bottom:15px'>
    <h2 style='margin:0; font-size:1.4em'>BurnoutGuard AI</h2>
    <p style='font-size:0.8em; margin:5px 0 0 0;
    opacity:0.9'>Academic Early Warning System</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Navigation**")
    page = st.radio(
        "",
        [
            "[1] System Dashboard",
            "[2] Module Analysis",
            "[3] Student Profile",
            "[4] Admin Panel",
            "[5] Live Detection"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Project Information**")
    st.markdown("Project ID: R26-IT-059")
    st.markdown("Student ID: IT22215710")
    st.markdown("Name: Karunarathne D C")
    st.markdown("Institute: SLIIT | 2025-2026")
    st.markdown("---")
    st.markdown("**Model Performance**")
    st.markdown("AUC V2: **0.6039**")
    st.markdown("AUC V1: 0.5840")
    st.markdown("Improvement: **+1.99%**")
    st.markdown("Best Module: FFF — 0.6312")

# ─────────────────────────────────────────
# PAGE 1 — SYSTEM DASHBOARD
# ─────────────────────────────────────────
if page == "[1] System Dashboard":

    st.markdown("""
    <div class='main-header'>
    <h1>BurnoutGuard AI System</h1>
    <p>Explainable Multi-Modal Artificial Intelligence
    for Academic Burnout and Dropout Prediction</p>
    <p>R26-IT-059 | SLIIT | IT22215710 | Karunarathne D C</p>
    </div>
    """, unsafe_allow_html=True)

    total_students = anomaly_df['id_student'].nunique()
    high_risk = anomaly_df[
        anomaly_df['combined_score'] >
        anomaly_df['combined_score'].quantile(0.75)
    ]['id_student'].nunique()
    avg_compliance = anomaly_df['curriculum_compliance'].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Students Monitored", f"{total_students:,}")
    col2.metric(
        "High Risk Students",
        f"{high_risk:,}",
        delta=f"{high_risk/total_students*100:.1f}% of total"
    )
    col3.metric("Avg Compliance Rate", f"{avg_compliance:.1%}")
    col4.metric("Model AUC (V2)", "0.6039",
                delta="+1.99% vs V1")
    col5.metric("Modules Tracked", "7")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Score Distribution")
        fig = px.histogram(
            anomaly_df,
            x='combined_score',
            color='at_risk',
            color_discrete_map={0: '#2d6a9f', 1: '#c0392b'},
            labels={
                'combined_score': 'Behavioral Risk Score',
                'at_risk': 'At Risk (1=Yes, 0=No)'
            },
            barmode='overlay',
            opacity=0.7,
            title='Behavioral Risk Score Distribution by At-Risk Status'
        )
        fig.update_layout(template='plotly_dark', height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Weekly Risk and Compliance Trends")
        weekly = anomaly_df.groupby('week').agg(
            avg_risk=('combined_score', 'mean'),
            avg_compliance=('curriculum_compliance', 'mean')
        ).reset_index()

        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(
            go.Scatter(
                x=weekly['week'],
                y=weekly['avg_risk'],
                name='Avg Risk Score',
                line=dict(color='#c0392b', width=2),
                mode='lines+markers'
            ), secondary_y=False
        )
        fig2.add_trace(
            go.Scatter(
                x=weekly['week'],
                y=weekly['avg_compliance'],
                name='Avg Compliance',
                line=dict(color='#27ae60', width=2),
                mode='lines+markers'
            ), secondary_y=True
        )
        fig2.update_layout(
            template='plotly_dark',
            height=350,
            title='Weekly Risk Score and Curriculum Compliance'
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Module Performance Overview")

    module_stats = anomaly_df.groupby('code_module').agg(
        students=('id_student', 'nunique'),
        avg_risk=('combined_score', 'mean'),
        avg_compliance=('curriculum_compliance', 'mean'),
        at_risk_rate=('at_risk', 'mean')
    ).reset_index()

    module_auc = {
        'AAA': 0.5390, 'BBB': 0.5633,
        'CCC': 0.6091, 'DDD': 0.6026,
        'EEE': 0.6126, 'FFF': 0.6312,
        'GGG': 0.5562
    }
    module_stats['auc'] = module_stats['code_module'].map(module_auc)

    col1, col2 = st.columns(2)

    with col1:
        fig3 = px.bar(
            module_stats.sort_values('auc', ascending=False),
            x='code_module', y='auc',
            color='auc',
            color_continuous_scale='RdYlGn',
            title='Detection AUC Score by Module',
            labels={'code_module': 'Module', 'auc': 'AUC Score'}
        )
        fig3.add_hline(
            y=0.5, line_dash="dash",
            line_color="red",
            annotation_text="Random Baseline (0.5)"
        )
        fig3.update_layout(template='plotly_dark', height=350)
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        fig4 = px.bar(
            module_stats.sort_values('at_risk_rate', ascending=False),
            x='code_module', y='at_risk_rate',
            color='at_risk_rate',
            color_continuous_scale='RdYlGn_r',
            title='At-Risk Student Rate by Module',
            labels={
                'code_module': 'Module',
                'at_risk_rate': 'At-Risk Rate'
            }
        )
        fig4.update_layout(template='plotly_dark', height=350)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.subheader("Top 20 Highest Risk Students")

    top_risk = anomaly_df.groupby(
        ['id_student', 'code_module']
    ).agg(
        avg_risk=('combined_score', 'mean'),
        avg_compliance=('curriculum_compliance', 'mean'),
        high_risk_weeks=('combined_score',
                         lambda x: (x > 0.5).sum()),
        at_risk=('at_risk', 'first')
    ).reset_index().sort_values(
        'avg_risk', ascending=False
    ).head(20)

    top_risk['Risk Level'] = top_risk['avg_risk'].apply(
        lambda x: 'HIGH' if x > 0.5
        else 'MEDIUM' if x > 0.3
        else 'LOW'
    )
    top_risk['Compliance'] = top_risk[
        'avg_compliance'
    ].apply(lambda x: f"{x:.1%}")

    st.dataframe(
        top_risk[[
            'id_student', 'code_module', 'avg_risk',
            'Compliance', 'high_risk_weeks',
            'Risk Level', 'at_risk'
        ]].rename(columns={
            'id_student': 'Student ID',
            'code_module': 'Module',
            'avg_risk': 'Avg Risk Score',
            'high_risk_weeks': 'High Risk Weeks',
            'at_risk': 'Actually At-Risk'
        }),
        use_container_width=True,
        height=400
    )

# ─────────────────────────────────────────
# PAGE 2 — MODULE ANALYSIS
# ─────────────────────────────────────────
elif page == "[2] Module Analysis":

    st.markdown("## Module Level Analysis")
    st.markdown(
        "Analyze behavioral patterns and risk levels "
        "per academic module."
    )
    st.markdown("---")

    selected_module = st.selectbox(
        "Select Module",
        sorted(anomaly_df['code_module'].unique())
    )

    module_data = anomaly_df[
        anomaly_df['code_module'] == selected_module
    ].copy()

    module_features = features_df[
        features_df['code_module'] == selected_module
    ].copy()

    total = module_data['id_student'].nunique()
    at_risk_count = module_data[
        module_data['at_risk'] == 1
    ]['id_student'].nunique()
    avg_compliance = module_data['curriculum_compliance'].mean()
    module_auc_map = {
        'AAA': 0.5390, 'BBB': 0.5633, 'CCC': 0.6091,
        'DDD': 0.6026, 'EEE': 0.6126, 'FFF': 0.6312,
        'GGG': 0.5562
    }
    auc = module_auc_map.get(selected_module, 0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", f"{total:,}")
    col2.metric(
        "At-Risk Students",
        f"{at_risk_count:,}",
        delta=f"{at_risk_count/total*100:.1f}% of module"
    )
    col3.metric("Avg Compliance", f"{avg_compliance:.1%}")
    col4.metric("Module Detection AUC", f"{auc:.4f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Weekly Compliance Trend")
        weekly_module = module_data.groupby('week').agg(
            avg_compliance=('curriculum_compliance', 'mean'),
            avg_risk=('combined_score', 'mean')
        ).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=weekly_module['week'],
            y=weekly_module['avg_compliance'],
            mode='lines+markers',
            name='Avg Compliance',
            line=dict(color='#27ae60', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=weekly_module['week'],
            y=weekly_module['avg_risk'],
            mode='lines+markers',
            name='Avg Risk Score',
            line=dict(color='#c0392b', width=2)
        ))
        fig.add_hline(
            y=0.5, line_dash="dash",
            line_color="orange",
            annotation_text="Risk Threshold"
        )

        cal = academic_calendar.get(selected_module, {})
        for hw in cal.get('holiday_weeks', []):
            fig.add_vrect(
                x0=hw-0.5, x1=hw+0.5,
                fillcolor="blue", opacity=0.15,
                annotation_text="Holiday"
            )

        fig.update_layout(
            template='plotly_dark', height=350,
            title=f'Module {selected_module} — Weekly Trends'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Risk Score Distribution")
        fig2 = px.box(
            module_data,
            x='at_risk', y='combined_score',
            color='at_risk',
            color_discrete_map={0: '#2d6a9f', 1: '#c0392b'},
            labels={
                'at_risk': 'At-Risk Status (1=Yes, 0=No)',
                'combined_score': 'Combined Risk Score'
            },
            title=f'Risk Score by Status — Module {selected_module}'
        )
        fig2.update_layout(template='plotly_dark', height=350)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("LMS Behavioral Feature Trends")

    weekly_features = module_features[
        module_features['week_type'] == 'normal'
    ].groupby('week')[
        ['weekly_clicks', 'active_days',
         'forum_clicks', 'resource_clicks', 'quiz_clicks']
    ].mean().reset_index()

    fig3 = px.line(
        weekly_features, x='week',
        y=['weekly_clicks', 'active_days',
           'forum_clicks', 'resource_clicks', 'quiz_clicks'],
        title=f'Module {selected_module} — Average Weekly Features',
        markers=True
    )
    fig3.update_layout(template='plotly_dark', height=400)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.subheader(
        f"Top At-Risk Students — Module {selected_module}"
    )

    top_module = module_data.groupby('id_student').agg(
        avg_risk=('combined_score', 'mean'),
        avg_compliance=('curriculum_compliance', 'mean'),
        high_risk_weeks=('combined_score',
                         lambda x: (x > 0.5).sum()),
        at_risk=('at_risk', 'first')
    ).reset_index().sort_values(
        'avg_risk', ascending=False
    ).head(15)

    top_module['Risk Level'] = top_module['avg_risk'].apply(
        lambda x: 'HIGH' if x > 0.5
        else 'MEDIUM' if x > 0.3
        else 'LOW'
    )

    st.dataframe(
        top_module[[
            'id_student', 'avg_risk', 'avg_compliance',
            'high_risk_weeks', 'Risk Level', 'at_risk'
        ]].rename(columns={
            'id_student': 'Student ID',
            'avg_risk': 'Risk Score',
            'avg_compliance': 'Compliance',
            'high_risk_weeks': 'High Risk Weeks',
            'at_risk': 'Actually At-Risk'
        }),
        use_container_width=True
    )

# ─────────────────────────────────────────
# PAGE 3 — STUDENT PROFILE
# ─────────────────────────────────────────
elif page == "[3] Student Profile":

    st.markdown("## Student Behavioral Profile")
    st.markdown(
        "Track individual student behavior "
        "week by week and identify anomalies."
    )
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        selected_module = st.selectbox(
            "Select Module",
            sorted(anomaly_df['code_module'].unique())
        )
    with col2:
        module_students = sorted(
            anomaly_df[
                anomaly_df['code_module'] == selected_module
            ]['id_student'].unique()
        )
        selected_student = st.selectbox(
            "Select Student ID", module_students
        )

    student_anomaly = anomaly_df[
        (anomaly_df['id_student'] == selected_student) &
        (anomaly_df['code_module'] == selected_module)
    ].sort_values('week')

    student_features = features_df[
        (features_df['id_student'] == selected_student) &
        (features_df['code_module'] == selected_module)
    ].sort_values('week')

    if len(student_anomaly) == 0:
        st.warning(
            "No data available for this student "
            "and module combination."
        )
        st.stop()

    max_week = int(student_anomaly['week'].max())
    selected_week = st.slider(
        "Simulate Time — Show data up to week:",
        min_value=3, max_value=max_week,
        value=min(8, max_week),
        help="Move slider to simulate weeks passing"
    )

    visible = student_anomaly[
        student_anomaly['week'] <= selected_week
    ]

    at_risk = student_anomaly['at_risk'].iloc[0]
    avg_risk = visible['combined_score'].mean()
    avg_compliance = visible['curriculum_compliance'].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Student ID", str(selected_student))
    col2.metric("Avg Risk Score", f"{avg_risk:.4f}")
    col3.metric("Avg Compliance", f"{avg_compliance:.1%}")
    col4.metric(
        "Actual Outcome",
        "AT RISK" if at_risk == 1 else "NORMAL"
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Score Timeline")

        fig = go.Figure()
        fig.add_vrect(
            x0=-0.5, x1=2.5,
            fillcolor="blue", opacity=0.1,
            annotation_text="Baseline Period (Weeks 0-2)",
            annotation_position="top left"
        )

        marker_colors = visible['combined_score'].apply(
            lambda x: '#c0392b' if x > 0.5
            else '#f39c12' if x > 0.3
            else '#27ae60'
        )

        fig.add_trace(go.Scatter(
            x=visible['week'],
            y=visible['combined_score'],
            mode='lines+markers',
            name='Risk Score',
            line=dict(color='#2d6a9f', width=2),
            marker=dict(size=12, color=list(marker_colors))
        ))

        fig.add_trace(go.Scatter(
            x=visible['week'],
            y=visible['curriculum_compliance'],
            mode='lines+markers',
            name='Compliance Rate',
            line=dict(color='#27ae60', width=2, dash='dot')
        ))

        fig.add_hline(
            y=0.5, line_dash="dash", line_color="#c0392b",
            annotation_text="High Risk Threshold"
        )
        fig.add_hline(
            y=0.3, line_dash="dash", line_color="#f39c12",
            annotation_text="Medium Risk Threshold"
        )

        fig.update_layout(
            template='plotly_dark', height=400,
            title=f'Student {selected_student} — Risk Timeline',
            xaxis_title='Week',
            yaxis_title='Score (0-1)'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Behavioral Feature Trends")

        student_feat_visible = student_features[
            student_features['week'] <= selected_week
        ]

        if len(student_feat_visible) > 0:
            fig2 = px.line(
                student_feat_visible, x='week',
                y=['weekly_clicks', 'active_days',
                   'resource_clicks', 'forum_clicks'],
                title=f'Student {selected_student} — Feature History',
                markers=True
            )
            fig2.update_layout(
                template='plotly_dark', height=400
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader(f"Week {selected_week} — Detailed Analysis")

    current = visible[visible['week'] == selected_week]
    if len(current) > 0:
        risk = current.iloc[0]['combined_score']
        compliance = current.iloc[0]['curriculum_compliance']

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Risk Score", f"{risk:.4f}")
            st.metric("Curriculum Compliance",
                      f"{compliance:.1%}")

            if risk > 0.5:
                st.error(
                    f"HIGH RISK — Week {selected_week}\n\n"
                    f"Curriculum compliance: {compliance:.1%}\n\n"
                    f"Recommended Action: "
                    f"Schedule immediate welfare check-in."
                )
            elif risk > 0.3:
                st.warning(
                    f"MEDIUM RISK — Week {selected_week}\n\n"
                    f"Curriculum compliance: {compliance:.1%}\n\n"
                    f"Recommended Action: "
                    f"Monitor closely for next 2 weeks."
                )
            else:
                st.success(
                    f"LOW RISK — Week {selected_week}\n\n"
                    f"Curriculum compliance: {compliance:.1%}\n\n"
                    f"Status: No action required."
                )

        with col2:
            st.markdown("**Weekly History**")
            display = visible[[
                'week', 'combined_score',
                'curriculum_compliance'
            ]].copy()
            display['Risk Level'] = display[
                'combined_score'
            ].apply(
                lambda x: 'HIGH' if x > 0.5
                else 'MEDIUM' if x > 0.3
                else 'LOW'
            )
            display['Phase'] = display['week'].apply(
                lambda w: 'Baseline' if w < 3
                else 'Monitoring'
            )
            st.dataframe(
                display.rename(columns={
                    'week': 'Week',
                    'combined_score': 'Risk Score',
                    'curriculum_compliance': 'Compliance'
                }),
                use_container_width=True
            )

# ─────────────────────────────────────────
# PAGE 4 — ADMIN PANEL
# ─────────────────────────────────────────
elif page == "[4] Admin Panel":

    st.markdown("## Admin Configuration Panel")
    st.markdown(
        "Lecturers and administrators can configure "
        "the system settings here."
    )
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "Academic Calendar",
        "Module Settings",
        "System Status"
    ])

    with tab1:
        st.subheader("Academic Calendar Configuration")
        st.info(
            "Define semester structure. The system will "
            "automatically exclude holiday weeks from "
            "anomaly detection to prevent false alarms."
        )

        selected_mod = st.selectbox(
            "Configure Module:",
            sorted(academic_calendar.keys())
        )
        cal = academic_calendar[selected_mod]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Holiday Weeks**")
            holiday_input = st.text_input(
                "Enter week numbers (comma separated)",
                value=", ".join(
                    map(str, cal['holiday_weeks'])
                )
            )
            st.markdown("**Exam Weeks**")
            exam_input = st.text_input(
                "Enter week numbers (comma separated)",
                value=", ".join(
                    map(str, cal['exam_weeks'])
                )
            )

        with col2:
            st.markdown("**Current Configuration**")
            st.json({
                'module': selected_mod,
                'holiday_weeks': cal['holiday_weeks'],
                'exam_weeks': cal['exam_weeks'],
                'excluded_from_detection': cal['holiday_weeks']
            })

        if st.button("Save Calendar", type="primary"):
            st.success(
                f"Calendar updated for "
                f"Module {selected_mod}."
            )

    with tab2:
        st.subheader("Module Expected Behavior")
        st.info(
            "Define what a normally engaged student "
            "is expected to do each week per module. "
            "The system uses this to calculate "
            "curriculum compliance scores."
        )

        selected_mod2 = st.selectbox(
            "Configure Module Expectations:",
            sorted(expected_df['code_module'].unique()),
            key="mod2"
        )

        mod_expected = expected_df[
            expected_df['code_module'] == selected_mod2
        ].head(10)

        st.markdown("**Current Expected Behavior**")
        st.dataframe(mod_expected, use_container_width=True)

        st.markdown("**Add New Week Expectation**")
        col1, col2, col3 = st.columns(3)
        with col1:
            new_week = st.number_input("Week", 0, 40, 1)
        with col2:
            new_clicks = st.number_input(
                "Expected Clicks", 0, 1000, 100
            )
        with col3:
            new_days = st.number_input(
                "Expected Active Days", 0, 7, 4
            )

        if st.button("Add Expectation", type="primary"):
            st.success(
                f"Added expectation for Week {new_week}: "
                f"{new_clicks} expected clicks, "
                f"{new_days} expected active days."
            )

    with tab3:
        st.subheader("System Status")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Component Status**")
            st.success("VAE V2 Model: Loaded")
            st.success("Feature Scaler: Loaded")
            st.success("Feature Dataset: Loaded")
            st.success("Expected Behavior Data: Loaded")
            st.success("Academic Calendar: Configured")

            st.markdown("**Performance Summary**")
            perf_data = {
                'Metric': [
                    'Overall AUC',
                    'Best Temporal Lag AUC',
                    'Top Module AUC',
                    'Students Monitored',
                    'Features Used'
                ],
                'Version 1': [
                    '0.5840', '0.5865',
                    'N/A', '24,064', '10'
                ],
                'Version 2': [
                    '0.6039', '0.6134',
                    '0.6312 (FFF)', '24,005', '10'
                ]
            }
            st.dataframe(
                pd.DataFrame(perf_data),
                use_container_width=True
            )

        with col2:
            st.markdown("**Module Detection Status**")
            module_status = pd.DataFrame({
                'Module': [
                    'AAA', 'BBB', 'CCC',
                    'DDD', 'EEE', 'FFF', 'GGG'
                ],
                'AUC Score': [
                    0.5390, 0.5633, 0.6091,
                    0.6026, 0.6126, 0.6312, 0.5562
                ],
                'Performance': [
                    'Fair', 'Fair', 'Good',
                    'Good', 'Good', 'Best', 'Fair'
                ]
            })
            st.dataframe(
                module_status,
                use_container_width=True
            )

# ─────────────────────────────────────────
# PAGE 5 — LIVE DETECTION
# ─────────────────────────────────────────
elif page == "[5] Live Detection":

    st.markdown("## Live Student Anomaly Detection")
    st.markdown(
        "Enter a student's weekly LMS activity data "
        "to generate a real-time behavioral risk assessment."
    )
    st.markdown("---")

    st.info(
        "System Overview: Enter weekly LMS activity data "
        "for any student. The VAE V2 model analyzes behavior "
        "against curriculum expectations and generates a "
        "personalized anomaly score with advisor recommendations."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Engagement Metrics")
        weekly_clicks = st.slider(
            "Weekly Total Clicks", 0, 500, 50
        )
        active_days = st.slider(
            "Active Days per Week", 0, 7, 3
        )
        content_diversity = st.slider(
            "Content Type Diversity", 0, 10, 3
        )
        forum_clicks = st.slider(
            "Forum Activity (Clicks)", 0, 100, 0
        )

    with col2:
        st.subheader("Study Activity Metrics")
        resource_clicks = st.slider(
            "Resource Views", 0, 200, 10
        )
        quiz_clicks = st.slider(
            "Quiz Attempts", 0, 50, 0
        )
        avg_submission_latency = st.slider(
            "Submission Latency (days, negative=early)",
            -30, 30, 0
        )
        late_submissions = st.slider(
            "Late Submission Count", 0, 10, 0
        )

    with col3:
        st.subheader("Module Configuration")
        selected_module_live = st.selectbox(
            "Student Module",
            ['AAA', 'BBB', 'CCC',
             'DDD', 'EEE', 'FFF', 'GGG']
        )
        current_week = st.slider(
            "Current Week Number", 3, 17, 5
        )

        exp = expected_df[
            (expected_df['code_module'] ==
             selected_module_live) &
            (expected_df['week'] == current_week)
        ]

        if len(exp) > 0:
            exp_clicks = exp.iloc[0]['expected_clicks']
            exp_days = exp.iloc[0]['expected_active_days']
        else:
            exp_clicks = 100
            exp_days = 4

        curriculum_compliance = min(
            weekly_clicks / (exp_clicks + 1), 2.0
        )
        click_compliance = min(
            weekly_clicks / (exp_clicks + 1), 2.0
        )

        st.markdown("**Curriculum Expectation**")
        st.markdown(
            f"Expected clicks this week: **{exp_clicks:.0f}**"
        )
        st.markdown(
            f"Expected active days: **{exp_days:.1f}**"
        )
        st.markdown(
            f"Student compliance: "
            f"**{curriculum_compliance:.1%}**"
        )

    st.markdown("---")

    if st.button(
        "Run Behavioral Analysis",
        type="primary",
        use_container_width=True
    ):
        input_data = np.array([[
            weekly_clicks, active_days,
            content_diversity, forum_clicks,
            resource_clicks, quiz_clicks,
            avg_submission_latency, late_submissions,
            curriculum_compliance, click_compliance
        ]])

        input_scaled = scaler.transform(input_data)
        input_tensor = torch.FloatTensor(input_scaled)

        with torch.no_grad():
            mu, logvar = vae_model.encode(input_tensor)
            recon = vae_model.decode(mu)
            recon_error = torch.mean(
                (recon - input_tensor) ** 2
            ).item()

        vae_score = math.log1p(recon_error)
        combined = (
            vae_score * 0.4 +
            (1 - curriculum_compliance) * 0.6
        )

        if combined > 0.5:
            risk_level = "HIGH RISK"
            risk_color = "error"
        elif combined > 0.3:
            risk_level = "MEDIUM RISK"
            risk_color = "warning"
        else:
            risk_level = "LOW RISK"
            risk_color = "success"

        st.markdown("---")
        st.subheader("Analysis Results")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("VAE Anomaly Score", f"{vae_score:.4f}")
        col2.metric("Combined Risk Score", f"{combined:.4f}")
        col3.metric(
            "Curriculum Compliance",
            f"{curriculum_compliance:.1%}"
        )
        col4.metric("Risk Classification", risk_level)

        if risk_color == "error":
            st.error(
                f"HIGH RISK DETECTED\n\n"
                f"Module: {selected_module_live} | "
                f"Week: {current_week}\n\n"
                f"Student is meeting only "
                f"{curriculum_compliance:.1%} of "
                f"expected curriculum engagement.\n\n"
                f"Recommended Action: Schedule an immediate "
                f"welfare check-in. Student behavioral patterns "
                f"indicate significant disengagement consistent "
                f"with early burnout indicators."
            )
        elif risk_color == "warning":
            st.warning(
                f"MEDIUM RISK DETECTED\n\n"
                f"Module: {selected_module_live} | "
                f"Week: {current_week}\n\n"
                f"Curriculum compliance: "
                f"{curriculum_compliance:.1%}\n\n"
                f"Recommended Action: Monitor for 2 additional "
                f"weeks. Consider a proactive welfare check-in "
                f"if the pattern continues."
            )
        else:
            st.success(
                f"LOW RISK — Student Engaged\n\n"
                f"Module: {selected_module_live} | "
                f"Week: {current_week}\n\n"
                f"Curriculum compliance: "
                f"{curriculum_compliance:.1%}\n\n"
                f"Status: Student behavior is within normal "
                f"expected range. No immediate action required."
            )

        st.subheader("Feature-Level Anomaly Breakdown")

        recon_np = recon.numpy()[0]
        input_np = input_scaled[0]
        feature_errors = np.abs(recon_np - input_np)

        breakdown_df = pd.DataFrame({
            'Feature': FEATURE_COLS,
            'Input Value': input_data[0],
            'Anomaly Contribution': feature_errors
        }).sort_values('Anomaly Contribution', ascending=True)

        fig = go.Figure(go.Bar(
            x=breakdown_df['Anomaly Contribution'],
            y=breakdown_df['Feature'],
            orientation='h',
            marker_color=[
                '#c0392b'
                if e > np.percentile(feature_errors, 75)
                else '#2d6a9f'
                for e in breakdown_df['Anomaly Contribution']
            ]
        ))
        fig.update_layout(
            template='plotly_dark',
            title='Feature Anomaly Breakdown '
                  '(Red = Most Anomalous Behaviors)',
            height=400,
            xaxis_title='Anomaly Contribution Score',
            yaxis_title='Feature'
        )
        st.plotly_chart(fig, use_container_width=True)

        top3 = breakdown_df.nlargest(
            3, 'Anomaly Contribution'
        )['Feature'].tolist()

        st.subheader("Advisor Decision Support Summary")
        st.info(
            f"Module: {selected_module_live} | "
            f"Week: {current_week} | "
            f"Risk Classification: {risk_level}\n\n"
            f"Most anomalous behavioral features: "
            f"{top3[0]}, {top3[1]}, {top3[2]}\n\n"
            f"Curriculum compliance rate: "
            f"{curriculum_compliance:.1%} "
            f"(Target: 100%)\n\n"
            f"Note: This system provides decision support "
            f"only. All welfare interventions require "
            f"professional judgment from a qualified "
            f"student welfare advisor.\n\n"
            f"Model: VAE V2 with Curriculum Compliance | "
            f"AUC: 0.6039 | R26-IT-059"
        )