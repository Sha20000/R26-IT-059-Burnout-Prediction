import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os

# Page config
st.set_page_config(
    page_title="Behavioral Anomaly Detection | R26-IT-059",
    page_icon="🧠",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    anomaly = pd.read_csv(
        "../results/metrics/anomaly_scores.csv"
    )
    features = pd.read_csv(
        "../results/metrics/features_weekly.csv"
    )
    return anomaly, features

anomaly_df, features_df = load_data()

# Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/e/e9/SLIIT_logo.png", width=150)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📊 System Overview", 
     "👤 Student Detail", 
     "📈 Model Results",
     "🔍 Predict New Student"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**R26-IT-059**")
st.sidebar.markdown("IT22215710 | Karunarathne D C")
st.sidebar.markdown("Behavioral Anomaly Detection VAE")

# ─────────────────────────────────────────
# PAGE 1 — SYSTEM OVERVIEW
# ─────────────────────────────────────────
if page == "📊 System Overview":
    st.title("🧠 Behavioral Anomaly Detection System")
    st.markdown("### Explainable Multi-Modal AI for Academic Burnout Prediction")
    st.markdown("**R26-IT-059 | SLIIT | IT22215710 Karunarathne D C**")
    st.markdown("---")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    total_students = anomaly_df['student_id'].nunique()
    high_risk = anomaly_df[
        anomaly_df['high_anomaly_flag'] == 1
    ]['student_id'].nunique()
    avg_score = anomaly_df['behavioral_risk_score'].mean()
    auc = 0.5840

    col1.metric(
        "Total Students Monitored", 
        f"{total_students:,}"
    )
    col2.metric(
        "High Risk Students", 
        f"{high_risk:,}",
        delta=f"{high_risk/total_students*100:.1f}% of total"
    )
    col3.metric(
        "Avg Behavioral Risk Score", 
        f"{avg_score:.4f}"
    )
    col4.metric(
        "Model ROC-AUC", 
        f"{auc}",
        delta="+16.8% over random"
    )

    st.markdown("---")

    # Risk distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Risk Score Distribution")
        fig = px.histogram(
            anomaly_df,
            x='behavioral_risk_score',
            color='actual_label',
            color_discrete_map={0: 'steelblue', 1: 'red'},
            labels={
                'behavioral_risk_score': 'Behavioral Risk Score',
                'actual_label': 'At Risk'
            },
            title='Risk Score Distribution by At-Risk Status',
            barmode='overlay',
            opacity=0.7
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 Weekly Risk Trend")
        weekly_risk = anomaly_df.groupby('week')[
            'behavioral_risk_score'
        ].mean().reset_index()

        fig2 = px.line(
            weekly_risk,
            x='week',
            y='behavioral_risk_score',
            title='Average Behavioral Risk Score per Week',
            markers=True
        )
        fig2.add_hline(
            y=anomaly_df['behavioral_risk_score'].quantile(0.75),
            line_dash="dash",
            line_color="red",
            annotation_text="High Risk Threshold"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Top at-risk students table
    st.subheader("🚨 Top 20 Highest Risk Students")
    top_risk = anomaly_df.groupby('student_id').agg(
        avg_risk=('behavioral_risk_score', 'mean'),
        max_risk=('behavioral_risk_score', 'max'),
        high_anomaly_weeks=('high_anomaly_flag', 'sum'),
        actual_label=('actual_label', 'first')
    ).reset_index().sort_values(
        'avg_risk', ascending=False
    ).head(20)

    top_risk['risk_level'] = top_risk['avg_risk'].apply(
        lambda x: '🔴 HIGH' if x > top_risk['avg_risk'].quantile(0.75)
        else '🟡 MEDIUM' if x > top_risk['avg_risk'].quantile(0.5)
        else '🟢 LOW'
    )

    st.dataframe(
        top_risk[[
            'student_id', 'avg_risk', 
            'max_risk', 'high_anomaly_weeks', 
            'risk_level', 'actual_label'
        ]].rename(columns={
            'student_id': 'Student ID',
            'avg_risk': 'Avg Risk Score',
            'max_risk': 'Peak Risk Score',
            'high_anomaly_weeks': 'High Risk Weeks',
            'risk_level': 'Risk Level',
            'actual_label': 'Actually At-Risk'
        }),
        use_container_width=True
    )

# ─────────────────────────────────────────
# PAGE 2 — STUDENT DETAIL
# ─────────────────────────────────────────
elif page == "👤 Student Detail":
    st.title("👤 Student Behavioral Profile")
    st.markdown("---")

    # Student selector
    student_ids = sorted(anomaly_df['student_id'].unique())
    selected_student = st.selectbox(
        "Select Student ID",
        student_ids
    )

    # Get student data
    student_anomaly = anomaly_df[
        anomaly_df['student_id'] == selected_student
    ].sort_values('week')

    student_features = features_df[
        features_df['id_student'] == selected_student
    ].sort_values('week')

    # Student summary
    avg_risk = student_anomaly['behavioral_risk_score'].mean()
    max_risk = student_anomaly['behavioral_risk_score'].max()
    high_weeks = student_anomaly['high_anomaly_flag'].sum()
    at_risk = student_anomaly['actual_label'].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Student ID", selected_student)
    col2.metric("Avg Risk Score", f"{avg_risk:.4f}")
    col3.metric("Peak Risk Score", f"{max_risk:.4f}")
    col4.metric(
        "Status",
        "🔴 AT RISK" if at_risk == 1 else "🟢 NORMAL"
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Weekly anomaly score
        st.subheader("📈 Weekly Behavioral Risk Score")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=student_anomaly['week'],
            y=student_anomaly['behavioral_risk_score'],
            mode='lines+markers',
            name='Risk Score',
            line=dict(color='steelblue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=student_anomaly[
                student_anomaly['high_anomaly_flag'] == 1
            ]['week'],
            y=student_anomaly[
                student_anomaly['high_anomaly_flag'] == 1
            ]['behavioral_risk_score'],
            mode='markers',
            name='High Anomaly',
            marker=dict(color='red', size=12, symbol='x')
        ))
        fig.update_layout(
            xaxis_title='Week',
            yaxis_title='Risk Score',
            title=f'Student {selected_student} Risk Timeline'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Feature patterns
        st.subheader("📊 Behavioral Feature Patterns")
        FEATURE_COLS = [
            'weekly_total_clicks', 'active_days',
            'content_diversity', 'forum_posts',
            'quiz_attempts', 'resource_views',
            'night_activity', 'inactive_weeks_count'
        ]

        available_features = [
            f for f in FEATURE_COLS 
            if f in student_features.columns
        ]

        if len(student_features) > 0:
            fig2 = px.line(
                student_features,
                x='week',
                y=available_features[:4],
                title=f'Top Features — Student {selected_student}'
            )
            st.plotly_chart(fig2, use_container_width=True)

    # SHAP explanation
    st.markdown("---")
    st.subheader("🔍 SHAP Feature Importance")

    shap_img_path = "../results/figures/shap_feature_importance.png"
    if os.path.exists(shap_img_path):
        img = Image.open(shap_img_path)
        st.image(img, caption="SHAP Feature Importance — VAE Anomaly Detection")
    
    st.info(
        "**Top 3 behavioral features driving anomalies:**\n\n"
        "1. **weekly_total_clicks** (0.0182) — Total LMS interactions\n"
        "2. **content_diversity** (0.0168) — Variety of content accessed\n"
        "3. **night_activity** (0.0150) — Late night LMS usage"
    )

# ─────────────────────────────────────────
# PAGE 3 — MODEL RESULTS
# ─────────────────────────────────────────
elif page == "📈 Model Results":
    st.title("📈 Model Evaluation Results")
    st.markdown("---")

    # Key results
    st.subheader("🏆 Key Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("ROC-AUC", "0.5840", "+16.8% over random")
    col2.metric("Random Baseline", "0.5000")
    col3.metric("Training Loss", "0.1365")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # ROC curve
        st.subheader("📊 ROC Curve")
        roc_path = "../results/figures/roc_curve_combined.png"
        if os.path.exists(roc_path):
            img = Image.open(roc_path)
            st.image(img, caption="ROC Curve — Behavioral Anomaly Detection")

    with col2:
        # Temporal lag
        st.subheader("⏱️ Temporal Lag Analysis")
        lag_path = "../results/figures/temporal_lag_analysis.png"
        if os.path.exists(lag_path):
            img = Image.open(lag_path)
            st.image(img, caption="Temporal Lag Analysis")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Training loss
        st.subheader("📉 VAE Training Loss")
        loss_path = "../results/figures/vae_training_loss.png"
        if os.path.exists(loss_path):
            img = Image.open(loss_path)
            st.image(img, caption="VAE Training Convergence")

    with col2:
        # Feature distributions
        st.subheader("📊 Feature Distributions")
        feat_path = "../results/figures/feature_distributions.png"
        if os.path.exists(feat_path):
            img = Image.open(feat_path)
            st.image(img, caption="LMS Behavioral Feature Distributions")

    st.markdown("---")

    # Temporal lag table
    st.subheader("⏱️ Temporal Lag Analysis Results")
    lag_data = {
        'Lag (Weeks Ahead)': [0,1,2,3,4,5,6,7,8],
        'ROC-AUC': [
            0.5865, 0.5738, 0.5704, 0.5655,
            0.5589, 0.5549, 0.5607, 0.5584, 0.5584
        ],
        'Above Random': [
            '✅','✅','✅','✅','✅','✅','✅','✅','✅'
        ]
    }
    lag_df = pd.DataFrame(lag_data)
    st.dataframe(lag_df, use_container_width=True)

    st.markdown("---")

    # Research novelty
    st.subheader("🔬 Research Novelty")
    st.success(
        "**First unsupervised VAE on LMS behavioral logs for burnout detection**\n\n"
        "No existing paper combines:\n"
        "- ✅ Unsupervised VAE (zero labels during training)\n"
        "- ✅ LMS behavioral logs (Moodle/OULAD data)\n"
        "- ✅ 10 engagement behavioral features\n"
        "- ✅ Burnout/dropout detection target\n"
        "- ✅ SHAP-based behavioral explainability\n"
        "- ✅ Temporal lag analysis across 8 weeks\n\n"
        "Confirmed by literature review of 13 papers across IEEE, ACM, Nature and MDPI."
    )
    
# ─────────────────────────────────────────
# PAGE 4 — REAL TIME STUDENT TRACKING
# ─────────────────────────────────────────
elif page == "🔍 Predict New Student":
    st.title("🔍 Real-Time Student Behavioral Tracking")
    st.markdown("### Week by Week Anomaly Detection")
    st.markdown("**R26-IT-059 | IT22215710 | Personalized VAE Detection**")
    st.markdown("---")

    import torch
    import torch.nn as nn
    import pickle
    import numpy as np
    import math

    # Define VAE
    class VAE(nn.Module):
        def __init__(self, input_dim=10, latent_dim=6):
            super(VAE, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU()
            )
            self.fc_mu = nn.Linear(16, latent_dim)
            self.fc_logvar = nn.Linear(16, latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, input_dim)
            )
        def encode(self, x):
            h = self.encoder(x)
            return self.fc_mu(h), self.fc_logvar(h)
        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        def decode(self, z):
            return self.decoder(z)
        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar

    # Load model and scaler
    @st.cache_resource
    def load_model():
        device = torch.device('cpu')
        vae = VAE(input_dim=10, latent_dim=6).to(device)
        vae.load_state_dict(
            torch.load(
                '../models/saved/vae_behavioral.pt',
                map_location=device
            )
        )
        vae.eval()
        with open('../models/saved/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return vae, scaler

    vae, scaler = load_model()

    FEATURE_COLS = [
        'weekly_total_clicks', 'active_days',
        'content_diversity', 'forum_posts',
        'forum_replies', 'quiz_attempts',
        'resource_views', 'night_activity',
        'days_since_login', 'inactive_weeks_count'
    ]

    # Load feature data
    features_data = pd.read_csv(
        "../results/metrics/features_weekly.csv"
    )

    # Student selector
    st.subheader("👤 Select Student")
    student_ids = sorted(
        features_data['id_student'].unique()
    )
    selected_student = st.selectbox(
        "Select Student ID to Track",
        student_ids
    )

    # Get student data
    student_data = features_data[
        features_data['id_student'] == selected_student
    ].sort_values('week').reset_index(drop=True)

    # Week slider
    max_week = int(student_data['week'].max())
    selected_week = st.slider(
        "▶️ Simulate Time — Show data up to week:",
        min_value=1,
        max_value=max_week,
        value=4,
        help="Move slider to simulate weeks passing"
    )

    # Show student info
    at_risk = student_data['at_risk'].iloc[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("Student ID", selected_student)
    col2.metric(
        "Actual Status",
        "🔴 AT RISK" if at_risk == 1 else "🟢 NORMAL"
    )
    col3.metric("Weeks Tracked", selected_week)

    st.markdown("---")

    # Get data up to selected week
    visible_data = student_data[
        student_data['week'] <= selected_week
    ].copy()

    # Baseline = first 3 weeks
    baseline_data = visible_data[
        visible_data['week'] < 3
    ][FEATURE_COLS].values

    # Generate anomaly scores week by week
    anomaly_scores = []
    weeks = []
    risk_levels = []

    for _, row in visible_data.iterrows():
        week_num = row['week']
        X = row[FEATURE_COLS].values.reshape(1, -1)

        # Scale using global scaler
        X_scaled = scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)

        # Get deterministic anomaly score
        with torch.no_grad():
            mu, logvar = vae.encode(X_tensor)
            reconstructed = vae.decode(mu)
            recon_error = torch.mean(
                (reconstructed - X_tensor) ** 2
            ).item()

        score = math.log1p(recon_error)
        anomaly_scores.append(score)
        weeks.append(week_num)

        # Risk level
        if score > 0.08:
            risk_levels.append('HIGH')
        elif score > 0.04:
            risk_levels.append('MEDIUM')
        else:
            risk_levels.append('LOW')

    # Create results dataframe
    results_df = pd.DataFrame({
        'week': weeks,
        'anomaly_score': anomaly_scores,
        'risk_level': risk_levels
    })

    # Color map
    color_map = {
        'LOW': 'green',
        'MEDIUM': 'orange',
        'HIGH': 'red'
    }

    # Plot anomaly trend
    st.subheader("📈 Weekly Anomaly Score Trend")

    fig = go.Figure()

    # Baseline period shading
    fig.add_vrect(
        x0=0, x1=2,
        fillcolor="blue",
        opacity=0.1,
        annotation_text="Baseline Period",
        annotation_position="top left"
    )

    # Anomaly score line
    fig.add_trace(go.Scatter(
        x=results_df['week'],
        y=results_df['anomaly_score'],
        mode='lines+markers',
        name='Anomaly Score',
        line=dict(color='steelblue', width=2),
        marker=dict(
            size=12,
            color=[
                color_map[r] 
                for r in results_df['risk_level']
            ]
        )
    ))

    # High risk threshold
    fig.add_hline(
        y=0.08,
        line_dash="dash",
        line_color="red",
        annotation_text="High Risk Threshold"
    )

    # Medium threshold
    fig.add_hline(
        y=0.04,
        line_dash="dash",
        line_color="orange",
        annotation_text="Medium Risk Threshold"
    )

    fig.update_layout(
        xaxis_title='Week',
        yaxis_title='Anomaly Score',
        title=f'Student {selected_student} — Behavioral Anomaly Timeline',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Weekly breakdown table
    st.subheader("📊 Week by Week Analysis")

    display_df = results_df.copy()
    display_df['risk_emoji'] = display_df['risk_level'].map({
        'LOW': '🟢 LOW',
        'MEDIUM': '🟡 MEDIUM',
        'HIGH': '🔴 HIGH'
    })
    display_df['phase'] = display_df['week'].apply(
        lambda w: '📚 Baseline' if w < 3 else '👁️ Monitoring'
    )

    st.dataframe(
        display_df[[
            'week', 'anomaly_score',
            'risk_emoji', 'phase'
        ]].rename(columns={
            'week': 'Week',
            'anomaly_score': 'Anomaly Score',
            'risk_emoji': 'Risk Level',
            'phase': 'Phase'
        }),
        use_container_width=True
    )

    st.markdown("---")

    # Current week analysis
    st.subheader(f"🔍 Week {selected_week} Detailed Analysis")

    current_row = visible_data[
        visible_data['week'] == selected_week
    ]

    if len(current_row) > 0:
        current_features = current_row[
            FEATURE_COLS
        ].values[0]
        current_score = anomaly_scores[-1]
        current_risk = risk_levels[-1]

        # Feature values
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Current Week Features:**")
            for feat, val in zip(
                FEATURE_COLS, current_features
            ):
                st.markdown(f"- **{feat}**: {val:.1f}")

        with col2:
            st.markdown("**Anomaly Summary:**")
            if current_risk == 'HIGH':
                st.error(
                    f"🔴 HIGH RISK at Week {selected_week}\n\n"
                    f"Anomaly Score: {current_score:.4f}\n\n"
                    f"Student behavior has deviated significantly "
                    f"from normal patterns!\n\n"
                    f"**Action: Immediate welfare check-in recommended!**"
                )
            elif current_risk == 'MEDIUM':
                st.warning(
                    f"🟡 MEDIUM RISK at Week {selected_week}\n\n"
                    f"Anomaly Score: {current_score:.4f}\n\n"
                    f"Some behavioral changes detected.\n\n"
                    f"**Action: Monitor closely next 2 weeks.**"
                )
            else:
                st.success(
                    f"🟢 LOW RISK at Week {selected_week}\n\n"
                    f"Anomaly Score: {current_score:.4f}\n\n"
                    f"Student behavior appears normal.\n\n"
                    f"**Action: No action required.**"
                )

    st.markdown("---")

    # Behavioral features chart
    st.subheader("📊 Behavioral Features Over Time")

    top_features = [
        'weekly_total_clicks',
        'active_days',
        'forum_posts',
        'quiz_attempts'
    ]

    available = [
        f for f in top_features
        if f in visible_data.columns
    ]

    fig2 = px.line(
        visible_data,
        x='week',
        y=available,
        title=f'Student {selected_student} — Feature Trends',
        markers=True
    )
    fig2.add_vrect(
        x0=0, x1=2,
        fillcolor="blue",
        opacity=0.1,
        annotation_text="Baseline"
    )
    st.plotly_chart(fig2, use_container_width=True)