# ============================================================
# Zomato Restaurant Success Predictor — Streamlit App
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Zomato Insights & Predictor",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #E23744;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Resources ───────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clean_sample_zomato.csv')
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

model = load_model()
df = load_data()

# ── Header ───────────────────────────────────────────────────
st.markdown('<p class="main-header">🍽️ Zomato Restaurant Insights</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analyze trends & predict restaurant success in Bangalore</p>', unsafe_allow_html=True)
st.divider()

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊 Insights Dashboard", "🔮 Success Predictor"])

# ==========================================
# TAB 1: INSIGHTS DASHBOARD
# ==========================================
with tab1:
    if df is not None:
        # ── Key Metrics Row ──────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📍 Total Restaurants",
                value=f"{len(df):,}",
                delta="Sample data")
        
        with col2:
            success_pct = (df['is_successful'].sum() / len(df)) * 100
            st.metric(
                label="⭐ Successful Rate",
                value=f"{success_pct:.1f}%",
                delta="Rating ≥ 4.0")
        
        with col3:
            avg_cost = df['approx_cost(for two people)'].mean()
            st.metric(
                label="💰 Avg Cost for Two",
                value=f"₹{avg_cost:.0f}",
                delta=f"Median: ₹{df['approx_cost(for two people)'].median():.0f}")
        
        with col4:
            st.metric(
                label="👥 Avg Votes",
                value=f"{df['votes'].mean():.0f}",
                delta=f"Max: {df['votes'].max():,}")
        
        st.divider()
        
        # ── Row 1: Cost vs Rating + Top Locations ────────────
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("💵 Cost vs Rating Analysis")
            scatter_df = df.sample(min(2000, len(df)))
            fig1 = px.scatter(
                scatter_df,
                x="approx_cost(for two people)",
                y="rate",
                color="is_successful",
                color_discrete_map={0: "#FF6B6B", 1: "#4ECDC4"},
                hover_data=["rest_type", "location"],
                labels={
                    "approx_cost(for two people)": "Cost (₹)",
                    "is_successful": "Success"
                }
            )
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with c2:
            st.subheader("📍 Top 10 Locations")
            top_loc = df['location'].value_counts().nlargest(10).reset_index()
            top_loc.columns = ['Location', 'Count']
            fig2 = px.bar(
                top_loc,
                x="Count",
                y="Location",
                orientation='h',
                color="Count",
                color_continuous_scale="Viridis"
            )
            fig2.update_layout(
                yaxis={'categoryorder':'total ascending'},
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # ── Row 2: Cuisines + Rating Distribution ────────────
        c3, c4 = st.columns(2)
        
        with c3:
            st.subheader("🍜 Top 10 Cuisines")
            top_cuisine = df['cuisines'].value_counts().nlargest(10).reset_index()
            top_cuisine.columns = ['Cuisine', 'Count']
            fig3 = px.bar(
                top_cuisine,
                x="Count",
                y="Cuisine",
                orientation='h',
                color="Count",
                color_continuous_scale="Sunset"
            )
            fig3.update_layout(
                yaxis={'categoryorder':'total ascending'},
                height=400
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        with c4:
            st.subheader("⭐ Rating Distribution")
            fig4 = px.histogram(
                df,
                x="rate",
                nbins=20,
                marginal="box",
                color_discrete_sequence=['#E23744']
            )
            fig4.update_layout(height=400)
            st.plotly_chart(fig4, use_container_width=True)
        
        # ── Row 3: Online Order + Table Booking Stats ────────
        st.divider()
        st.subheader("📱 Service Availability")
        
        c5, c6 = st.columns(2)
        
        with c5:
            online_counts = df['online_order'].value_counts()
            fig5 = go.Figure(data=[go.Pie(
                labels=online_counts.index,
                values=online_counts.values,
                hole=.4,
                marker_colors=['#4ECDC4', '#FF6B6B']
            )])
            fig5.update_layout(
                title_text="Online Order Availability",
                height=350
            )
            st.plotly_chart(fig5, use_container_width=True)
        
        with c6:
            table_counts = df['book_table'].value_counts()
            fig6 = go.Figure(data=[go.Pie(
                labels=table_counts.index,
                values=table_counts.values,
                hole=.4,
                marker_colors=['#95E1D3', '#F38181']
            )])
            fig6.update_layout(
                title_text="Table Booking Availability",
                height=350
            )
            st.plotly_chart(fig6, use_container_width=True)
    
    else:
        st.error("⚠️ Data not found! Run `train_model.py` first.")

# ==========================================
# TAB 2: SUCCESS PREDICTOR
# ==========================================
with tab2:
    st.header("🔮 Predict Restaurant Success")
    
    if model is None or df is None:
        st.error("⚠️ Model not found! Run `train_model.py` first.")
    else:
        st.info("💡 Enter restaurant details to predict if it will achieve **rating ≥ 4.0**")
        
        with st.form("prediction_form"):
            colA, colB = st.columns(2)
            
            with colA:
                st.subheader("📋 Location & Type")
                
                location = st.selectbox(
                    "Location",
                    sorted(df['location'].dropna().unique())
                )
                
                rest_type = st.selectbox(
                    "Restaurant Type",
                    sorted(df['rest_type'].dropna().unique())
                )
                
                listed_in_type = st.selectbox(
                    "Listed In (Type)",
                    sorted(df['listed_in(type)'].dropna().unique())
                )
                
                cuisines = st.selectbox(
                    "Cuisines",
                    sorted(df['cuisines'].value_counts().nlargest(50).index)
                )
            
            with colB:
                st.subheader("💼 Business Details")
                
                online_order = st.radio(
                    "Online Order Available?",
                    ["Yes", "No"],
                    horizontal=True
                )
                
                book_table = st.radio(
                    "Table Booking Available?",
                    ["Yes", "No"],
                    horizontal=True
                )
                
                cost = st.number_input(
                    "Approx Cost for Two (₹)",
                    min_value=50,
                    max_value=15000,
                    value=int(df['approx_cost(for two people)'].median()),
                    step=50
                )
                
                votes = st.number_input(
                    "Expected Votes",
                    min_value=0,
                    max_value=20000,
                    value=int(df['votes'].median()),
                    step=50
                )
            
            st.divider()
            submitted = st.form_submit_button(
                "🚀 Predict Success",
                type="primary",
                use_container_width=True
            )
        
        if submitted:
            input_data = pd.DataFrame([{
                'online_order': online_order,
                'book_table': book_table,
                'votes': votes,
                'location': location,
                'rest_type': rest_type,
                'cuisines': cuisines,
                'listed_in(type)': listed_in_type,
                'approx_cost(for two people)': cost
            }])
            
            with st.spinner("🔄 Analyzing..."):
                try:
                    prediction = model.predict(input_data)[0]
                    prob = model.predict_proba(input_data)[0]
                    
                    st.divider()
                    st.subheader("📊 Prediction Result")
                    
                    if prediction == 1:
                        st.success(f"### ✅ SUCCESSFUL Restaurant!")
                        st.metric(
                            label="Success Probability",
                            value=f"{prob[1]*100:.1f}%",
                            delta="High confidence"
                        )
                        st.balloons()
                    else:
                        st.error(f"### ❌ NOT Successful")
                        st.metric(
                            label="Failure Probability",
                            value=f"{prob[0]*100:.1f}%",
                            delta="Consider changes"
                        )
                    
                    # Probability bar
                    fig_prob = go.Figure(go.Bar(
                        x=[prob[0]*100, prob[1]*100],
                        y=['Not Successful', 'Successful'],
                        orientation='h',
                        marker=dict(color=['#FF6B6B', '#4ECDC4'])
                    ))
                    fig_prob.update_layout(
                        title="Confidence Breakdown",
                        xaxis_title="Probability (%)",
                        height=250
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error: {e}")

# ── Footer ───────────────────────────────────────────────────
st.divider()
st.caption("Built with Streamlit | Data: Zomato Bangalore")