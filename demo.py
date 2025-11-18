"""
Enhanced Interactive Dashboard for News Anomaly Detection
Supports combined Heading + Article analysis
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="News Anomaly Detection - Enhanced",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {padding: 0rem 1rem;}
    .stMetric {background-color: #f0f2f6; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);}
    h1 {color: #1f77b4; text-align: center;}
    h2 {color: #ff7f0e; border-bottom: 2px solid #ff7f0e; padding-bottom: 10px;}
    h3 {color: #2ca02c;}
    .highlight {background-color: #fff3cd; padding: 10px; border-radius: 5px; border-left: 4px solid #ffc107;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA
# ============================================================================
@st.cache_data
def load_results(filepath):
    """Load processed results"""
    try:
        return pd.read_csv(filepath, encoding='utf-8')
    except:
        return pd.read_csv(filepath, encoding='ISO-8859-1')

# ============================================================================
# HEADER
# ============================================================================
st.title("📰 Hyperlocal News Anomaly Detection Dashboard")

st.markdown("---")

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    uploaded_file = st.file_uploader("Upload Results CSV", type=['csv'])
    
    if uploaded_file:
        df = load_results(uploaded_file)
        st.success(f"✓ Loaded {len(df):,} articles")
    else:
        st.info("Please upload the results CSV file")
        df = None
    
    if df is not None:
        st.markdown("---")
        st.subheader("📊 Filters")
        
        # News type filter
        if 'NewsType' in df.columns:
            news_types = ['All'] + sorted(df['NewsType'].unique().tolist())
            selected_news_type = st.selectbox("News Type", news_types)
        else:
            selected_news_type = 'All'
        
        # Anomaly filter
        anomaly_filter = st.radio(
            "Show",
            ["All Articles", "Anomalies Only", "Normal Only"]
        )
        
        # Sentiment filter
        if 'sentiment_category' in df.columns:
            sentiment_filter = st.multiselect(
                "Sentiment",
                options=['positive', 'neutral', 'negative'],
                default=['positive', 'neutral', 'negative']
            )
        
        # Anomaly score threshold
        if 'combined_anomaly_score' in df.columns:
            st.subheader("🎚️ Anomaly Threshold")
            threshold = st.slider(
                "Minimum Anomaly Score",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05
            )
        
        # Date range
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            if df['Date'].notna().any():
                min_date = df['Date'].min().date()
                max_date = df['Date'].max().date()
                date_range = st.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )

# ============================================================================
# MAIN CONTENT
# ============================================================================
if df is not None:
    
    # Apply filters
    df_filtered = df.copy()
    
    if selected_news_type != 'All':
        df_filtered = df_filtered[df_filtered['NewsType'] == selected_news_type]
    
    if anomaly_filter == "Anomalies Only" and 'is_anomaly' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['is_anomaly'] == 1]
    elif anomaly_filter == "Normal Only" and 'is_anomaly' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['is_anomaly'] == 0]
    
    if 'sentiment_category' in df_filtered.columns and sentiment_filter:
        df_filtered = df_filtered[df_filtered['sentiment_category'].isin(sentiment_filter)]
    
    if 'combined_anomaly_score' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['combined_anomaly_score'] >= threshold]
    
    # ========================================================================
    # KEY METRICS
    # ========================================================================
    st.header("📊 Key Performance Indicators")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Total Articles", f"{len(df_filtered):,}")
    
    with col2:
        if 'is_anomaly' in df_filtered.columns:
            anomalies = df_filtered['is_anomaly'].sum()
            anomaly_pct = (anomalies / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
            st.metric("Anomalies", f"{anomalies:,}", f"{anomaly_pct:.1f}%")
    
    with col3:
        if 'source_discrepancy' in df_filtered.columns:
            discrepancies = df_filtered['source_discrepancy'].sum()
            disc_pct = (discrepancies / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
            st.metric("Source Issues", f"{discrepancies:,}", f"{disc_pct:.1f}%")
    
    with col4:
        if 'Extracted_Location' in df_filtered.columns:
            unique_locations = df_filtered['Extracted_Location'].nunique()
            st.metric("Locations", f"{unique_locations:,}")
    
    with col5:
        if 'sentiment_polarity' in df_filtered.columns:
            avg_sentiment = df_filtered['sentiment_polarity'].mean()
            sentiment_trend = "↑" if avg_sentiment > 0 else "↓" if avg_sentiment < 0 else "→"
            st.metric("Avg Sentiment", f"{avg_sentiment:.3f}", sentiment_trend)
    
    with col6:
        if 'combined_anomaly_score' in df_filtered.columns:
            high_risk = (df_filtered['combined_anomaly_score'] > 0.8).sum()
            st.metric("High Risk", f"{high_risk:,}", "⚠️")
    
    st.markdown("---")
    
    # ========================================================================
    # ANOMALY ANALYSIS
    # ========================================================================
    st.header("🔍 Anomaly Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Distribution", "Scores", "Trends"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'is_anomaly' in df_filtered.columns and 'NewsType' in df_filtered.columns:
                anomaly_by_type = df_filtered.groupby('NewsType').agg({
                    'is_anomaly': ['sum', 'count']
                }).reset_index()
                anomaly_by_type.columns = ['NewsType', 'Anomalies', 'Total']
                anomaly_by_type['Percentage'] = (anomaly_by_type['Anomalies'] / anomaly_by_type['Total'] * 100).round(2)
                
                fig = px.bar(
                    anomaly_by_type,
                    x='NewsType',
                    y='Percentage',
                    title='Anomaly Rate by News Type (%)',
                    color='Percentage',
                    color_continuous_scale='Reds',
                    text='Percentage'
                )
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'is_anomaly' in df_filtered.columns:
                anomaly_counts = df_filtered['is_anomaly'].value_counts()
                
                fig = go.Figure(data=[go.Pie(
                    labels=['Normal', 'Anomaly'],
                    values=[anomaly_counts.get(0, 0), anomaly_counts.get(1, 0)],
                    hole=.4,
                    marker_colors=['#2ca02c', '#d62728']
                )])
                fig.update_layout(title='Overall Anomaly Distribution')
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'combined_anomaly_score' in df_filtered.columns:
                fig = px.histogram(
                    df_filtered,
                    x='combined_anomaly_score',
                    nbins=50,
                    title='Combined Anomaly Score Distribution',
                    labels={'combined_anomaly_score': 'Anomaly Score'},
                    color_discrete_sequence=['#ff7f0e']
                )
                fig.add_vline(x=0.5, line_dash="dash", line_color="red", 
                             annotation_text="Threshold")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'anomaly_score_if' in df_filtered.columns:
                fig = px.box(
                    df_filtered,
                    x='is_anomaly',
                    y='anomaly_score_if',
                    title='Isolation Forest Score by Classification',
                    labels={'is_anomaly': 'Anomaly Status', 'anomaly_score_if': 'IF Score'},
                    color='is_anomaly',
                    color_discrete_map={0: '#2ca02c', 1: '#d62728'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if 'Date' in df_filtered.columns and df_filtered['Date'].notna().any():
            col1, col2 = st.columns(2)
            
            with col1:
                daily_stats = df_filtered.groupby(df_filtered['Date'].dt.date).agg({
                    'is_anomaly': ['sum', 'count']
                }).reset_index()
                daily_stats.columns = ['Date', 'Anomalies', 'Total']
                daily_stats['Rate'] = (daily_stats['Anomalies'] / daily_stats['Total'] * 100).round(2)
                
                fig = px.line(
                    daily_stats,
                    x='Date',
                    y='Rate',
                    title='Daily Anomaly Rate Trend',
                    markers=True,
                    labels={'Rate': 'Anomaly Rate (%)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(
                    df_filtered[df_filtered['is_anomaly'] == 1],
                    x='Date',
                    y='combined_anomaly_score',
                    title='Anomaly Scores Over Time',
                    color='combined_anomaly_score',
                    size='combined_anomaly_score',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================================================
    # SENTIMENT ANALYSIS
    # ========================================================================
    st.header("💭 Sentiment Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'sentiment_category' in df_filtered.columns:
            sentiment_counts = df_filtered['sentiment_category'].value_counts()
            
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title='Sentiment Distribution',
                color_discrete_sequence=['#2ca02c', '#ff7f0e', '#d62728'],
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'sentiment_polarity' in df_filtered.columns and 'is_anomaly' in df_filtered.columns:
            fig = px.violin(
                df_filtered,
                x='is_anomaly',
                y='sentiment_polarity',
                title='Sentiment Polarity: Normal vs Anomalous',
                color='is_anomaly',
                color_discrete_map={0: '#2ca02c', 1: '#d62728'},
                box=True
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        if 'sentiment_intensity' in df_filtered.columns:
            fig = px.histogram(
                df_filtered,
                x='sentiment_intensity',
                title='Sentiment Intensity Distribution',
                nbins=30,
                color_discrete_sequence=['#9467bd']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================================================
    # GEOGRAPHIC ANALYSIS
    # ========================================================================
    st.header("🌍 Geographic Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Extracted_Location' in df_filtered.columns:
            top_locations = df_filtered['Extracted_Location'].value_counts().head(15)
            
            locations_df = pd.DataFrame({
                'Location': top_locations.index,
                'Count': top_locations.values
            })
            
            fig = px.bar(
                locations_df,
                y='Location',
                x='Count',
                orientation='h',
                title='Top 15 Mentioned Locations',
                color='Count',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'source_discrepancy' in df_filtered.columns and 'Extracted_Location' in df_filtered.columns:
            discrepancy_by_loc = df_filtered[df_filtered['Extracted_Location'] != 'Unknown'].groupby(
                'Extracted_Location'
            )['source_discrepancy'].sum().sort_values(ascending=False).head(10)
            
            discrepancy_df = pd.DataFrame({
                'Location': discrepancy_by_loc.index,
                'Discrepancies': discrepancy_by_loc.values
            })
            
            fig = px.bar(
                discrepancy_df,
                y='Location',
                x='Discrepancies',
                orientation='h',
                title='Top 10 Locations with Source Discrepancies',
                color_discrete_sequence=['#e377c2']
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================================================
    # CONTENT ANALYSIS
    # ========================================================================
    st.header("📝 Content Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'combined_length' in df_filtered.columns:
            fig = px.histogram(
                df_filtered,
                x='combined_length',
                title='Combined Text Length Distribution',
                nbins=50,
                color_discrete_sequence=['#17becf']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'word_count' in df_filtered.columns:
            fig = px.box(
                df_filtered,
                y='word_count',
                x='is_anomaly',
                title='Word Count: Normal vs Anomalous',
                color='is_anomaly',
                color_discrete_map={0: '#2ca02c', 1: '#d62728'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        if 'lexical_diversity' in df_filtered.columns:
            fig = px.histogram(
                df_filtered,
                x='lexical_diversity',
                title='Lexical Diversity Distribution',
                nbins=30,
                color_discrete_sequence=['#bcbd22']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================================================
    # ADVANCED METRICS
    # ========================================================================
    st.header("📈 Advanced Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'heading_ratio' in df_filtered.columns and 'is_anomaly' in df_filtered.columns:
            fig = px.scatter(
                df_filtered.sample(min(1000, len(df_filtered))),
                x='heading_ratio',
                y='combined_anomaly_score',
                color='is_anomaly',
                title='Heading Ratio vs Anomaly Score',
                color_discrete_map={0: '#2ca02c', 1: '#d62728'},
                opacity=0.6
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'topic_diversity' in df_filtered.columns:
            fig = px.histogram(
                df_filtered,
                x='topic_diversity',
                color='is_anomaly',
                title='Topic Diversity by Anomaly Status',
                nbins=30,
                barmode='overlay',
                color_discrete_map={0: '#2ca02c', 1: '#d62728'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================================================
    # DETAILED ANOMALY INSPECTOR
    # ========================================================================
    st.header("🔎 Detailed Anomaly Inspector")
    
    if 'is_anomaly' in df_filtered.columns:
        # Add risk level
        if 'combined_anomaly_score' in df_filtered.columns:
            df_filtered['Risk_Level'] = pd.cut(
                df_filtered['combined_anomaly_score'],
                bins=[0, 0.5, 0.7, 0.9, 1.0],
                labels=['Low', 'Medium', 'High', 'Critical']
            )
        
        anomalous_articles = df_filtered[df_filtered['is_anomaly'] == 1].copy()
        
        if len(anomalous_articles) > 0:
            # Sort by anomaly score
            anomalous_articles = anomalous_articles.sort_values('combined_anomaly_score', ascending=False)
            
            # Select columns to display
            display_cols = ['Heading', 'NewsType', 'Extracted_Location', 
                           'sentiment_category', 'combined_anomaly_score']
            
            if 'Date' in anomalous_articles.columns:
                display_cols.insert(0, 'Date')
            if 'source_discrepancy' in anomalous_articles.columns:
                display_cols.append('source_discrepancy')
            if 'Risk_Level' in anomalous_articles.columns:
                display_cols.append('Risk_Level')
            
            # Filter available columns
            display_cols = [col for col in display_cols if col in anomalous_articles.columns]
            
            # Display count
            st.markdown(f"""
            <div class="highlight">
            📊 <strong>{len(anomalous_articles)}</strong> anomalous articles found 
            (sorted by anomaly score, highest first)
            </div>
            """, unsafe_allow_html=True)
            
            # Display table
            st.dataframe(
                anomalous_articles[display_cols].head(50),
                use_container_width=True,
                height=400
            )
            
            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                csv = anomalous_articles.to_csv(index=False)
                st.download_button(
                    label="📥 Download All Anomalies",
                    data=csv,
                    file_name="anomalous_articles.csv",
                    mime="text/csv"
                )
            
            with col2:
                high_risk = anomalous_articles[anomalous_articles['combined_anomaly_score'] > 0.8]
                if len(high_risk) > 0:
                    csv_high = high_risk.to_csv(index=False)
                    st.download_button(
                        label="⚠️ Download High Risk Only",
                        data=csv_high,
                        file_name="high_risk_articles.csv",
                        mime="text/csv"
                    )
        else:
            st.info("✨ No anomalies detected in the filtered data")
    
    st.markdown("---")
    
    # ========================================================================
    # STATISTICS SUMMARY
    # ========================================================================
    st.header("📊 Statistical Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Sentiment Statistics")
        if 'sentiment_polarity' in df_filtered.columns:
            stats = df_filtered[['sentiment_polarity', 'sentiment_subjectivity', 'sentiment_intensity']].describe()
            st.dataframe(stats, use_container_width=True)
    
    with col2:
        st.subheader("Content Statistics")
        if 'combined_length' in df_filtered.columns:
            stats = df_filtered[['combined_length', 'word_count', 'lexical_diversity']].describe()
            st.dataframe(stats, use_container_width=True)
    
    with col3:
        st.subheader("Anomaly Statistics")
        if 'combined_anomaly_score' in df_filtered.columns:
            stats = df_filtered[['combined_anomaly_score', 'anomaly_score_if']].describe()
            st.dataframe(stats, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================================================
    # CORRELATION MATRIX
    # ========================================================================
    with st.expander("📈 Feature Correlation Matrix"):
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Select key features
            key_features = [col for col in numeric_cols if any(
                x in col for x in ['sentiment', 'anomaly', 'length', 'word', 'topic_diversity']
            )][:10]  # Limit to 10 features
            
            if key_features:
                corr_matrix = df_filtered[key_features].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    title='Feature Correlation Heatmap',
                    color_continuous_scale='RdBu_r',
                    aspect='auto'
                )
                st.plotly_chart(fig, use_container_width=True)

else:
    # Welcome screen
    st.markdown("""
    <div style='text-align: center; padding: 50px;'>
        <h2>Enhanced News Anomaly Detection Dashboard</h2>
        <p style='font-size: 18px; color: #666;'>
        Please upload your results CSV file in the sidebar to begin analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🔍 Detection Features
        - Multi-method anomaly detection
        - Combined text analysis
        - Source attribution
        - Risk level classification
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Visualizations
        - Interactive charts & graphs
        - Temporal trend analysis
        - Geographic distribution
        - Statistical summaries
        """)
    
    with col3:
        st.markdown("""
        ### ⚡ Capabilities
        - Real-time filtering
        - Multiple view modes
        - Export functionality
        - High-risk alerts
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Hyperlocal News Anomaly Detection System v2.0</strong></p>
    <p>Enhanced with Combined Heading + Article Analysis | Powered by Advanced NLP & ML</p>
    <p>🔒 Secure | ⚡ Fast | 📊 Accurate</p>
</div>
""", unsafe_allow_html=True)