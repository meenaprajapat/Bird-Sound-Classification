"""Plotly charts for the audio waveform, MFCC heatmap, and top-K predictions."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_waveform(audio, sample_rate):
    """Line chart of the raw audio waveform."""
    time = np.linspace(0, len(audio) / sample_rate, len(audio))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time,
        y=audio,
        mode="lines",
        name="Waveform",
        line=dict(color="#4CAF50", width=1),
    ))
    fig.update_layout(
        title="Audio Waveform",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        template="plotly_white",
        height=300,
        hovermode="x unified",
    )
    return fig


def plot_mfcc(mfccs):
    """Heatmap of the MFCC feature matrix."""
    fig = go.Figure(data=go.Heatmap(
        z=mfccs,
        x=np.arange(mfccs.shape[1]),
        y=np.arange(mfccs.shape[0]),
        colorscale="Viridis",
    ))
    fig.update_layout(
        title="MFCC Features Heatmap",
        xaxis_title="Time Frames",
        yaxis_title="MFCC Coefficients",
        height=400,
        template="plotly_white",
    )
    return fig


def plot_top_predictions(top_predictions):
    """Horizontal bar chart of the top-K predictions."""
    df = pd.DataFrame(top_predictions)

    fig = px.bar(
        df,
        x="Score",
        y="Species",
        orientation="h",
        text="Confidence",
        color="Score",
        color_continuous_scale="Greens",
    )
    fig.update_layout(
        title="Top 5 Predictions",
        xaxis_title="Confidence Score (%)",
        yaxis_title="Bird Species",
        showlegend=False,
        height=400,
        template="plotly_white",
    )
    fig.update_traces(textposition="outside")
    return fig
