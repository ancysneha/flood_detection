
import streamlit as st
import torch
import numpy as np
from PIL import Image
from cnn_model import FloodCNN
from lstm_model import LSTMForecast
import pandas as pd

st.title("AI Flood Detection & Forecasting")

st.header("Flood Detection (CNN)")
img = st.file_uploader("Upload satellite image", type=["jpg","png"])
if img:
    image = Image.open(img).resize((128,128))
    st.image(image)
    arr = np.array(image)/255.0
    arr = torch.tensor(arr).permute(2,0,1).unsqueeze(0).float()
    model = FloodCNN()
    model.eval()
    pred = model(arr)
    st.success(f"Flood Probability: {float(pred):.3f}")

st.header("Flood Forecasting (LSTM)")
csv = st.file_uploader("Upload rainfall CSV")
if csv:
    df = pd.read_csv(csv)
    st.dataframe(df)
    seq = torch.tensor(df.values, dtype=torch.float32).unsqueeze(0)
    lstm = LSTMForecast()
    lstm.eval()
    out = lstm(seq)
    st.warning(f"Forecasted Severity: {float(out):.3f}")
