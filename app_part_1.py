import streamlit as st
import numpy as np
import pandas as pd

# =============================================
# Configuração
# =============================================
st.set_page_config(page_title="Sistema Taguchi", layout="wide")
st.title("Sistema Taguchi")
st.caption("Upload de fatores e réplicas, seleção de OA, plano, S/N (da média e das réplicas) e efeitos médios com Delta.")

# Variável de interesse (aparece nas tabelas)
var_label = st.text_input("Variável de interesse (ex.: Produção de H₂)", "Produção de H₂")
st.write(f"**Variável definida:** {var_label}")


