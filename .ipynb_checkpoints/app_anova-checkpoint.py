import streamlit as st
import numpy as np
import pandas as pd
import math
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import io
from itertools import product
from datetime import datetime

# (depois dos imports já existentes)
try:
    from scipy.stats import f as f_dist
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False
    f_dist = None


# =============================================
# Configuração
# =============================================
st.set_page_config(page_title="Taguchi App", layout="wide")
st.title("Taguchi App")
st.caption(
    """
    <div style="font-size:16px; font-weight:bold;">
        Taguchi App — Planejamento e Análise Experimental Taguchi — Versão v25.01<br><br>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Estado do fluxo (wizard) ---
if 'step' not in st.session_state:
    st.session_state['step'] = 'start'  # 'start' | 'results'

# =============================================
# pyDOE3 (opcional)
# =============================================
try:
    from pyDOE3 import get_orthogonal_array
    HAS_PYDOE3 = True
except Exception:
    HAS_PYDOE3 = False
    get_orthogonal_array = None

# ---------------------------------------------
# Variável de interesse
# ---------------------------------------------
var_label = st.text_input(
    "Variável de interesse (ex.: Produção de H₂)",
    "Produção de H₂",
    help="Digite o nome da variável de interesse. Tecle ENTER ao finalizar!"
)
if var_label:
    st.success(f"✅ **Variável definida:** {var_label}")
else:
    st.write("**Variável definida:** Produção de H₂")

# ---------------------------------------------
# Utilitários de OA
# ---------------------------------------------
def built_in_catalog():
    return {
        "L4(2^3)"     : {"cols2": 3,  "cols3": 0,  "n": 4},
        "L8(2^7)"     : {"cols2": 7,  "cols3": 0,  "n": 8},
        "L9(3^4)"     : {"cols2": 0,  "cols3": 4,  "n": 9},
        "L12(2^11)"   : {"cols2": 11, "cols3": 0,  "n": 12},
        "L16(2^15)"   : {"cols2": 15, "cols3": 0,  "n": 16},
        "L18(2^1 3^7)": {"cols2": 1,  "cols3": 7,  "n": 18},
        "L27(3^13)"   : {"cols2": 0,  "cols3": 13, "n": 27},
    }

PYDOE3_NAME_MAP = {
    "L18(2^1 3^7)": "L18(6^1 3^6)",
    "L27(3^13)":    "L27(2^1 3^12)",
}

def oa_from_name(name: str) -> np.ndarray:
    # 1) Tenta pyDOE3
    if HAS_PYDOE3 and get_orthogonal_array is not None:
        try:
            lookup = PYDOE3_NAME_MAP.get(name, name)
            arr = np.asarray(get_orthogonal_array(lookup), dtype=int)
            if arr.min() == 1:
                arr = arr - 1
            return arr
        except Exception:
            pass
    # 2) Fallbacks internos (0-based)
    import numpy as np
    if name == "L4(2^3)":
        return np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]], dtype=int)
    if name == "L8(2^7)":
        return np.array([
            [0,0,0,0,0,0,0],
            [0,0,0,1,1,1,1],
            [0,1,1,0,0,1,1],
            [0,1,1,1,1,0,0],
            [1,0,1,0,1,0,1],
            [1,0,1,1,0,1,0],
            [1,1,0,0,1,1,0],
            [1,1,0,1,0,0,1],
        ], dtype=int)
    if name == "L9(3^4)":
        return np.array([
            [0,0,0,0],[0,1,1,1],[0,2,2,2],
            [1,0,1,2],[1,1,2,0],[1,2,0,1],
            [2,0,2,1],[2,1,0,2],[2,2,1,0]
        ], dtype=int)
    if name == "L16(2^15)":
        arr12 = np.array([
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,2,2,2,2,2,2,2,2],
            [1,1,1,2,2,2,2,1,1,1,1,2,2,2,2],
            [1,1,1,2,2,2,2,2,2,2,2,1,1,1,1],
            [1,2,2,1,1,2,2,1,1,2,2,1,1,2,2],
            [1,2,2,1,1,2,2,2,2,1,1,2,2,1,1],
            [1,2,2,2,2,1,1,1,1,2,2,2,2,1,1],
            [1,2,2,2,2,1,1,2,2,1,1,1,1,2,2],
            [2,1,2,1,2,1,2,1,2,1,2,1,2,1,2],
            [2,1,2,1,2,1,2,2,1,2,1,2,1,2,1],
            [2,1,2,2,1,2,1,1,2,1,2,2,1,2,1],
            [2,1,2,2,1,2,1,2,1,2,1,1,2,1,2],
            [2,2,1,1,2,2,1,1,2,2,1,1,2,2,1],
            [2,2,1,1,2,2,1,2,1,1,2,2,1,1,2],
            [2,2,1,2,1,1,2,1,2,2,1,2,1,1,2],
            [2,2,1,2,1,1,2,2,1,1,2,1,2,2,1],
        ], dtype=int)
        return arr12 - 1
    if name == "L18(2^1 3^7)":
        part1 = np.array([
            [1,1,1,1,1,1,1],[1,1,2,2,2,2,2],[1,1,3,3,3,3,3],
            [1,2,1,1,2,2,3],[1,2,2,2,3,3,1],[1,2,3,3,1,1,2],
            [1,3,1,2,1,3,2],[1,3,2,3,2,1,3],[1,3,3,1,3,2,1],
            [2,1,1,3,3,2,2],[2,1,2,1,1,3,3],[2,1,3,2,2,1,1],
            [2,2,1,2,3,1,3],[2,2,2,3,1,2,1],[2,2,3,1,2,3,2],
            [2,3,1,3,2,3,1],[2,3,2,1,3,1,2],[2,3,3,2,1,2,3]
        ], dtype=int)
        col8 = np.array([[1],[2],[3],[3],[1],[2],[3],[1],[2],[1],[2],[3],[2],[3],[1],[2],[3],[1]], dtype=int)
        return np.hstack([(part1[:,0:1]-1), (part1[:,1:]-1), (col8-1)])
    if name == "L27(3^13)":
        arr27 = np.array([
            [1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,2,2,2,2,2,2,2,2,2],[1,1,1,1,3,3,3,3,3,3,3,3,3],
            [1,2,2,2,1,1,1,2,2,2,3,3,3],[1,2,2,2,2,2,2,3,3,3,1,1,1],[1,2,2,2,3,3,3,1,1,1,2,2,2],
            [1,3,3,3,1,1,1,3,3,3,2,2,2],[1,3,3,3,2,2,2,1,1,1,3,3,3],[1,3,3,3,3,3,3,2,2,2,1,1,1],
            [2,1,2,3,1,2,3,1,2,3,1,2,3],[2,1,2,3,2,3,1,2,3,1,2,3,1],[2,1,2,3,3,1,2,3,1,2,3,1,2],
            [2,2,3,1,1,2,3,2,3,1,3,1,2],[2,2,3,1,2,3,1,3,1,2,1,2,3],[2,2,3,1,3,1,2,1,2,3,2,3,1],
            [2,3,1,2,1,2,3,3,1,2,2,3,1],[2,3,1,2,2,3,1,1,2,3,3,1,2],[2,3,1,2,3,1,2,2,3,1,1,2,3]
        ], dtype=int)
        return arr27 - 1
    raise RuntimeError(f"OA '{name}' não disponível.")


def full_factorial_runs(levels_by_factor: list[int]) -> int:
    runs = 1
    for n in levels_by_factor:
        runs *= int(n)
    return runs

# ---------------------------------------------
# Upload de fatores
# ---------------------------------------------
upl = st.file_uploader(
    "**Carregar arquivo de fatores**",
    type=["xlsx"],
    key="fatores_upl",
    help="Selecione o arquivo Excel com a configuração dos fatores (aba 'Fatores')."
)

if upl:
    try:
        df_fatores = pd.read_excel(upl, sheet_name='Fatores')
        if 'Factor' not in df_fatores.columns:
            st.error("❌ Coluna 'Factor' não encontrada no arquivo.")
        else:
            st.success("✅ Arquivo carregado com sucesso!")
            st.dataframe(df_fatores, use_container_width=True, hide_index=True)  # <- sem índice

            st.subheader("🔍 Análise Automática dos Fatores")
            fatores = df_fatores['Factor'].astype(str).tolist()
            num_fatores = len(fatores)
            level_cols = [col for col in df_fatores.columns if col.startswith('Level')]
            niveis_por_fator, niveis_rotulos = [], []
            for _, row in df_fatores.iterrows():
                lvls = [row[col] for col in level_cols if pd.notna(row[col])]
                niveis_por_fator.append(len(lvls))
                niveis_rotulos.append([str(x) for x in lvls])
            niveis_unicos = list(set(niveis_por_fator))
            mesmo_numero_niveis = len(niveis_unicos) == 1
            dof_necessario = sum([n - 1 for n in niveis_por_fator])

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Número de Fatores", num_fatores)
            c2.metric("Níveis por Fator", f"{niveis_unicos[0]}" if mesmo_numero_niveis else f"misto: {min(niveis_por_fator)}–{max(niveis_por_fator)}")
            c3.metric("Graus de Liberdade Necessários", dof_necessario)
            c4.metric("Experimentos no Fatorial Completo", full_factorial_runs(niveis_por_fator))

            st.subheader("🎯 Matrizes Ortogonais Recomendadas")
            catalog = built_in_catalog()
            matrizes_candidatas = []
            for nome, specs in catalog.items():
                if specs['n'] - 1 < dof_necessario:
                    continue
                if mesmo_numero_niveis:
                    if niveis_unicos[0] == 2 and specs['cols2'] >= num_fatores:
                        matrizes_candidatas.append((nome, specs))
                    elif niveis_unicos[0] == 3 and specs['cols3'] >= num_fatores:
                        matrizes_candidatas.append((nome, specs))
                else:
                    f2 = sum(1 for n in niveis_por_fator if n == 2)
                    f3 = sum(1 for n in niveis_por_fator if n == 3)
                    if specs['cols2'] >= f2 and specs['cols3'] >= f3:
                        matrizes_candidatas.append((nome, specs))
            matrizes_candidatas.sort(key=lambda x: x[1]['n'])

            if matrizes_candidatas:
                total_full = full_factorial_runs(niveis_por_fator)
                linhas = []
                for nome, specs in matrizes_candidatas:
                    eficiencia = (1 - specs['n'] / total_full) * 100 if total_full > 0 else 0.0
                    linhas.append({
                        "Matriz": nome,
                        "Experimentos (n)": specs['n'],
                        "Colunas (2 níveis)": specs['cols2'],
                        "Colunas (3 níveis)": specs['cols3'],
                        "Economia de corridas (%)": f"{eficiencia:.1f}%"
                    })
                df_recomendacoes = pd.DataFrame(linhas)
                st.dataframe(df_recomendacoes, use_container_width=True, hide_index=True)  # <- sem índice
                st.caption("ℹ️ Economia de corridas em relação ao fatorial completo")

                st.subheader("🎛️ Seleção da Matriz Ortogonal")
                matriz_opcoes = [m[0] for m in matrizes_candidatas]
                matriz_selecionada = st.selectbox(
                    "Escolha a matriz para gerar o experimento:",
                    options=matriz_opcoes,
                    index=0,
                )

                if st.button("🔄 Gerar Matriz Experimental", type="primary"):
                    try:
                        matriz_oa = oa_from_name(matriz_selecionada)
                        if matriz_oa.shape[1] < num_fatores:
                            st.error("❌ A OA selecionada tem menos colunas do que o número de fatores.")
                        else:
                            matriz_oa = matriz_oa[:, :num_fatores]
                            df_codificada = pd.DataFrame(matriz_oa, columns=fatores)
                            df_niveis = pd.DataFrame(index=df_codificada.index)
                            for j, fator in enumerate(fatores):
                                rotulos = niveis_rotulos[j]
                                max_code = matriz_oa[:, j].max()
                                if max_code >= len(rotulos):
                                    st.warning(
                                        f"⚠️ Fator **{fator}** tem {len(rotulos)} níveis, mas a OA possui código até {int(max_code)}. Revise."
                                    )
                                df_niveis[fator] = [rotulos[c] if c < len(rotulos) else f"lvl{c+1}" for c in matriz_oa[:, j]]
                            df_niveis.insert(0, "Experimento", range(1, len(df_niveis) + 1))

                            st.session_state['matriz_selecionada'] = matriz_selecionada
                            st.session_state['matriz_oa'] = matriz_oa
                            st.session_state['df_fatores'] = df_fatores
                            st.session_state['df_experimentos_cod'] = df_codificada
                            st.session_state['df_experimentos'] = df_niveis
                            st.session_state['var_label'] = var_label
                            st.session_state['step'] = 'results'

                            st.success(f"✅ Matriz {matriz_selecionada} gerada com sucesso!")

                    except Exception as e:
                        st.error(f"❌ Erro ao gerar a matriz: {str(e)}")
            else:
                st.warning("⚠️ Nenhuma matriz ortogonal padrão adequada foi encontrada.")
    except Exception as e:
        st.error(f"❌ Erro ao processar o arquivo: {str(e)}")

# =========================
# Seção persistente de Resultados
# =========================
if st.session_state.get('df_experimentos') is not None:
    df_plan = st.session_state['df_experimentos']
    st.subheader("📊 Matriz Experimental Gerada")
    st.dataframe(df_plan, use_container_width=True, hide_index=True)

    # Download CSV
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False, sep=';').encode('utf-8')
    csv = convert_df_to_csv(df_plan)
    st.download_button(
        label="📥 Baixar Matriz Experimental (CSV)",
        data=csv,
        file_name=f"matriz_experimental_{matriz_selecionada}.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.subheader("📤 Upload de Resultados Experimentais (Réplicas/triplicatas)")
    var_label = st.session_state.get('var_label', 'Variável de Interesse')

    # Tipo de razão S/N
    sn_tipo = st.selectbox(
        "Tipo de razão Sinal-Ruído (S/N) (Taguchi)",
        options=["Maior é melhor", "Menor é melhor", "Nominal é melhor"],
        index=0,
    )
    nominal_target = None
    if sn_tipo == "Nominal é melhor":
        nominal_target = st.number_input("Alvo (m)", value=0.0, help="Para Nominal é melhor")

    # Fórmula em LaTeX
    sn_formulas = {
        "Maior é melhor":  r"S/N = -10 \log_{10} \left( \dfrac{1}{n} \sum_{i=1}^{n} \dfrac{1}{y_i^{2}} \right)",
        "Menor é melhor": r"S/N = -10 \log_{10} \left( \dfrac{1}{n} \sum_{i=1}^{n} y_i^{2} \right)",
        "Nominal é melhor":   r"S/N = 10 \log_{10} \left( \dfrac{m^{2}}{s^{2}} \right) \quad (m = \bar{y} \text{ se alvo não informado})"
    }
    st.markdown("**Fórmula da Razão Sinal-Ruído (S/N) selecionada:**")
    st.latex(sn_formulas[sn_tipo])

    result_upl = st.file_uploader(
        "**Carregar arquivo de resultados (réplicas do experimento)**",
        type=["xlsx", "csv"],
        key="resultados_upl",
    )

    if result_upl:
        try:
            if result_upl.name.endswith('.csv'):
                df_resultados = pd.read_csv(result_upl, sep=';')
            else:
                df_resultados = pd.read_excel(result_upl)

            # Normaliza/acha coluna Experimento
            exp_col = None
            for c in df_resultados.columns:
                if str(c).strip().lower() in {"experimento", "experiments", "exp", "run"}:
                    exp_col = c
                    break
            if exp_col is None:
                st.error("❌ O arquivo de resultados precisa ter a coluna 'Experimento'.")
            else:
                df_res = df_resultados.copy()
                df_res.rename(columns={exp_col: "Experimento"}, inplace=True)

                # Colunas numéricas (réplicas)
                num_cols = [c for c in df_res.columns if c != "Experimento" and pd.api.types.is_numeric_dtype(df_res[c])]
                if len(num_cols) == 0:
                    st.error("❌ Nenhuma coluna numérica de resposta encontrada.")
                else:
                    # Valida quantidade e índices
                    n_exp_plan = len(df_plan)
                    n_exp_res  = df_res['Experimento'].nunique()
                    dups = df_res['Experimento'][df_res['Experimento'].duplicated()].unique()
                    if len(dups) > 0:
                        st.error(f"❌ Há experimentos repetidos: {sorted(dups)}"); st.stop()
                    if n_exp_res != n_exp_plan:
                        st.error(f"❌ Resultados possuem {n_exp_res} experimentos; plano tem {n_exp_plan}."); st.stop()
                    esperados = set(range(1, n_exp_plan + 1))
                    presentes = set(df_res['Experimento'])
                    faltando  = sorted(esperados - presentes)
                    if faltando:
                        st.error(f"❌ Faltando experimentos: {faltando}"); st.stop()

                    st.success("✅ Número de experimentos confere com a matriz experimental!")
                    st.success("✅ Arquivo de resultados carregado com sucesso!")
                    st.dataframe(df_res, use_container_width=True, hide_index=True)
                    st.markdown("---")

                    # Merge e arrays
                    df_join = pd.merge(df_plan, df_res, on='Experimento', how='left')
                    rep_values = df_join[num_cols].to_numpy(dtype=float)
                    mean_y = np.nanmean(rep_values, axis=1)
                    std_y = np.nanstd(rep_values, axis=1, ddof=1)

                    # Funções S/N (locais)
                    def sn_larger_better(vals):
                        vals = np.asarray(vals, dtype=float)
                        return -10.0 * np.log10(np.mean(1.0/(vals**2)))
                    def sn_smaller_better(vals):
                        vals = np.asarray(vals, dtype=float)
                        return -10.0 * np.log10(np.mean(vals**2))
                    def sn_nominal_best(vals, target):
                        vals = np.asarray(vals, dtype=float)
                        if vals.size < 2:
                            return np.nan
                        return 10.0 * np.log10((target**2) / np.var(vals, ddof=1))
                    def compute_snr(vals, tipo, target=None):
                        if tipo == "Maior é melhor":
                            return sn_larger_better(vals)
                        if tipo == "Menor é melhor":
                            return sn_smaller_better(vals)
                        if tipo == "Nominal é melhor":
                            return sn_nominal_best(vals, target)
                        return np.nan

                    # Lista de réplicas por corrida
                    replicates = [rep_values[i, ~np.isnan(rep_values[i, :])] for i in range(rep_values.shape[0])]

                    # S/N das réplicas
                    sn_reps = [compute_snr(v, sn_tipo, nominal_target) for v in replicates]

                    # S/N da média
                    sn_mean = []
                    for m in mean_y:
                        if sn_tipo == "Nominal é melhor":
                            sn_mean.append(np.nan)
                        else:
                            sn_mean.append(compute_snr(np.array([m]), sn_tipo, nominal_target))

                    # Tabela Resultado por Ensaio
                    sn_table = pd.DataFrame({
                        "Experimento": df_plan["Experimento"],
                        f"Média de {var_label}": mean_y.astype(float),
                        f"S/N das réplicas ({var_label}) [dB]": sn_reps,
                    })

                    st.markdown("### 📊 Resultado por ensaio")
                    st.dataframe(sn_table, use_container_width=True, hide_index=True)

                    
                    # -------------------------------------------------
                    # Médias globais (Y e S/N das réplicas)
                    # -------------------------------------------------
                    grand_mean = np.nanmean(sn_reps) 
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric(
                            label=f"Média global de {var_label}",
                            value=f"{np.nanmean(mean_y):.3f}"
                        )
                    with c2:
                        st.metric(
                            label="Média global de S/N (réplicas)",
                            value=f"{grand_mean:.3f} dB"
                        )

                    # =============================================
                    # Efeitos médios dos fatores (S/N das réplicas) — Tabelas separadas
                    # =============================================
                    st.markdown("---")
                    st.subheader("📈 Efeitos principais na razão S/N (médias por nível)")
                    
                    if st.toggle("🔴🔴🔴 O que é o 'efeito'? (clique para ver) 🔴🔴🔴", value=False, key="show_efeito"):
                        st.markdown(
                            r"""
                            O **efeito** de um fator $k$ no nível $\ell$ é definido como o desvio da resposta média da razão Sinal-Ruído (S/N), obtida nesse nível específico, em relação à média global do experimento. Em outros termos, para cada **fator** denotado por $k$ e cada **nível** $\ell$ desse fator, 
                    define-se o efeito como a diferença entre a média de S/N nesse nível e a média global:
                            """
                        )
                        st.latex(r"\text{Efeito}(k,\ell)=\overline{\mathrm{S/N}}_{k,\ell}-\overline{\mathrm{S/N}}_{\text{global}}")
                        st.markdown(
                            r"""
                            **em que,**  
                            • $k \in \{1,\dots,K\}$ é o índice do fator (ex.: Temperatura, Pressão, ...), sendo $K$ o número total de fatores.  
                            
                            • $\ell \in \{1,\dots,L_k\}$ representa o índice do nível do fator $k$, sendo $L_k$ o número de níveis do respectivo fator. 
                            
                            • $\overline{\mathrm{S/N}}_{k,\ell}$:$\:\:\:$    média da razão Sinal-Ruído considerando apenas os ensaios em que o fator $k$ foi fixado no nível $\ell$.
                            
                            • $\overline{\mathrm{S/N}}_{\text{global}}$:$\:\:$    média da razão Sinal-Ruído considerando todos os ensaios do experimento.
                            """
                        )
                    
                    st.markdown("---")
                    
                    # Junta plano + S/N das réplicas
                    df_effects = df_plan.merge(
                        sn_table[["Experimento", f"S/N das réplicas ({var_label}) [dB]"]],
                        on="Experimento",
                        how="left"
                    )
                    sn_col = f"S/N das réplicas ({var_label}) [dB]"
                    
                    # 1) Tabelas de efeitos médios por fator (separadas)
                    factor_cols = [c for c in df_plan.columns if c != "Experimento"]
                    per_factor_tables = {}
                    for fac in factor_cols:
                        g = df_effects.groupby(fac)[sn_col].mean().sort_index()
                        fac_df = pd.DataFrame({"Nível": g.index, "S/N médio (dB)": g.values})
                        per_factor_tables[fac] = fac_df
                    

                    st.markdown("🔍 Tabelas por fator (S/N médio por nível)")
                    
                    factor_cols = [c for c in df_plan.columns if c != "Experimento"]
                    per_factor_tables = {}
                    
                    for fac in factor_cols:
                        # níveis como string, mas ordenados naturalmente (1,2,3,...)
                        lvls_in_plan = df_plan[fac].astype(str).unique().tolist()
                        try:
                            order_nat = sorted(lvls_in_plan, key=lambda s: int(s))
                        except Exception:
                            order_nat = sorted(lvls_in_plan)
                    
                        g = (
                            df_effects
                            .assign(**{fac: df_effects[fac].astype(str)})
                            .groupby(fac, as_index=True)[sn_col]
                            .mean()
                            .reindex(order_nat)
                        )
                        fac_df = (
                            pd.DataFrame({"Nível": g.index, "S/N médio (dB)": g.values})
                            .reset_index(drop=True)
                        )
                        # tipos, arredondamento e NOVA COLUNA: Efeito (dB)
                        fac_df["S/N médio (dB)"] = pd.to_numeric(fac_df["S/N médio (dB)"], errors="coerce")
                        fac_df["Efeito (dB)"] = (fac_df["S/N médio (dB)"] - float(grand_mean))
                        fac_df[["S/N médio (dB)", "Efeito (dB)"]] = fac_df[["S/N médio (dB)", "Efeito (dB)"]].round(3)
                    
                        per_factor_tables[fac] = fac_df
                    
                    # render (até 4 por linha)
                    COLS_PER_ROW = 4
                    for i in range(0, len(factor_cols), COLS_PER_ROW):
                        bloco = factor_cols[i:i + COLS_PER_ROW]
                        cols = st.columns(len(bloco))
                        for j, fac in enumerate(bloco):
                            with cols[j]:
                                st.markdown(f"**Fator: {fac}**")
                                st.dataframe(per_factor_tables[fac], use_container_width=True, hide_index=True)

                        # ============================
                        # 📥 Baixar tabelas por fator (CSV único)
                        # ============================
                        if per_factor_tables:
                            # Empilha todas as tabelas e inclui a coluna "Fator"
                            df_emp = pd.concat(
                                [df.assign(**{"Fator": fac}) for fac, df in per_factor_tables.items()],
                                ignore_index=True
                            )
                            st.download_button(
                                "📥 Baixar tabelas por fator (CSV)",
                                data=df_emp.to_csv(index=False).encode("utf-8"),
                                file_name=f"tabelas_SN_por_fator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                key="dl_tabelas_sn_csv",
                                help="Todas as tabelas empilhadas em um único CSV."
                            )
                        else:
                            st.info("Nenhuma tabela por fator disponível para download.")

                    st.markdown("---")
        
                    # =============================================
                    # 📈 Efeitos médios — gráficos (estilo Minitab)
                    # =============================================
                    
                    # 👉 calcular média global do S/N (das réplicas) ANTES dos gráficos
                    grand_mean = df_effects[sn_col].mean()

                    # =============================================
                    # 🖼️ Todos os fatores em uma única figura (níveis numerados)
                    # =============================================
                    st.subheader("📊 Efeitos médios — gráficos por fator")

                    # Até 4 gráficos por linha
                    MAX_COLS = 4
                    cols = MAX_COLS if len(factor_cols) >= MAX_COLS else (len(factor_cols) if len(factor_cols) > 0 else 1)
                    rows = math.ceil(len(factor_cols) / cols) if len(factor_cols) > 0 else 1
                    fig_all = make_subplots(rows=rows, cols=cols, subplot_titles=factor_cols)
                    
                    # ✅ Mesma escala Y em todos os subplots (inclui a média global)
                    all_y = []
                    for _fac in factor_cols:
                        _df = per_factor_tables[_fac].copy().reset_index(drop=True)
                        all_y.extend(_df["S/N médio (dB)"].astype(float).tolist())
                    if not math.isnan(grand_mean):
                        all_y.append(float(grand_mean))
                    
                    if len(all_y) > 0:
                        ymin, ymax = min(all_y), max(all_y)
                        pad = 0.1 * (ymax - ymin if ymax > ymin else (abs(ymax) if ymax != 0 else 1.0))
                        y_range = [ymin - pad, ymax + pad]
                    else:
                        y_range = None
                    
                    r, c = 1, 1
                    for fac in factor_cols:
                        fac_df = per_factor_tables[fac].copy().reset_index(drop=True)
                    
                        # X categórico: evita marcas 1.5, 2.5 etc.
                        num_levels = len(fac_df)
                        x_cat = [str(i) for i in range(1, num_levels + 1)]
                        y_vals = fac_df["S/N médio (dB)"].astype(float).tolist()
                    
                        # Curva do fator
                        fig_all.add_trace(
                            go.Scatter(
                                x=x_cat, y=y_vals,
                                mode="lines+markers",
                                name=f"{fac}",
                                showlegend=False,
                                hovertemplate="Nível=%{x}<br>S/N médio=%{y:.3f} dB<extra></extra>",
                            ),
                            row=r, col=c
                        )
                    
                        # Linha da média global em TODOS os subplots (legenda só no 1º)
                        if not math.isnan(grand_mean):
                            fig_all.add_trace(
                                go.Scatter(
                                    x=x_cat, y=[grand_mean]*len(x_cat),
                                    mode="lines",
                                    name="Média global",
                                    line=dict(dash="dash"),
                                    showlegend=(r == 1 and c == 1),
                                    hovertemplate="Média global=%{y:.3f} dB<extra></extra>",
                                ),
                                row=r, col=c
                            )
                    
                        # Eixos: Y rotulado só no 1º subplot; todos com o mesmo range
                        if r == 1 and c == 1:
                            fig_all.update_yaxes(title_text="S/N médio (dB)", range=y_range, row=r, col=c)
                        else:
                            fig_all.update_yaxes(title_text=None, range=y_range, row=r, col=c)
                    
                        # X categórico e título
                        fig_all.update_xaxes(
                            title_text="Níveis dos parâmetros",
                            type="category",          # força eixo categórico
                            tickmode="array",         # usa ticks explícitos
                            tickvals=x_cat,           # ["1","2","3",...]
                            ticktext=x_cat,           # rótulos iguais aos valores
                            categoryorder="category ascending",
                            row=r, col=c
                        )
                    
                        # avança até 'cols' colunas por linha (máx 4)
                        c += 1
                        if c > cols:
                            c = 1
                            r += 1
                    
                    fig_all.update_layout(height=280*rows, margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(fig_all, use_container_width=True)


                    # ============================
                    # Downloads (cores vs P&B) + anti-corte
                    # ============================
                    
                    
                    st.markdown("📄 Baixar figura")
                    color_mode = st.radio(
                        "Modo de cores para exportação:",
                        ["Cores (original)", "Preto e branco"],
                        index=0,
                        help="A visualização na tela permanece em cores. A opção afeta apenas os arquivos baixados."
                    )
                    
                    # Faz uma CÓPIA para exportação (preserva a que está na tela)
                    fig_exp = fig_all.to_dict()  # cópia “barata”
                    fig_exp = go.Figure(fig_exp)
                    
                    # Margens generosas e fundos brancos para evitar cortes e fundos cinza
                    rows = math.ceil(len(factor_cols) / cols) if len(factor_cols) > 0 else 1
                    export_width  = 1100                      # largura fixa para exportar
                    export_height = 320 * rows + 80           # mais alto que o mostrado (anti-corte)
                    
                    fig_exp.update_layout(
                        width=export_width,
                        height=export_height,
                        margin=dict(l=70, r=30, t=60, b=70),  # margens maiores -> evita corte à esquerda/embaixo
                        paper_bgcolor="white",
                        plot_bgcolor="white",
                        template="plotly_white",
                    )
                    
                    # Aplica estilo preto-e-branco se selecionado
                    if color_mode == "Preto e branco":
                        # Traços: pretos; diferenciar com traços/dashes diferentes
                        dash_cycle = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
                        t_idx = 0
                        for i, tr in enumerate(fig_exp.data):
                            if isinstance(tr, go.Scatter):
                                # Linha da média global: trace sem marcador e (se houver) legenda “Média global”
                                is_global_mean = (getattr(tr, "name", "") == "Média global") or (
                                    hasattr(tr, "hovertemplate") and "Média global" in str(tr.hovertemplate)
                                )
                                tr.update(
                                    line=dict(color="black", width=2, dash=("dot" if is_global_mean else dash_cycle[t_idx % len(dash_cycle)])),
                                    marker=dict(color="black", size=7),
                                )
                                if not is_global_mean:
                                    t_idx += 1
                    
                    # Render na tela segue como está; exportação usa kaleido
                    # Tenta exportar; se kaleido não estiver instalado, mostra instrução amigável
                    def _export_bytes(fmt: str):
                        try:
                            return fig_exp.to_image(format=fmt, scale=2, width=export_width, height=export_height)
                        except Exception as e:
                            st.warning(
                                "Para exportar imagens, é necessário o pacote **kaleido**.\n\n"
                                "Instale com:\n\n"
                                "`pip install -U kaleido`\n\n"
                                "ou\n\n"
                                "`conda install -c conda-forge python-kaleido -y`"
                            )
                            raise
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        try:
                            png_bytes = _export_bytes("png")
                            st.download_button("📥 PNG", data=png_bytes, file_name="efeitos_medios_todos_fatores.png", mime="image/png")
                        except Exception:
                            pass
                    with col2:
                        try:
                            svg_bytes = _export_bytes("svg")
                            st.download_button("📥 SVG (vetorial)", data=svg_bytes, file_name="efeitos_medios_todos_fatores.svg", mime="image/svg+xml")
                        except Exception:
                            pass
                    with col3:
                        try:
                            pdf_bytes = _export_bytes("pdf")
                            st.download_button("📥 PDF", data=pdf_bytes, file_name="efeitos_medios_todos_fatores.pdf", mime="application/pdf")
                        except Exception:
                            pass
                    with col4:
                        # HTML interativo sempre em CORES (original da tela)
                        html_bytes = pio.to_html(fig_all, include_plotlyjs="cdn", full_html=False).encode("utf-8")
                        st.download_button("📥 HTML (interativo)", data=html_bytes, file_name="efeitos_medios_todos_fatores.html", mime="text/html")


                    # =============================================
                    # 📐 Δ por fator — Tabela simples
                    # Requer: per_factor_tables (dict: fac -> DataFrame com colunas ["Nível","S/N médio (dB)"])
                    # =============================================
                    st.markdown("---")
                    st.subheader("📐 A regra Delta por fator")

                    if st.toggle("🔴🔴🔴 O que é o 'Delta'? (clique para ver) 🔴🔴🔴", value=False, key="show_delta"):
                        st.markdown(r"""
                        Em linha gerais, o valor de $\Delta$ fornece uma medida comparativa de influência de cada fator sobre a resposta do problema, sendo que fatores com maiores valores de $\Delta$ são considerados mais relevantes, pois produzem maior variação na razão sinal-ruído média entre seus níveis. Especificamente, para cada fator $ k $, o **Delta** $( \Delta_k )$ é dado pela **amplitude** entre a maior e a menor **S/N média** dos seus níveis:
                        """
                                   )

                        st.latex(r"\Delta_k = \max_{\ell} \, \overline{\mathrm{S/N}}_{k,\ell} - \min_{\ell} \, \overline{\mathrm{S/N}}_{k,\ell}")
                    
                        st.markdown(r"""
                    **Procedimento de cálculo (passos):**
                    1. Agrupe a $\mathrm{S/N}$ por **nível** do fator $k$.
                    2. Calcule a **$\mathrm{S/N}$ média** em cada nível.
                    3. Identifique **máximo** e **mínimo** dessas médias.  
                    4. Faça $\Delta = \textrm{máx} - \textrm{mín}$ (em dB).
                    
                    **Interpretação.** 
                    - Valor de $\Delta_k$  grande $\implies$ o fator $k$ **altera fortemente** a resposta (maior influência)  
                    - Valor de $ \Delta_k \approx 0 $ $\implies$ pouca ou nenhuma influência detectável via $\mathrm{S/N}$
                    
                    **Observações rápidas:**
                    - Válido para qualquer tipo de S/N (maior-melhor, menor-melhor, nominal-melhor).  
                    - Em **empates** de S/N média entre níveis, adote uma regra estável (p.ex., a **ordem natural** dos níveis).
                    - Ordena fatores por influência (***regra Delta***), porém **não** testa significância.
                    - Para **significância estatística**, use **ANOVA sobre S/N** em complemento à regra delta
                    
                    """)
                    
                    st.markdown("---")
                    st.markdown("🔍 Tabelas de cálculo da regra delta por fator")
                    
                    rows = []
                    for fac, fac_df in per_factor_tables.items():
                        if fac_df.empty or "S/N médio (dB)" not in fac_df.columns:
                            rows.append({"Fator": fac, "S/N médio máx. (dB)": float("nan"),
                                         "S/N médio mín. (dB)": float("nan"), "Δ (dB)": float("nan")})
                            continue
                        s = pd.to_numeric(fac_df["S/N médio (dB)"], errors="coerce")
                        vmax = float(s.max())
                        vmin = float(s.min())
                        rows.append({"Fator": fac,
                                     "S/N médio máx. (dB)": round(vmax, 3),
                                     "S/N médio mín. (dB)": round(vmin, 3),
                                     "Δ (dB)": round(vmax - vmin, 3)})
                    
                    delta_simple_df = (
                        pd.DataFrame(rows)
                        .sort_values("Δ (dB)", ascending=False, na_position="last")
                        .reset_index(drop=True)
                    )
                    delta_simple_df["Rank (Δ)"] = np.arange(1, len(delta_simple_df) + 1)
                    
                    st.dataframe(delta_simple_df, use_container_width=True, hide_index=True)
                    
                    # Download CSV
                    buf = io.StringIO()
                    delta_simple_df.to_csv(buf, index=False)
                    st.download_button(
                        "📥 Baixar delta por fator (CSV)",
                        data=buf.getvalue().encode("utf-8"),
                        file_name=f"delta_simples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="dl_delta_simples_csv",
                    )



                
                # Sumário com Melhor nível, Delta e ranking (usando S/N médio)
                summary_rows = []
                for fac, fac_df in per_factor_tables.items():
                    if fac_df.empty or fac_df["S/N médio (dB)"].isna().all():
                        summary_rows.append({"Fator": fac, "Melhor nível": "-", "Delta (dB)": float("nan")})
                        continue
                    vmax = fac_df["S/N médio (dB)"].max()
                    vmin = fac_df["S/N médio (dB)"].min()
                    best_lvl = fac_df.loc[fac_df["S/N médio (dB)"] == vmax, "Nível"].iloc[0]
                    delta = float(vmax - vmin)
                    summary_rows.append({"Fator": fac, "Melhor nível": best_lvl, "Delta (dB)": round(delta, 3)})
                
                summary_df = (
                    pd.DataFrame(summary_rows)
                    .sort_values("Delta (dB)", ascending=False, na_position="last")
                    .reset_index(drop=True)
                )
                summary_df["Rank (Delta)"] = np.arange(1, len(summary_df) + 1)
                
                st.markdown("🔍 Sumário (melhor nível, Delta e ranking)")
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

                
                # 3) Downloads
                colA, colB = st.columns(2)
                with colA:
                    csv_efx = pd.concat(
                        [df.assign(**{"Fator": fac}) for fac, df in per_factor_tables.items()],
                        ignore_index=True
                    )
                    st.download_button(
                        "📥 Baixar efeitos por nível (CSV)",
                        data=csv_efx.to_csv(index=False).encode("utf-8"),
                        file_name="efeitos_medios_por_nivel.csv",
                        mime="text/csv",
                    )
                with colB:
                    st.download_button(
                        "📥 Baixar sumário (CSV)",
                        data=summary_df.to_csv(index=False).encode("utf-8"),
                        file_name="sumario_melhor_nivel_delta.csv",
                        mime="text/csv",
                    )

            st.markdown("---")
            # ================================================================
            # 📊 Observado × Predito — em duas tabelas (Y) e (S/N)
            # ================================================================
            st.subheader("📊 Valor médio observado versus valor médio predito (por ensaio)")
            st.caption("As estimativas são calculadas usando a média das réplicas de cada ensaio e o modelo aditivo de efeitos principais. Os resíduos correspondem à diferença entre o valor médio observado e o valor médio predito.")

            if st.toggle("🔴🔴🔴 Como é calculado o valor predito? (clique para ver) 🔴🔴🔴", value=False, key="show_pred"):
                        st.markdown(r"""
                        O valor **predito** para cada ensaio é obtido pelo **modelo aditivo de efeitos principais**, cujo resultado é dado por:
                        """
                                   )
                        st.latex(r"\hat{Y}_\ell \;=\; \bar{Y} \;+\; \sum_{j=1}^{K} \text{Efeito}(j,\ell)")

                        st.markdown(
                            r"""
                            **em que,**  
                            •  $\hat{Y}_\ell$ é a resposta predita para a combinação de níveis $\ell$;
                            
                            • $\bar{Y}$ é a média global da resposta;
                            
                            • $\text{Efeito}(j,\ell)$ é o efeito principal do fator $j$ no nível selecionado;
                            
                            • $K$ é o número total de fatores.
                            """
                        )


            st.markdown("---") 
            # ---------- Preparos ----------
            n_factors = len(factor_cols)

            # Vetores por ensaio
            y_by_run = np.asarray(mean_y, dtype=float)                          # média das réplicas (Y observado)
            sn_by_run = df_effects[sn_col].astype(float).to_numpy()             # S/N observado (dB)

            Y_bar  = float(np.nanmean(y_by_run))
            SN_bar = float(np.nanmean(sn_by_run))

            # Médias por nível (Y)
            tmp_y = df_plan.copy()
            tmp_y["__mean_y__"] = y_by_run
            mean_y_level = {fac: tmp_y.groupby(df_plan[fac].astype(str))["__mean_y__"].mean().to_dict()
                            for fac in factor_cols}

            # Médias por nível (S/N) — usa tabelas por fator; fallback por ensaios
            mean_sn_level = {}
            for fac in factor_cols:
                d = {}
                fac_df = per_factor_tables.get(fac, pd.DataFrame())
                if not fac_df.empty and {"Nível","S/N médio (dB)"}.issubset(fac_df.columns):
                    d.update(dict(zip(fac_df["Nível"].astype(str), fac_df["S/N médio (dB)"].astype(float))))
                # garante todos os níveis existentes no plano
                for lvl in df_plan[fac].astype(str).unique():
                    if lvl not in d:
                        mask = (df_plan[fac].astype(str) == lvl)
                        d[lvl] = float(df_effects.loc[mask, sn_col].mean())
                mean_sn_level[fac] = d

            # ---------- Monta TABELA 1: Y ----------
            rows_y = []
            for i in range(len(df_plan)):
                lvl_dict = {fac: str(df_plan.loc[i, fac]) for fac in factor_cols}
                sum_levels_y = sum(float(mean_y_level[fac].get(lvl_dict[fac], np.nan)) for fac in factor_cols)
                y_pred = float(sum_levels_y - (n_factors - 1) * Y_bar)
                rows_y.append({
                    **lvl_dict,
                    "Y observado": y_by_run[i],
                    "Y predito": y_pred,
                    "Resíduo Y": y_by_run[i] - y_pred,
                })
            df_obs_pred_y = pd.DataFrame(rows_y)

            st.markdown("🔍 Tabela de predições ao problema (Y)")
            st.dataframe(
                df_obs_pred_y.round({"Y observado":3, "Y predito":3, "Resíduo Y":3}),
                use_container_width=True, hide_index=True
            )

            # Download Y
            import io
            from datetime import datetime
            buf_y = io.StringIO()
            df_obs_pred_y.to_csv(buf_y, index=False)
            st.download_button(
                "📥 Baixar tabela Y (CSV)",
                data=buf_y.getvalue().encode("utf-8"),
                file_name=f"observado_predito_residuo_Y_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="dl_obs_pred_Y"
            )

            st.divider()

            # ---------- Monta TABELA 2: S/N ----------
            rows_sn = []
            for i in range(len(df_plan)):
                lvl_dict = {fac: str(df_plan.loc[i, fac]) for fac in factor_cols}
                sum_levels_sn = sum(float(mean_sn_level[fac].get(lvl_dict[fac], np.nan)) for fac in factor_cols)
                sn_pred = float(sum_levels_sn - (n_factors - 1) * SN_bar)
                rows_sn.append({
                    **lvl_dict,
                    "S/N observado (dB)": sn_by_run[i],
                    "S/N predito (dB)": sn_pred,
                    "Resíduo S/N (dB)": sn_by_run[i] - sn_pred,
                })
            df_obs_pred_sn = pd.DataFrame(rows_sn)

            st.markdown("🔍 Tabela de predições em relação a razão Sinal-Ruído")
            st.dataframe(
                df_obs_pred_sn.round({"S/N observado (dB)":3, "S/N predito (dB)":3, "Resíduo S/N (dB)":3}),
                use_container_width=True, hide_index=True
            )

            # Download S/N
            buf_sn = io.StringIO()
            df_obs_pred_sn.to_csv(buf_sn, index=False)
            st.download_button(
                "📥 Baixar tabela S/N (CSV)",
                data=buf_sn.getvalue().encode("utf-8"),
                file_name=f"observado_predito_residuo_SN_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="dl_obs_pred_SN"
            )

            st.divider()


             # ========= MODO 1: tudo no corpo principal (sem colunas externas) =========
            st.subheader("🧮 Predição de valores para qualquer combinação de fatores")
            st.caption(
                "Use esta seção para estimar a resposta ou a razão S/N em qualquer combinação de fatores, "
                "mesmo que não esteja na matriz ortogonal."
            )

            # 🔸 Texto orientando a entrada do usuário
            st.markdown("**Selecione abaixo um nível para cada fator e o sistema calculará automaticamente a resposta predita.**")


            # ----------------- Seleção de níveis pelo usuário (vertical, corpo principal) -----------------
            user_levels = {}
            for fac in factor_cols:
                niveis = sorted(df_plan[fac].astype(str).unique())
                user_levels[fac] = st.selectbox(f"Nível para {fac}:", niveis, key=f"pred_{fac}")

            # ----------------- Cálculo das previsões -----------------
            # Y (usa média das réplicas por ensaio já calculada em mean_y)
            try:
                y_by_run = np.asarray(mean_y, dtype=float)
                Y_bar = float(np.nanmean(y_by_run))
                efeitos = []
                for fac in factor_cols:
                    nivel = str(user_levels[fac])
                    mask = (df_plan[fac].astype(str) == nivel).values
                    media_nivel = float(np.nanmean(y_by_run[mask])) if mask.any() else np.nan
                    efeitos.append(media_nivel - Y_bar)
                Y_hat = float(Y_bar + np.nansum(efeitos))
            except Exception as e:
                Y_hat = float("nan")
                st.warning(f"Não foi possível calcular a previsão de {var_label}: {e}")

            # S/N (usa tabelas por fator; se faltar nível, faz fallback direto dos ensaios)
            try:
                sn_bar = float(df_effects[sn_col].mean())
                efeitos_sn = []
                for fac in factor_cols:
                    nivel = str(user_levels[fac])
                    fac_df = per_factor_tables.get(fac, pd.DataFrame())
                    media_sn = np.nan
                    if not fac_df.empty and {"Nível","S/N médio (dB)"}.issubset(set(fac_df.columns)):
                        media_sn = fac_df.loc[fac_df["Nível"].astype(str) == nivel, "S/N médio (dB)"].mean()
                    if pd.isna(media_sn):  # fallback
                        mask = (df_plan[fac].astype(str) == nivel)
                        media_sn = float(df_effects.loc[mask, sn_col].mean())
                    efeitos_sn.append(media_sn - sn_bar)
                eta_hat = float(sn_bar + np.nansum(efeitos_sn))
            except Exception as e:
                eta_hat = float("nan")
                st.warning(f"Não foi possível calcular a previsão de S/N: {e}")

            st.divider()
            st.markdown("🔍 **Resultados das predições no ponto fornecido pelo usuário**") 

            # ----------------- Cards de resultados (mesmo estilo verde dos seus cards) -----------------
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    f"""
                    <div style="text-align:center; margin: 14px 0 8px;">
                      <div style="display:inline-block; padding:12px 22px; background:#ecfdf5;
                                  border-radius:10px; box-shadow:0 3px 12px rgba(0,0,0,0.12);">
                        <div style="font-size:14px; color:#065f46; font-weight:600; margin-bottom:4px;">
                          Previsão para {var_label}
                        </div>
                        <div style="font-size:26px; font-weight:700; color:#064e3b;">
                          {("n/d" if np.isnan(Y_hat) else f"{Y_hat:.3f}")}
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
                    <div style="text-align:center; margin: 14px 0 8px;">
                      <div style="display:inline-block; padding:12px 22px; background:#ecfdf5;
                                  border-radius:10px; box-shadow:0 3px 12px rgba(0,0,0,0.12);">
                        <div style="font-size:14px; color:#065f46; font-weight:600; margin-bottom:4px;">
                          Previsão para S/N (dB)
                        </div>
                        <div style="font-size:26px; font-weight:700; color:#064e3b;">
                          {("n/d" if np.isnan(eta_hat) else f"{eta_hat:.3f} dB")}
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )



             # =========================
            # 📥 Exportações de predição (dois botões lado a lado)
            # =========================

            # --- helper: previsão para uma combinação arbitrária de níveis (dict fac->nivel em str)
            def _predict_combo(level_dict):
                # Y
                y_by_run = np.asarray(mean_y, dtype=float)
                Y_bar = float(np.nanmean(y_by_run))
                efeitos_y = []
                for fac in factor_cols:
                    nivel = str(level_dict[fac])
                    mask = (df_plan[fac].astype(str) == nivel).values
                    media_nivel = float(np.nanmean(y_by_run[mask])) if mask.any() else np.nan
                    efeitos_y.append(media_nivel - Y_bar)
                y_pred = float(Y_bar + np.nansum(efeitos_y))

                # S/N
                sn_bar = float(df_effects[sn_col].mean())
                efeitos_sn = []
                for fac in factor_cols:
                    nivel = str(level_dict[fac])
                    fac_df = per_factor_tables.get(fac, pd.DataFrame())
                    media_sn = np.nan
                    if not fac_df.empty and {"Nível","S/N médio (dB)"}.issubset(fac_df.columns):
                        media_sn = fac_df.loc[fac_df["Nível"].astype(str) == nivel, "S/N médio (dB)"].mean()
                    if pd.isna(media_sn):
                        mask = (df_plan[fac].astype(str) == nivel)
                        media_sn = float(df_effects.loc[mask, sn_col].mean())
                    efeitos_sn.append(media_sn - sn_bar)
                eta_pred = float(sn_bar + np.nansum(efeitos_sn))

                return y_pred, eta_pred


            # ---------- (1) Ensaio atual ----------
            row_dict = {fac: user_levels[fac] for fac in factor_cols}
            y_pred_one, eta_pred_one = (Y_hat, eta_hat) if np.isfinite(Y_hat) and np.isfinite(eta_hat) else _predict_combo(row_dict)

            df_pred_one = pd.DataFrame([{
                **row_dict,
                f"Previsão {var_label}": (np.nan if not np.isfinite(y_pred_one) else round(y_pred_one, 6)),
                "Previsão S/N (dB)": (np.nan if not np.isfinite(eta_pred_one) else round(eta_pred_one, 6)),
            }])

            buf_one = io.StringIO()
            df_pred_one.to_csv(buf_one, index=False)
            fname_one = f"ensaio_predito_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            # ---------- (2) Matriz fatorial completa ----------
            levels_map = {fac: sorted(df_plan[fac].astype(str).unique()) for fac in factor_cols}
            rows = []
            for combo in product(*[levels_map[fac] for fac in factor_cols]):
                combo_dict = {fac: level for fac, level in zip(factor_cols, combo)}
                y_pred, eta_pred = _predict_combo(combo_dict)
                rows.append({
                    **combo_dict,
                    f"Previsão {var_label}": (np.nan if not np.isfinite(y_pred) else round(y_pred, 6)),
                    "Previsão S/N (dB)": (np.nan if not np.isfinite(eta_pred) else round(eta_pred, 6)),
                })

            df_full = pd.DataFrame(rows)
            buf_full = io.StringIO()
            df_full.to_csv(buf_full, index=False)
            fname_full = f"matriz_fatorial_predicoes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            # ---------- Exibir botões em duas colunas ----------
            #st.divider()
            st.markdown("<br><br>", unsafe_allow_html=True)
            col_b1, col_b2 = st.columns(2)

            with col_b1:
                st.download_button("📥 Baixar ensaio (predição atual)",
                                   buf_one.getvalue().encode("utf-8"),
                                   file_name=fname_one, mime="text/csv", key="dl_pred_one")

            with col_b2:
                st.download_button("📥 Baixar matriz fatorial completa (predições)",
                                   buf_full.getvalue().encode("utf-8"),
                                   file_name=fname_full, mime="text/csv", key="dl_pred_full")

            st.markdown("---")
            st.subheader("🎯 Análise do Ponto Ótimo")

            # --- Níveis ótimos por fator — Taguchi (S/N das réplicas) ---
            opt_levels = {}
            opt_rows = []
            selected_level_means = []

            # PRIMEIRO: Precisamos da média da resposta (Y) no nível ótimo para cada fator
            # Junta os dados para calcular médias de Y por nível
            df_effects_y = df_plan.merge(
                df_res[['Experimento'] + num_cols],  # Pega todas as colunas numéricas (réplicas)
                on='Experimento',
                how='left'
            )

            # Calcula média de Y para cada experimento (média das réplicas)
            df_effects_y['Media_Y'] = df_effects_y[num_cols].mean(axis=1)

            for fac in factor_cols:
                fac_df = per_factor_tables[fac]
                if fac_df.empty or fac_df["S/N médio (dB)"].isna().all():
                    opt_levels[fac] = {"Níveis ótimos": [], "S/N médio (dB)": float("nan")}
                    # NOVO: Calcula média de Y no nível ótimo (se disponível)
                    media_y_otimo = float("nan")
                else:
                    vmax = fac_df["S/N médio (dB)"].max()
                    best_levels = (
                        fac_df.loc[fac_df["S/N médio (dB)"] == vmax, "Nível"]
                        .astype(str)
                        .tolist()
                    )

                    opt_levels[fac] = {"Níveis ótimos": best_levels, "S/N médio (dB)": float(vmax)}
                    selected_level_means.append(float(vmax))

                    # NOVO: Calcula a média de Y no nível ótimo
                    # Pega o primeiro nível ótimo (se houver múltiplos, usa o primeiro)
                    nivel_otimo = best_levels[0] if best_levels else None

                    if nivel_otimo:
                        # Filtra os experimentos com este nível ótimo e calcula média de Y
                        mask = df_effects_y[fac] == nivel_otimo
                        media_y_otimo = df_effects_y.loc[mask, 'Media_Y'].mean()
                    else:
                        media_y_otimo = float("nan")

                # Adiciona ao dataframe de resultados
                opt_rows.append({
                    "Fator": fac,
                    "Nível(éis) ótimo(s)": " / ".join(best_levels) if best_levels else "-",
                    "S/N médio (dB)": float(vmax) if not fac_df.empty else float("nan"),
                    # NOVA COLUNA: Média no Nível Ótimo
                    f"Média de {var_label} no Nível Ótimo": media_y_otimo
                })

            st.markdown("🔍 **Níveis ótimos por fator**")
            opt_table = pd.DataFrame(opt_rows)

            # Formata a nova coluna para melhor visualização
            opt_table[f"Média de {var_label} no Nível Ótimo"] = opt_table[
                f"Média de {var_label} no Nível Ótimo"
            ].round(3)

            st.dataframe(opt_table, use_container_width=True, hide_index=True)

            st.download_button(
                "📥 Baixar níveis ótimos (CSV)",
                data=opt_table.to_csv(index=False).encode("utf-8"),
                file_name="ponto_otimo_taguchi.csv",
                mime="text/csv",
            )
  
            st.markdown("---") 
            st.subheader("🎯 Estimativa de valores no ponto ótimo")

            # ==============================
            # 🔹 Resumo do ponto ótimo (antes das caixas)
            # ==============================
            # Seleciona o(s) nível(is) ótimo(s) por fator a partir das tabelas por fator (S/N)
            opt_levels = {}
            selected_level_means = []  # S/N médio (dB) do nível ótimo por fator

            for fac in factor_cols:
                fac_df = per_factor_tables.get(fac, pd.DataFrame())
                if not fac_df.empty and {"Nível","S/N médio (dB)"}.issubset(fac_df.columns):
                    vmax = float(fac_df["S/N médio (dB)"].max())
                    best_levels = fac_df.loc[fac_df["S/N médio (dB)"] == vmax, "Nível"].astype(str).tolist()
                    # Pega o primeiro se houver empates (exibição)
                    opt_levels[fac] = best_levels[0] if best_levels else "-"
                    selected_level_means.append(vmax)
                else:
                    opt_levels[fac] = "-"
                    selected_level_means.append(np.nan)

            # Linha de título do ponto ótimo
            st.markdown("🔍 **Ponto ótimo**")

            # Render simples dos níveis ótimos em uma linha “chipada”
            chips_html = "<div style='display:flex; flex-wrap:wrap; gap:8px;'>"
            for fac in factor_cols:
                chips_html += f"""
                    <div style="padding:6px 12px; background:#ecfdf5;
                                border-radius:999px; font-size:13px; color:#064e3b;
                                box-shadow:0 2px 6px rgba(0,0,0,0.08);">
                        <span style="font-weight:600; color:#065f46;">{fac}:</span> {opt_levels[fac]}
                    </div>"""
            chips_html += "</div>"
            st.markdown(chips_html, unsafe_allow_html=True)

            st.divider()
            st.markdown("🔍 **Resultados das predições no ponto ótimo**")
            
            # ==============================
            # 🔹 Caixas (mantendo seu formato)
            # ==============================
            colY, colSN = st.columns(2)

            # --------- COLUNA ESQUERDA: Y ---------
            with colY:
                try:
                    # Preferência: usar a coluna "Média de {var_label} no Nível Ótimo" se já existir
                    col_media_otimo = f"Média de {var_label} no Nível Ótimo"
                    if 'opt_table' in locals() and col_media_otimo in opt_table.columns:
                        Y_best_means = opt_table[col_media_otimo].to_numpy(dtype=float)
                        k = len(Y_best_means)
                    else:
                        # Fallback: calcula média de Y no nível ótimo diretamente do plano
                        y_by_run = np.asarray(mean_y, dtype=float)
                        Y_best_means = []
                        for fac in factor_cols:
                            nivel = str(opt_levels[fac])
                            if nivel == "-":
                                Y_best_means.append(np.nan)
                            else:
                                mask = (df_plan[fac].astype(str) == nivel).values
                                Y_best_means.append(float(np.nanmean(y_by_run[mask])) if mask.any() else np.nan)
                        k = len(factor_cols)

                    Y_bar = float(np.nanmean(mean_y)) if 'mean_y' in locals() else float('nan')
                    if k > 0 and not np.isnan(Y_bar) and not np.isnan(np.array(Y_best_means)).any():
                        Y_hat_taguchi = float(np.sum(Y_best_means) - (k - 1) * Y_bar)
                    else:
                        Y_hat_taguchi = float("nan")
                except Exception as e:
                    st.warning(f"Não foi possível calcular a previsão de {var_label}: {e}")
                    Y_hat_taguchi = float("nan")

                # Card (verde)
                st.markdown(
                    f"""
                    <div style="text-align:center; margin: 14px 0 8px;">
                      <div style="display:inline-block; padding:12px 22px; background:#ecfdf5;
                                  border-radius:10px; box-shadow:0 3px 12px rgba(0,0,0,0.12);">
                        <div style="font-size:14px; color:#065f46; font-weight:600; margin-bottom:4px;">
                          Valor previsto (Taguchi) — {var_label}
                        </div>
                        <div style="font-size:26px; font-weight:700; color:#064e3b;">
                          {("n/d" if np.isnan(Y_hat_taguchi) else f"{Y_hat_taguchi:.3f}")}
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # --------- COLUNA DIREITA: S/N ---------
            with colSN:
                # grand_mean (S/N)
                try:
                    grand_mean = float(grand_mean)
                except Exception:
                    grand_mean = float(df_effects[sn_col].mean())

                # Garante selected_level_means (S/N por fator no nível ótimo)
                try:
                    needs_init = (not selected_level_means)
                except NameError:
                    needs_init = True
                if needs_init:
                    selected_level_means = []
                    for fac in factor_cols:
                        fac_df = per_factor_tables.get(fac, pd.DataFrame())
                        if fac_df.empty or fac_df["S/N médio (dB)"].isna().all():
                            selected_level_means.append(float("nan"))
                        else:
                            selected_level_means.append(float(fac_df["S/N médio (dB)"].max()))

                k = len(factor_cols)
                best_means_sn = np.array(selected_level_means, dtype=float)
                if k > 0 and not np.isnan(grand_mean) and not np.isnan(best_means_sn).any():
                    eta_hat_taguchi = float(best_means_sn.sum() - (k - 1) * grand_mean)
                else:
                    eta_hat_taguchi = float("nan")

                # Card (verde)
                st.markdown(
                    f"""
                    <div style="text-align:center; margin: 14px 0 8px;">
                      <div style="display:inline-block; padding:12px 22px; background:#ecfdf5;
                                  border-radius:10px; box-shadow:0 3px 12px rgba(0,0,0,0.12);">
                        <div style="font-size:14px; color:#065f46; font-weight:600; margin-bottom:4px;">
                          S/N previsto (Taguchi Aditivo) — {var_label}
                        </div>
                        <div style="font-size:26px; font-weight:700; color:#064e3b;">
                          {("n/d" if np.isnan(eta_hat_taguchi) else f"{eta_hat_taguchi:.3f} dB")}
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            #st.divider()
            st.markdown("<br><br>", unsafe_allow_html=True)

            # ==============================
            # 🔹 Baixar ponto ótimo com estimativas
            # ==============================

            # DataFrame com: níveis ótimos por fator + estimativas globais
            row = {fac: opt_levels[fac] for fac in factor_cols}
            row.update({
                f"Previsão {var_label}": (np.nan if np.isnan(Y_hat_taguchi) else round(Y_hat_taguchi, 6)),
                "Previsão S/N (dB)": (np.nan if np.isnan(eta_hat_taguchi) else round(eta_hat_taguchi, 6)),
            })

            df_opt_export = pd.DataFrame([row])
            buf_opt = io.StringIO()
            df_opt_export.to_csv(buf_opt, index=False)
            fname_opt = f"ponto_otimo_estimada_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            st.download_button(
                "📥 Baixar ponto ótimo com estimativas",
                data=buf_opt.getvalue().encode("utf-8"),
                file_name=fname_opt,
                mime="text/csv",
                key="dl_opt_estimates"
            )


            st.markdown("---")

            # ================================================================
            # 🧪 Ensaio de confirmação
            # ================================================================
            st.subheader("🧪 Ensaio de confirmação")
            
            st.caption(
                "Use esta seção para comparar os resultados de um ensaio de confirmação "
                "com os valores preditos pelo modelo aditivo de efeitos principais."
            )
            

            
            # ----------------- Passo 1: escolher ponto ótimo ou outra combinação -----------------
            st.markdown("**1️⃣ Escolha o ponto de análise do ensaio de confirmação**")
            
            # valor padrão anterior do modo (para detectar mudança)
            if "modo_conf_prev" not in st.session_state:
                st.session_state["modo_conf_prev"] = "Ponto ótimo (recomendado)"
            
            # garante valor inicial para n_reps_confirm
            if "n_reps_confirm" not in st.session_state:
                st.session_state["n_reps_confirm"] = 1
            
            modo_conf = st.radio(
                "Selecione a combinação de níveis a ser utilizada:",
                ("Ponto ótimo (recomendado)", "Outra combinação de níveis"),
                index=0,
                key="modo_conf"
            )
            
            # 🔄 sempre que o modo mudar (para qualquer um dos dois), resetar repetições e limpar campos
            if st.session_state["modo_conf_prev"] != modo_conf:
                # volta para 1 repetição
                st.session_state["n_reps_confirm"] = 1
            
                # limpa valores digitados anteriormente
                for k in list(st.session_state.keys()):
                    if k.startswith("y_conf_") or k.startswith("sn_conf_"):
                        st.session_state.pop(k)
            
                # atualiza modo anterior
                st.session_state["modo_conf_prev"] = modo_conf


            
            conf_levels = {}
            
            if modo_conf == "Ponto ótimo (recomendado)":
                if "opt_levels" in locals() and opt_levels:
                    st.markdown("Usando os níveis ótimos encontrados na análise anterior:")
                    lista_niveis = []
                    for fac in factor_cols:
                        val = opt_levels.get(fac, None)
            
                        # Caso 1: opt_levels[fac] seja um dicionário com "Níveis ótimos"
                        if isinstance(val, dict):
                            niveis_otimos = val.get("Níveis ótimos", [])
                            if niveis_otimos:
                                nivel_esc = str(niveis_otimos[0])
                            else:
                                nivel_esc = str(sorted(df_plan[fac].astype(str).unique())[0])
            
                        # Caso 2: opt_levels[fac] seja uma string (nível único)
                        elif isinstance(val, str) and val != "-":
                            nivel_esc = val
            
                        # Caso 3: qualquer outra coisa → fallback para primeiro nível disponível
                        else:
                            nivel_esc = str(sorted(df_plan[fac].astype(str).unique())[0])
            
                        conf_levels[fac] = nivel_esc
                        lista_niveis.append(f"- **{fac}**: nível `{nivel_esc}`")
            
                    st.markdown("\n".join(lista_niveis))
                else:
                    st.warning(
                        "Níveis ótimos não encontrados no sistema. "
                        "Selecione manualmente os níveis na opção 'Outra combinação de níveis'."
                    )
                    modo_conf = "Outra combinação de níveis"

            
            if modo_conf == "Outra combinação de níveis":
                st.markdown("Selecione manualmente os níveis utilizados no ensaio de confirmação:")
                for fac in factor_cols:
                    niveis = sorted(df_plan[fac].astype(str).unique())
                    conf_levels[fac] = st.selectbox(
                        f"Nível para {fac} no ensaio de confirmação:",
                        niveis,
                        key=f"conf_{fac}"
                    )
            
                        # ----------------- Passo 2: entrada dos resultados experimentais (várias repetições) -----------------
            st.markdown("**2️⃣ Carregue os resultados do ensaio de confirmação**")

            st.markdown(
                "Faça o upload de uma **matriz de repetições** do ensaio de confirmação. "
                "O arquivo pode ser `.xlsx` ou `.csv`. Todas as colunas numéricas serão "
                "usadas como valores reais de "
                f"**{var_label}** nas repetições (todas as linhas)."
            )

            # Vetor com os valores do ensaio de confirmação (todas as repetições)
            y_conf_vals = np.array([], dtype=float)

            conf_upl = st.file_uploader(
                "📤 Carregar matriz de repetições do ensaio de confirmação",
                type=["xlsx", "csv"],
                key="conf_upl",
            )

            if conf_upl is not None:
                try:
                    # Leitura do arquivo (similar ao upload de resultados principal)
                    if conf_upl.name.endswith(".csv"):
                        df_conf = pd.read_csv(conf_upl, sep=";")
                    else:
                        df_conf = pd.read_excel(conf_upl)

                    # Seleciona apenas colunas numéricas (valores reais do experimento)
                    num_cols = [
                        c for c in df_conf.columns
                        if pd.api.types.is_numeric_dtype(df_conf[c])
                    ]

                    if len(num_cols) == 0:
                        st.error("❌ Nenhuma coluna numérica encontrada no arquivo de confirmação.")
                    else:
                        vals = df_conf[num_cols].to_numpy(dtype=float).ravel()
                        vals = vals[~np.isnan(vals)]

                        if vals.size == 0:
                            st.error("❌ Não há valores numéricos válidos na matriz de confirmação.")
                        else:
                            y_conf_vals = vals
                            st.success(
                                f"✅ {len(y_conf_vals)} valores de {var_label} "
                                "carregados para o ensaio de confirmação."
                            )
                            st.markdown("**Valores utilizados no ensaio de confirmação:**")
                            st.dataframe(
                                pd.DataFrame({var_label: y_conf_vals}),
                                use_container_width=True,
                                hide_index=True
                            )

                            # Mostra o tipo de S/N que será usado no cálculo final
                            st.info(
                                f"A razão S/N do ensaio de confirmação será calculada com o mesmo tipo "
                                f"selecionado na análise principal: **{sn_tipo}**."
                            )

                except Exception as e:
                    st.error(f"❌ Erro ao processar o arquivo de confirmação: {e}")
                    y_conf_vals = np.array([], dtype=float)

            # ----------------- Passo 3: cálculo da média observada e comparação com o modelo -----------------
            st.markdown("**3️⃣ Comparação entre médias observadas e valores preditos**")

            if y_conf_vals.size == 0:
                st.info(
                    "⏳ Para calcular as médias observadas e comparar com o modelo, "
                    "primeiro carregue a matriz de resultados do ensaio de confirmação no **Passo 2**."
                )
            else:
                # Médias das repetições informadas
                y_conf_mean = float(np.nanmean(y_conf_vals))

                # S/N observado no ensaio de confirmação
                try:
                    sn_conf_mean = float(compute_snr(y_conf_vals, sn_tipo, nominal_target))
                except Exception:
                    sn_conf_mean = float("nan")

                # Predição Y para a combinação de confirmação
                try:
                    y_by_run = np.asarray(mean_y, dtype=float)
                    Y_bar = float(np.nanmean(y_by_run))
                    efeitos_conf = []
                    for fac in factor_cols:
                        nivel = str(conf_levels[fac])
                        mask = (df_plan[fac].astype(str) == nivel).values
                        media_nivel = float(np.nanmean(y_by_run[mask])) if mask.any() else np.nan
                        efeitos_conf.append(media_nivel - Y_bar)
                    Y_hat_conf = float(Y_bar + np.nansum(efeitos_conf))
                except Exception as e:
                    Y_hat_conf = float("nan")
                    st.warning(
                        f"Não foi possível calcular a previsão de {var_label} "
                        f"para o ensaio de confirmação: {e}"
                    )

                # Predição S/N para a combinação de confirmação (mesmo modelo de efeitos principais)
                try:
                    sn_bar = float(df_effects[sn_col].mean())
                    efeitos_sn_conf = []
                    for fac in factor_cols:
                        nivel = str(conf_levels[fac])
                        fac_df = per_factor_tables.get(fac, pd.DataFrame())
                        media_sn = np.nan
                        if not fac_df.empty and {"Nível", "S/N médio (dB)"}.issubset(set(fac_df.columns)):
                            media_sn = fac_df.loc[
                                fac_df["Nível"].astype(str) == nivel,
                                "S/N médio (dB)"
                            ].mean()
                        if pd.isna(media_sn):  # fallback pelos ensaios
                            mask = (df_plan[fac].astype(str) == nivel)
                            media_sn = float(df_effects.loc[mask, sn_col].mean())
                        efeitos_sn_conf.append(media_sn - sn_bar)
                    eta_hat_conf = float(sn_bar + np.nansum(efeitos_sn_conf))
                except Exception as e:
                    eta_hat_conf = float("nan")
                    st.warning(
                        f"Não foi possível calcular a previsão de S/N para o ensaio de confirmação: {e}"
                    )

                # Erros com base nas MÉDIAS observadas
                err_y = abs(y_conf_mean - Y_hat_conf) if not np.isnan(Y_hat_conf) else float("nan")
                err_rel_y = (
                    100.0 * err_y / abs(Y_hat_conf)
                    if (not np.isnan(err_y) and Y_hat_conf not in [0.0, np.nan])
                    else float("nan")
                )
                err_sn = abs(sn_conf_mean - eta_hat_conf) if not np.isnan(eta_hat_conf) else float("nan")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(
                        f"""
                        <div style="text-align:center; margin: 14px 0 8px;">
                          <div style="display:inline-block; padding:16px 26px; background:#eff6ff;
                                      border-radius:12px; box-shadow:0 3px 12px rgba(0,0,0,0.14);">
                            <div style="font-size:17px; color:#1d4ed8; font-weight:700; margin-bottom:6px;">
                              {var_label}: Média observada × Predito
                            </div>
                            <div style="font-size:15px; color:#1f2937; margin-bottom:6px; line-height:1.35;">
                              Média observada: <strong style="font-size:17px;">{("n/d" if np.isnan(y_conf_mean) else f"{y_conf_mean:.4f}")}</strong><br/>
                              Predito: <strong style="font-size:17px;">{("n/d" if np.isnan(Y_hat_conf) else f"{Y_hat_conf:.4f}")}</strong>
                            </div>
                            <div style="font-size:15px; color:#374151; line-height:1.35;">
                              Erro absoluto: <strong style="font-size:17px;">{("n/d" if np.isnan(err_y) else f"{err_y:.4f}")}</strong><br/>
                              Erro relativo: <strong style="font-size:17px;">{("n/d" if np.isnan(err_rel_y) else f"{err_rel_y:.2f}%")}</strong>
                            </div>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


                    with col2:
                        st.markdown(
                            f"""
                            <div style="text-align:center; margin: 14px 0 8px;">
                              <div style="display:inline-block; padding:16px 26px; background:#eff6ff;
                                          border-radius:12px; box-shadow:0 3px 12px rgba(0,0,0,0.14);">
                                <div style="font-size:17px; color:#1d4ed8; font-weight:700; margin-bottom:6px;">
                                  S/N (dB) observado vs Predito
                                </div>
                                <div style="font-size:15px; color:#1f2937; margin-bottom:6px; line-height:1.35;">
                                  S/N observado: <strong style="font-size:17px;">{("n/d" if np.isnan(sn_conf_mean) else f"{sn_conf_mean:.4f} dB")}</strong><br/>
                                  Predito: <strong style="font-size:17px;">{("n/d" if np.isnan(eta_hat_conf) else f"{eta_hat_conf:.4f} dB")}</strong>
                                </div>
                                <div style="font-size:15px; color:#374151; line-height:1.35;">
                                  Erro absoluto: <strong style="font-size:17px;">{("n/d" if np.isnan(err_sn) else f"{err_sn:.4f} dB")}</strong>
                                </div>
                              </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )



                    # ================================================================
                    # 📊 ANOVA sobre a razão S/N (opcional)
                    # ================================================================
                    st.markdown("---")
                    st.subheader("📊 ANOVA sobre a razão S/N (opcional)")
        
                    st.caption(
                        "Esta ANOVA é baseada na razão S/N por ensaio, usando apenas efeitos principais. "
                        "Ela decompõe a variação total de S/N em parcelas atribuídas a cada fator e ao erro."
                    )
        
                    if st.toggle("🔴🔴🔴 O que é esta ANOVA? (clique para ver) 🔴🔴🔴", value=False, key="show_anova_help"):
                        st.markdown(r"""
                        A ANOVA (Análise de Variância) aqui considera a **razão S/N de cada ensaio** como resposta
                        e decompõe a soma de quadrados total em:
        
                        - **Soma de Quadrados do Fator** ($SQ\_k$): quanto cada fator contribui para a variação de S/N;  
                        - **Soma de Quadrados de Erro**: variação não explicada pelos efeitos principais;  
                        - **Soma de Quadrados Total**: variação total da S/N em torno da média global.
        
                        Como o planejamento é ortogonal, a contribuição de cada fator é calculada por:
                        """)
                        st.latex(r"""
                        SS_k \;=\; \sum_{\ell} n_{k,\ell}\,\bigl(\overline{\mathrm{S/N}}_{k,\ell}
                        - \overline{\mathrm{S/N}}_{\text{global}}\bigr)^2
                        """)
                        st.markdown(r"""
                        em que $\overline{\mathrm{S/N}}_{k,\ell}$ é a média de S/N no nível $\ell$ do fator $k$
                        e $n_{k,\ell}$ é o número de ensaios nesse nível.
        
                        Quando há graus de liberdade de erro disponíveis, obtêm-se:
                        - $QM_k = SS_k / gl_k$  
                        - $QM_{erro} = SS_{erro} / gl_{erro}$  
                        - $F = QM_k / QM_{erro}$ (e opcionalmente um p-valor, se o pacote SciPy estiver disponível).
                        """)
        
                                        # Botão para ativar/rodar a ANOVA
                    if st.button("📊 Calcular ANOVA (S/N)", key="btn_anova_sn"):

                        # Vetor de S/N por ensaio
                        y_sn = df_effects[sn_col].to_numpy(dtype=float)
                        N = len(y_sn)
                        if N <= 1:
                            st.error("❌ Número insuficiente de ensaios para calcular ANOVA.")
                        else:
                            grand_mean_sn = float(np.nanmean(y_sn))
                            ss_total = float(np.nansum((y_sn - grand_mean_sn) ** 2))
                            df_total = N - 1

                            # Soma de quadrados por fator
                            factor_entries = []
                            ss_factors_sum = 0.0
                            df_factors_sum = 0

                            for fac in factor_cols:
                                # Agrupa S/N por nível (como string) do fator
                                g = df_effects.groupby(df_effects[fac].astype(str))[sn_col]
                                means = g.mean()
                                counts = g.size()

                                # SS do fator k: sum n_{k,ℓ} (mean_{k,ℓ} - grand_mean)^2
                                ss_fac = float(np.nansum(counts * (means - grand_mean_sn) ** 2))
                                df_fac = len(means) - 1

                                ss_factors_sum += ss_fac
                                df_factors_sum += df_fac

                                factor_entries.append({
                                    "Fonte": fac,
                                    "gl": df_fac,
                                    "SQ": ss_fac,
                                })

                            # Erro "bruto" (antes de pooling)
                            ss_error_raw = ss_total - ss_factors_sum
                            if ss_error_raw < 0 and abs(ss_error_raw) < 1e-10:
                                ss_error_raw = 0.0  # corrige pequeno negativo numérico
                            df_error_raw = df_total - df_factors_sum

                            # Contribuição original (%) de cada fator
                            for ent in factor_entries:
                                if ss_total > 0:
                                    ent["Contrib_orig"] = 100.0 * ent["SQ"] / ss_total
                                else:
                                    ent["Contrib_orig"] = np.nan

                            used_pooling = False
                            pooled_names = []
                            kept_entries = factor_entries.copy()

                            ss_error = ss_error_raw
                            df_error = df_error_raw

                            # ================================
                            # Pooling automático se não houver GL de erro
                            # ================================
                            if df_error_raw <= 0:
                                used_pooling = True

                                # Ordena fatores pela contribuição crescente
                                sorted_entries = sorted(
                                    factor_entries,
                                    key=lambda e: (np.inf if np.isnan(e["Contrib_orig"]) else e["Contrib_orig"])
                                )

                                # Candidatos naturais: contribuição < 5%
                                candidates = [
                                    e for e in sorted_entries
                                    if (not np.isnan(e["Contrib_orig"])) and (e["Contrib_orig"] < 5.0)
                                ]

                                # Se ninguém tiver < 5%, pega o menor fator (desde que haja mais de 1 fator)
                                if not candidates and len(sorted_entries) > 1:
                                    candidates = [sorted_entries[0]]

                                # Garante que NÃO vamos poolar todos os fatores
                                if len(candidates) >= len(sorted_entries):
                                    candidates = candidates[:-1]

                                ss_pool = 0.0
                                df_pool = 0
                                pooled_names = [ent["Fonte"] for ent in candidates]

                                for ent in candidates:
                                    ss_pool += ent["SQ"]
                                    df_pool += ent["gl"]

                                ss_error = max(0.0, ss_error_raw) + ss_pool
                                df_error = max(0, df_error_raw) + df_pool

                                kept_entries = [ent for ent in factor_entries if ent["Fonte"] not in pooled_names]

                                # Se ainda assim não conseguimos GL de erro, volta para modo "sem erro"
                                if df_error <= 0 or len(kept_entries) == 0:
                                    used_pooling = False
                                    pooled_names = []
                                    kept_entries = factor_entries
                                    ss_error = ss_error_raw
                                    df_error = df_error_raw

                            rows_anova = []

                            # ================================
                            # Caso NÃO haja GL de erro nem com pooling
                            # ================================
                            if df_error <= 0:
                                st.warning(
                                    "Os graus de liberdade dos fatores esgotam (ou superam) os graus de liberdade totais, "
                                    "e mesmo com pooling não há gl de erro suficientes para calcular F e p-valores. "
                                    "A tabela abaixo mostra apenas SQ, GL e contribuição de cada fator e o total."
                                )

                                for ent in factor_entries:
                                    rows_anova.append({
                                        "Fonte": ent["Fonte"],
                                        "GL": ent["gl"],
                                        "SQ": ent["SQ"],
                                        "QM": np.nan,
                                        "F": np.nan,
                                        "p-valor": np.nan,
                                        "Contribuição (%)": ent["Contrib_orig"],
                                        "Significativo (5%)": "n/d",
                                    })

                                rows_anova.append({
                                    "Fonte": "Erro",
                                    "GL": 0,
                                    "SQ": ss_error_raw,
                                    "QM": np.nan,
                                    "F": np.nan,
                                    "p-valor": np.nan,
                                    "Contribuição (%)": np.nan,
                                    "Significativo (5%)": "n/d",
                                })

                                rows_anova.append({
                                    "Fonte": "Total",
                                    "GL": df_total,
                                    "SQ": ss_total,
                                    "QM": np.nan,
                                    "F": np.nan,
                                    "p-valor": np.nan,
                                    "Contribuição (%)": 100.0 if ss_total > 0 else np.nan,
                                    "Significativo (5%)": "n/d",
                                })

                                anova_df = pd.DataFrame(rows_anova)

                                for col in ["SQ", "QM", "F", "p-valor", "Contribuição (%)"]:
                                    if col in anova_df.columns:
                                        anova_df[col] = pd.to_numeric(anova_df[col], errors="coerce").round(4)

                                st.markdown("🔍 **Tabela ANOVA (razão S/N como resposta)**")
                                st.dataframe(anova_df, use_container_width=True, hide_index=True)



    
                            # ================================
                            # Caso haja GL de erro (normal ou via pooling)
                            # ================================
                            else:
                                ms_error = ss_error / df_error if df_error > 0 else np.nan

                                # Contribuição final: fatores mantidos + erro
                                def contrib_final_ss(ss_part):
                                    return 100.0 * ss_part / ss_total if ss_total > 0 else np.nan

                                # Linhas dos fatores mantidos
                                for ent in kept_entries:
                                    gl_k = ent["gl"]
                                    ss_k = ent["SQ"]
                                    ms_k = ss_k / gl_k if gl_k > 0 else np.nan
                                    F_k = ms_k / ms_error if (gl_k > 0 and ms_error > 0) else np.nan

                                    if HAS_SCIPY and f_dist is not None and gl_k > 0 and df_error > 0 and not np.isnan(F_k):
                                        try:
                                            p_k = float(f_dist.sf(F_k, gl_k, df_error))
                                        except Exception:
                                            p_k = np.nan
                                    else:
                                        p_k = np.nan

                                    if not np.isnan(p_k):
                                        signif = "Sim (p < 0,05)" if p_k < 0.05 else "Não"
                                    else:
                                        signif = "n/d"

                                    rows_anova.append({
                                        "Fonte": ent["Fonte"],
                                        "GL": gl_k,
                                        "SQ": ss_k,
                                        "QM": ms_k,
                                        "F": F_k,
                                        "p-valor": p_k,
                                        "Contribuição (%)": contrib_final_ss(ss_k),
                                        "Significativo (5%)": signif,
                                    })

                                # Linha de erro (já incluindo pooling, se houve)
                                rows_anova.append({
                                    "Fonte": "Erro" + (" (com pooling)" if used_pooling else ""),
                                    "GL": df_error,
                                    "SQ": ss_error,
                                    "QM": ms_error,
                                    "F": np.nan,
                                    "p-valor": np.nan,
                                    "Contribuição (%)": contrib_final_ss(ss_error),
                                    "Significativo (5%)": "n/d",
                                })

                                # Linha total
                                rows_anova.append({
                                    "Fonte": "Total",
                                    "GL": df_total,
                                    "SQ": ss_total,
                                    "QM": np.nan,
                                    "F": np.nan,
                                    "p-valor": np.nan,
                                    "Contribuição (%)": 100.0 if ss_total > 0 else np.nan,
                                    "Significativo (5%)": "n/d",
                                })

                                anova_df = pd.DataFrame(rows_anova)

                                # Arredonda resultados numéricos
                                for col in ["SQ", "QM", "F", "p-valor", "Contribuição (%)"]:
                                    if col in anova_df.columns:
                                        anova_df[col] = pd.to_numeric(anova_df[col], errors="coerce").round(4)

                                st.markdown("🔍 **Tabela ANOVA (razão S/N como resposta)**")
                                st.dataframe(anova_df, use_container_width=True, hide_index=True)

                                # Comentário sobre pooling
                                if used_pooling and pooled_names:
                                    pooled_str = ", ".join(pooled_names)
                                    st.info(
                                        f"🔁 **Pooling automático ativado**: "
                                        f"{len(pooled_names)} fator(es) com baixa contribuição foram agrupados no erro: "
                                        f"**{pooled_str}**."
                                    )

                                    # Tabela auxiliar só com as contribuições originais dos fatores poolados
                                    pooled_rows = []
                                    for ent in factor_entries:
                                        if ent["Fonte"] in pooled_names:
                                            pooled_rows.append({
                                                "Fator poolado": ent["Fonte"],
                                                "SQ (original)": ent["SQ"],
                                                "Contribuição original (%)": ent["Contrib_orig"],
                                            })
                                    if pooled_rows:
                                        pooled_df = pd.DataFrame(pooled_rows)
                                        for col in ["SQ (original)", "Contribuição original (%)"]:
                                            pooled_df[col] = pd.to_numeric(pooled_df[col], errors="coerce").round(4)
                                        st.markdown("📌 **Fatores agrupados no erro (pooling)**")
                                        st.dataframe(pooled_df, use_container_width=True, hide_index=True)

                                # Resumo textual de significância e contribuição
                                try:
                                    sig_mask = (
                                        (anova_df["Fonte"].isin(kept_entries_df := pd.DataFrame(kept_entries)["Fonte"])) &
                                        anova_df["p-valor"].notna() &
                                        (anova_df["p-valor"] < 0.05)
                                    )
                                except Exception:
                                    sig_mask = pd.Series([False] * len(anova_df))

                                fatores_signif = anova_df.loc[
                                    (anova_df["Fonte"] != "Erro") &
                                    (anova_df["Fonte"] != "Total") &
                                    (anova_df["p-valor"].notna()) &
                                    (anova_df["p-valor"] < 0.05),
                                    ["Fonte", "Contribuição (%)", "p-valor"]
                                ]

                                if not fatores_signif.empty:
                                    st.markdown("✅ **Fatores estatisticamente significativos (α = 5%)**:")
                                    for _, row in fatores_signif.iterrows():
                                        st.markdown(
                                            f"- **{row['Fonte']}** → contribuição ≈ {row['Contribuição (%)']:.2f}% "
                                            f"(p ≈ {row['p-valor']:.4f})"
                                        )
                                else:
                                    st.markdown(
                                        "ℹ️ **Nenhum fator foi identificado como estatisticamente significativo "
                                        "(p < 0,05) com base nesta ANOVA.**"
                                    )

                                # Destaque dos fatores com maior contribuição (mesmo que não sejam significativos)
                                fatores_ord = anova_df[
                                    (anova_df["Fonte"] != "Erro") & (anova_df["Fonte"] != "Total")
                                ].sort_values("Contribuição (%)", ascending=False)

                                if not fatores_ord.empty:
                                    top_list = []
                                    for _, row in fatores_ord.head(3).iterrows():
                                        top_list.append(
                                            f"**{row['Fonte']}** ({row['Contribuição (%)']:.2f}%)"
                                        )
                                    st.markdown(
                                        "📈 **Maiores contribuições na razão S/N:** " + ", ".join(top_list)
                                    )

                            # Botão de download da ANOVA
                            buf_anova = io.StringIO()
                            anova_df.to_csv(buf_anova, index=False)
                            st.download_button(
                                "📥 Baixar tabela ANOVA (CSV)",
                                data=buf_anova.getvalue().encode("utf-8"),
                                file_name=f"anova_SN_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                key="dl_anova_sn",
                            )

                    if not HAS_SCIPY:
                        st.info(
                            "ℹ️ Os p-valores não foram calculados porque o pacote **SciPy** não está disponível.\n"
                            "Se desejar p-valores, instale SciPy no ambiente de execução:\n\n"
                            "`pip install scipy`"
                        )

        
                        
        
                            

        
        
        except Exception as e:
            st.error(f"❌ Erro ao processar o arquivo de resultados: {str(e)}")
