import streamlit as st
import numpy as np
import pandas as pd
import math
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio

# =============================================
# Configuração
# =============================================
st.set_page_config(page_title="Plano Experimental Taguchi", layout="wide")
st.title("Plano Experimental Taguchi")
st.caption("Até: upload de fatores, recomendação/geração de OA, upload de resultados, S/N e **Resultado por ensaio**. (Sem análises adicionais — aguardando comandos.)")

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

                            # Download CSV
                            @st.cache_data
                            def convert_df_to_csv(df):
                                return df.to_csv(index=False, sep=';').encode('utf-8')
                            csv = convert_df_to_csv(df_niveis)
                            st.download_button(
                                label="📥 Baixar Matriz Experimental (CSV)",
                                data=csv,
                                file_name=f"matriz_experimental_{matriz_selecionada}.csv",
                                mime="text/csv",
                            )
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
    st.markdown("---")
    st.subheader("📊 Matriz Experimental Gerada")
    st.dataframe(df_plan, use_container_width=True, hide_index=True)

    st.subheader("📤 Upload de Resultados Experimentais")
    var_label = st.session_state.get('var_label', 'Variável de Interesse')

    # Tipo de razão S/N
    sn_tipo = st.selectbox(
        "Tipo de razão S/N (Taguchi)",
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
    st.markdown("**Fórmula S/N selecionada:**")
    st.latex(sn_formulas[sn_tipo])

    result_upl = st.file_uploader(
        "**Carregar arquivo de resultados**",
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
                    st.subheader("📈 Efeitos médios dos fatores (S/N das réplicas)")
                    
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

                    COLS_PER_ROW = 4  # ajuste para 2, 3, ou 4 conforme preferir
                    
                    for i in range(0, len(factor_cols), COLS_PER_ROW):
                        # pega o “bloco” de até 4 fatores
                        bloco = factor_cols[i:i + COLS_PER_ROW]
                        cols = st.columns(len(bloco))
                        for j, fac in enumerate(bloco):
                            with cols[j]:
                                st.markdown(f"**Fator: {fac}**")
                                st.dataframe(per_factor_tables[fac], use_container_width=True, hide_index=True)

                    
                    # 2) Sumário: melhor nível, Delta e ranking
                    summary_rows = []
                    for fac, fac_df in per_factor_tables.items():
                        if fac_df.empty:
                            best_lvl = np.nan; delta = np.nan
                        else:
                            vmax = fac_df["S/N médio (dB)"].max()
                            vmin = fac_df["S/N médio (dB)"].min()
                            best_lvl = fac_df.loc[fac_df["S/N médio (dB)"] == vmax, "Nível"].iloc[0]
                            delta = float(vmax - vmin)
                        summary_rows.append({"Fator": fac, "Melhor nível": best_lvl, "Delta (dB)": delta})
                    
                    summary_df = pd.DataFrame(summary_rows).sort_values("Delta (dB)", ascending=False).reset_index(drop=True)
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
                    
                    st.markdown("**Níveis ótimos por fator — Taguchi (S/N das réplicas):**")
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
      
    

                    st.subheader("📄 Previsão do Desempenho Ótimo pelo Método Taguchi")

                    st.markdown("**Modelo aditivo (visão geral):**")
                    st.latex(r"\hat{Y}_{\text{prev}} \;=\; \bar{Y} \;+\; \sum_{k=1}^{n} \;\text{Efeito}_{k,\ell^\star}")
                    
                    st.markdown("**Definição do efeito (por fator no nível ótimo):**")
                    st.latex(r"\text{Efeito}_{k,\ell^\star} \;=\; \bar{Y}_{k,\ell^\star} \;-\; \bar{Y}")
                    
                    st.markdown(r"""
                    **Onde:**
                    - `Y_previsto` ($\hat{Y}_{\text{prev}}$) = valor previsto da característica de qualidade na condição ótima  
                    - `Y_global` ($\bar{Y}$) = média geral de todas as observações do experimento  
                    - `Efeito do Fator no nível ótimo` ($\bar{Y}_{k,\ell^\star} - \bar{Y}$) = contribuição do fator $k$ no seu melhor nível $\ell^\star$  
                    - $n$ = número total de fatores
                    """)

                    st.markdown("**Forma equivalente (soma das melhores médias):**")
                    st.latex(r"\hat{Y}_{\text{prev}} \;=\; \left(\sum_{k=1}^{n}  \bar{Y}_{k,\ell^\star}\right) \;-\; (n-1)\,\bar{Y}")   
              

                    st.subheader("🎯 Estimativa de valores")
                    
                    colY, colSN = st.columns(2)
                    
                    # -------------------- COLUNA ESQUERDA: Y --------------------
                    with colY:
                        st.markdown(f"**Estimativa do Problema ({var_label})**")
                    
                        try:
                            col_media_otimo = f"Média de {var_label} no Nível Ótimo"
                            if 'opt_table' in locals() and col_media_otimo in opt_table.columns:
                                Y_best_means = opt_table[col_media_otimo].to_numpy(dtype=float)
                                k = len(Y_best_means)
                    
                                Y_bar = float(np.nanmean(mean_y)) if 'mean_y' in locals() else float('nan')
                    
                                if k > 0 and not np.isnan(Y_bar) and not np.isnan(Y_best_means).any():
                                    Y_hat_taguchi = float(np.sum(Y_best_means) - (k - 1) * Y_bar)
                                else:
                                    Y_hat_taguchi = float("nan")
                            else:
                                st.warning("Níveis ótimos ainda não calculados; gere a tabela antes.")
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
                    
                    # -------------------- COLUNA DIREITA: S/N --------------------
                    with colSN:
                        st.markdown(f"**Estimativa da Relação Sinal-Ruído (S/N) — {var_label}**")
                    
                        try:
                            grand_mean = float(grand_mean)
                        except Exception:
                            grand_mean = float(df_effects[sn_col].mean())
                    
                        try:
                            _needs_init = (not selected_level_means)
                        except NameError:
                            _needs_init = True
                    
                        if _needs_init:
                            selected_level_means = []
                            for fac in factor_cols:
                                fac_df = per_factor_tables[fac]
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


        
        except Exception as e:
            st.error(f"❌ Erro ao processar o arquivo de resultados: {str(e)}")
