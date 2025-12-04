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

# (se tiver scipy / pyDOE, ficam aqui também)

# =============================================
# Configuração DA PÁGINA  (TEM QUE SER A PRIMEIRA COISA do Streamlit)
# =============================================
st.set_page_config(page_title="TaguchiApp", layout="wide")

st.title("TaguchiApp")
st.caption(
    """
    <div style="font-size:16px; font-weight:bold;">
        Taguchi App — Planejamento e Análise Experimental Taguchi — Versão v25.02<br><br>
    </div>
    """,
    unsafe_allow_html=True
)

# aqui embaixo vêm as suas funções: oa_from_name, built_in_catalog, section_factors_and_oa, section_results, etc.


# ============================
# Imports opcionais
# ============================
try:
    from scipy.stats import f as f_dist, t as t_dist
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False
    f_dist = None
    t_dist = None

try:
    from pyDOE3 import get_orthogonal_array
    HAS_PYDOE3 = True
except Exception:
    HAS_PYDOE3 = False
    get_orthogonal_array = None


# ============================
# Utilitários de OA / catálogo
# ============================
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
            # se vier 1/2/3, convertemos para 0/1/2:
            if arr.min() == 1:
                arr = arr - 1
            return arr
        except Exception:
            # se der erro no pyDOE3, cai pros fallbacks internos
            pass

    # 2) Fallbacks internos (0-based)
    import numpy as np

    if name == "L4(2^3)":
        return np.array([
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=int)

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
            [0,0,0,0],
            [0,1,1,1],
            [0,2,2,2],
            [1,0,1,2],
            [1,1,2,0],
            [1,2,0,1],
            [2,0,2,1],
            [2,1,0,2],
            [2,2,1,0],
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
            [2,3,1,3,2,3,1],[2,3,2,1,3,1,2],[2,3,3,2,1,2,3],
        ], dtype=int)
        col8 = np.array([
            [1],[2],[3],[3],[1],[2],[3],[1],[2],
            [1],[2],[3],[2],[3],[1],[2],[3],[1],
        ], dtype=int)
        return np.hstack([(part1[:, 0:1] - 1), (part1[:, 1:] - 1), (col8 - 1)])

    if name == "L27(3^13)":
        arr27 = np.array([
            [1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,2,2,2,2,2,2,2,2,2],
            [1,1,1,1,3,3,3,3,3,3,3,3,3],
            [1,2,2,2,1,1,1,2,2,2,3,3,3],
            [1,2,2,2,2,2,2,3,3,3,1,1,1],
            [1,2,2,2,3,3,3,1,1,1,2,2,2],
            [1,3,3,3,1,1,1,3,3,3,2,2,2],
            [1,3,3,3,2,2,2,1,1,1,3,3,3],
            [1,3,3,3,3,3,3,2,2,2,1,1,1],
            [2,1,2,3,1,2,3,1,2,3,1,2,3],
            [2,1,2,3,2,3,1,2,3,1,2,3,1],
            [2,1,2,3,3,1,2,3,1,2,3,1,2],
            [2,2,3,1,1,2,3,2,3,1,3,1,2],
            [2,2,3,1,2,3,1,3,1,2,1,2,3],
            [2,2,3,1,3,1,2,1,2,3,2,3,1],
            [2,3,1,2,1,2,3,3,1,2,2,3,1],
            [2,3,1,2,2,3,1,1,2,3,3,1,2],
            [2,3,1,2,3,1,2,2,3,1,1,2,3],
        ], dtype=int)
        return arr27 - 1

    # se nada casou:
    raise RuntimeError(f"OA '{name}' não disponível.")



def full_factorial_runs(levels_by_factor: list[int]) -> int:
    runs = 1
    for n in levels_by_factor:
        runs *= int(n)
    return runs


# ============================
# Configuração da página
# ============================
def configure_page():
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

    # estado do fluxo (wizard)
    if "step" not in st.session_state:
        st.session_state["step"] = "start"


# ============================
# Entrada da variável de interesse
# ============================
def input_var_label() -> str:
    var_label = st.text_input(
        "Variável de interesse (ex.: Produção de H₂)",
        "Produção de H₂",
        help="Digite o nome da variável de interesse. Tecle ENTER ao finalizar!",
    )

    if var_label:
        st.success(f"✅ **Variável definida:** {var_label}")
    else:
        st.write("**Variável definida:** Produção de H₂")

    # salva no session_state para uso em outras seções
    st.session_state["var_label"] = var_label or "Produção de H₂"
    return st.session_state["var_label"]


# ---------------------------------------------
# Upload de fatores (em função)
# ---------------------------------------------
def section_factors_and_oa():
    with st.container():
        upl = st.file_uploader(
            "**Carregar arquivo de fatores**",
            type=["xlsx"],
            key="fatores_upl",
            help="Selecione o arquivo Excel com a configuração dos fatores (aba 'Fatores')."
        )

        # Se nada foi enviado, apenas sai da função
        if not upl:
            return

        try:
            df_fatores = pd.read_excel(upl, sheet_name='Fatores')
            if 'Factor' not in df_fatores.columns:
                st.error("❌ Coluna 'Factor' não encontrada no arquivo.")
                return

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
            c2.metric(
                "Níveis por Fator",
                f"{niveis_unicos[0]}" if mesmo_numero_niveis else f"misto: {min(niveis_por_fator)}–{max(niveis_por_fator)}"
            )
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

            if not matrizes_candidatas:
                st.warning("⚠️ Nenhuma matriz ortogonal padrão adequada foi encontrada.")
                return

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
                        return

                    matriz_oa = matriz_oa[:, :num_fatores]
                    df_codificada = pd.DataFrame(matriz_oa, columns=fatores)

                    df_niveis = pd.DataFrame(index=df_codificada.index)
                    for j, fator in enumerate(fatores):
                        rotulos = niveis_rotulos[j]
                        max_code = matriz_oa[:, j].max()
                        if max_code >= len(rotulos):
                            st.warning(
                                f"⚠️ Fator **{fator}** tem {len(rotulos)} níveis, "
                                f"mas a OA possui código até {int(max_code)}. Revise."
                            )
                        df_niveis[fator] = [
                            rotulos[c] if c < len(rotulos) else f"lvl{c+1}"
                            for c in matriz_oa[:, j]
                        ]

                    df_niveis.insert(0, "Experimento", range(1, len(df_niveis) + 1))

                    # 🔴 Aqui salvamos tudo no session_state
                    st.session_state['matriz_selecionada'] = matriz_selecionada
                    st.session_state['matriz_oa'] = matriz_oa
                    st.session_state['df_fatores'] = df_fatores
                    st.session_state['df_experimentos_cod'] = df_codificada
                    st.session_state['df_experimentos'] = df_niveis
                    st.session_state['var_label'] = st.session_state.get('var_label', 'Variável de Interesse')
                    st.session_state['step'] = 'results'

                    st.success(f"✅ Matriz {matriz_selecionada} gerada com sucesso!")

                except Exception as e:
                    st.error(f"❌ Erro ao gerar a matriz: {str(e)}")

        except Exception as e:
            st.error(f"❌ Erro ao processar o arquivo: {str(e)}")


# =========================
# Seção persistente de Resultados (compacta e modular)
# =========================
def section_results():

    # Se ainda não existe matriz experimental, sai da função
    if st.session_state.get("df_experimentos") is None:
        return

    df_plan = st.session_state["df_experimentos"]
    df_cod = st.session_state.get("df_experimentos_cod")
    var_label = st.session_state.get("var_label", "Variável")
    matriz_selecionada = st.session_state.get("matriz_selecionada", "OA")

    # ======================================================
    # DOWNLOAD DA MATRIZ
    # ======================================================
    st.subheader("📊 Matriz Experimental Gerada")
    st.dataframe(df_plan, use_container_width=True, hide_index=True)

    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False, sep=";").encode("utf-8")

    st.download_button(
        "📥 Baixar Matriz Experimental (CSV)",
        data=convert_df(df_plan),
        file_name=f"matriz_experimental_{matriz_selecionada}.csv",
        mime="text/csv",
    )

    st.markdown("---")

        # ======================================================
    # Função 1 — Upload dos Resultados
    # ======================================================
    def upload_resultados():
        st.subheader("📤 Upload de Resultados Experimentais (Réplicas/triplicatas)")
        var_label_local = st.session_state.get("var_label", "Variável de Interesse")

        # Tipo de razão S/N (igual ao app_regressao)
        sn_tipo = st.selectbox(
            "Tipo de razão Sinal-Ruído (S/N) (Taguchi)",
            options=["Maior é melhor", "Menor é melhor", "Nominal é melhor"],
            index=0,
        )
        alvo_nominal = None
        if sn_tipo == "Nominal é melhor":
            alvo_nominal = st.number_input("Alvo (m)", value=0.0, help="Para Nominal é melhor")

        # Fórmula em LaTeX (igual ao app_regressao)
        sn_formulas = {
            "Maior é melhor":  r"S/N = -10 \log_{10} \left( \dfrac{1}{n} \sum_{i=1}^{n} \dfrac{1}{y_i^{2}} \right)",
            "Menor é melhor":  r"S/N = -10 \log_{10} \left( \dfrac{1}{n} \sum_{i=1}^{n} y_i^{2} \right)",
            "Nominal é melhor": r"S/N = 10 \log_{10} \left( \dfrac{m^{2}}{s^{2}} \right) \quad (m = \bar{y} \text{ se alvo não informado})"
        }
        st.markdown("**Fórmula da Razão Sinal-Ruído (S/N) selecionada:**")
        st.latex(sn_formulas[sn_tipo])

        # Upload do arquivo de resultados
        upl = st.file_uploader(
            "**Carregar arquivo de resultados (réplicas do experimento)**",
            type=["xlsx", "csv"],
            key="resultados_upl",
        )
        if not upl:
            return None, None, sn_tipo, alvo_nominal

        # --------- Leitura ---------
        if upl.name.endswith(".csv"):
            df_resultados = pd.read_csv(upl, sep=";")
        else:
            df_resultados = pd.read_excel(upl)

        # Padroniza nome da coluna "Experimento"
        exp_col = None
        for c in df_resultados.columns:
            if str(c).strip().lower() in {"experimento", "experiments", "exp", "run"}:
                exp_col = c
                break

        if exp_col is None:
            st.error("❌ O arquivo de resultados precisa ter a coluna 'Experimento'.")
            return None, None, sn_tipo, alvo_nominal

        df_res = df_resultados.copy()
        df_res.rename(columns={exp_col: "Experimento"}, inplace=True)

        # Colunas numéricas (réplicas)
        num_cols = [
            c for c in df_res.columns
            if c != "Experimento" and pd.api.types.is_numeric_dtype(df_res[c])
        ]
        if len(num_cols) == 0:
            st.error("❌ Nenhuma coluna numérica de resposta encontrada.")
            return None, None, sn_tipo, alvo_nominal

        # ======= Validações =======
        n_exp_plan = len(df_plan)
        n_exp_res = df_res["Experimento"].nunique()
        dups = df_res["Experimento"][df_res["Experimento"].duplicated()].unique()
        if len(dups) > 0:
            st.error(f"❌ Há experimentos repetidos: {sorted(dups)}")
            return None, None, sn_tipo, alvo_nominal

        if n_exp_res != n_exp_plan:
            st.error(
                f"❌ Resultados possuem {n_exp_res} experimentos; plano tem {n_exp_plan}."
            )
            return None, None, sn_tipo, alvo_nominal

        esperados = set(range(1, n_exp_plan + 1))
        presentes = set(df_res["Experimento"])
        faltando = sorted(esperados - presentes)
        if faltando:
            st.error(f"❌ Faltando experimentos: {faltando}")
            return None, None, sn_tipo, alvo_nominal

        # Mensagens de sucesso + mostra a matriz de resultados carregada
        st.success("✅ Número de experimentos confere com a matriz experimental!")
        st.success("✅ Arquivo de resultados carregado com sucesso!")
        st.dataframe(df_res, use_container_width=True, hide_index=True)
        st.markdown("---")

        # Join com o plano (df_plan) — matriz combinada
        df_join = df_plan.merge(df_res, on="Experimento", how="left")

        return df_join, num_cols, sn_tipo, alvo_nominal

    # 🔹 Aqui era onde estava faltando a chamada:
    df_join, num_cols, sn_tipo, alvo_nominal = upload_resultados()

    # Se o usuário ainda não fez upload, para por aqui
    if df_join is None:
        return

    # ======================================================
    # Função 2 — Cálculo de médias e S/N
    # ======================================================
    def calcular_sn():
        reps = df_join[num_cols].to_numpy(dtype=float)

        # Médias Y
        mean_y = np.nanmean(reps, axis=1)

        # Desvios
        std_y = np.nanstd(reps, axis=1, ddof=1)

        # --- Funções S/N ---
        def sn_larger(vals):
            return -10 * np.log10(np.mean(1.0 / (vals**2)))

        def sn_smaller(vals):
            return -10 * np.log10(np.mean(vals**2))

        def sn_nominal(vals, target):
            if len(vals) < 2:
                return np.nan
            return 10 * np.log10((target**2) / np.var(vals, ddof=1))

        SNR = []
        for row in reps:
            vals = row[~np.isnan(row)]
            if sn_tipo == "Maior é melhor":
                SNR.append(sn_larger(vals))
            elif sn_tipo == "Menor é melhor":
                SNR.append(sn_smaller(vals))
            else:
                SNR.append(sn_nominal(vals, alvo_nominal))

        df_local = df_join.copy()
        df_local["_Ymean"] = mean_y
        df_local["_SN"] = SNR

        return df_local, mean_y, SNR

    df_join, mean_y, SNR = calcular_sn()



    # Nada a fazer ainda
    if df_join is None:
        return

    # ======================================================
    # Função 2 — Cálculo de médias e S/N
    # ======================================================
    def calcular_sn():
        # usa o df_join do escopo externo apenas para leitura
        reps = df_join[num_cols].to_numpy(dtype=float)

        # Médias Y
        mean_y = np.nanmean(reps, axis=1)

        # Desvios
        std_y = np.nanstd(reps, axis=1, ddof=1)

        # --- Funções S/N ---
        def sn_larger(vals):
            return -10 * np.log10(np.mean(1.0 / (vals**2)))

        def sn_smaller(vals):
            return -10 * np.log10(np.mean(vals**2))

        def sn_nominal(vals, target):
            if len(vals) < 2:
                return np.nan
            return 10 * np.log10((target**2) / np.var(vals, ddof=1))

        SNR = []
        for row in reps:
            vals = row[~np.isnan(row)]
            if sn_tipo == "Maior é melhor":
                SNR.append(sn_larger(vals))
            elif sn_tipo == "Menor é melhor":
                SNR.append(sn_smaller(vals))
            else:
                SNR.append(sn_nominal(vals, alvo_nominal))

        # 🔹 trabalha em uma cópia, não no df_join “de fora”
        df_local = df_join.copy()
        df_local["_Ymean"] = mean_y
        df_local["_SN"] = SNR

        return df_local, mean_y, SNR


    df_join, mean_y, SNR = calcular_sn()

        # ======================================================
    # Resumo: resultado por ensaio + médias globais
    # (comportamento similar ao app_regressao)
    # ======================================================
    st.markdown("### 📊 Resultado por ensaio")

    sn_table = pd.DataFrame({
        "Experimento": df_plan["Experimento"],
        f"Média de {var_label}": mean_y.astype(float),
        f"S/N das réplicas ({var_label}) [dB]": SNR,
    })

    st.dataframe(sn_table, use_container_width=True, hide_index=True)

    # Médias globais
    Y_bar = float(np.nanmean(mean_y))
    SN_bar = float(np.nanmean(SNR))

    c1, c2 = st.columns(2)
    with c1:
        st.metric(
            label=f"Média global de {var_label}",
            value=f"{Y_bar:.3f}",
        )
    with c2:
        st.metric(
            label="Média global de S/N (réplicas)",
            value=f"{SN_bar:.3f} dB",
        )

    st.markdown("---")


    # ======================================================
    # Pré-cálculo global para todas as funções da aba
    # ======================================================
    factor_cols = [c for c in df_plan.columns if c != "Experimento"]
    
    per_factor = {}     # S/N médio por nível
    per_factor_Y = {}   # Y médio por nível
    grand_mean = float(np.mean(SNR))
    
    for f in factor_cols:
        df_tmp = df_join.copy()
        df_tmp[f] = df_tmp[f].astype(str)
    
        # S/N MÉDIO POR NÍVEL
        g_sn = df_tmp.groupby(f)["_SN"].mean()
        per_factor[f] = g_sn.to_frame("S/N médio")
    
        # Y MÉDIO POR NÍVEL
        g_y = df_tmp.groupby(f)["_Ymean"].mean()
        per_factor_Y[f] = g_y.to_dict()


    # ======================================================
    # Função 3 — Efeitos + Gráficos
    # ======================================================
    def mostrar_efeitos_e_graficos():
        st.subheader("📈 Efeitos principais na razão S/N (médias por nível)")

        # 🔀 Toggle com a explicação do efeito (igual ao app_regressao)
        if st.toggle("🔴🔴🔴 O que é o 'efeito'? (clique para ver) 🔴🔴🔴",
                     value=False,
                     key="show_efeito"):
            st.markdown(
                r"""
                O **efeito** de um fator $k$ no nível $\ell$ é definido como o desvio
                da resposta média da razão Sinal-Ruído (S/N), obtida nesse nível específico,
                em relação à média global do experimento. Em outros termos, para cada
                **fator** denotado por $k$ e cada **nível** $\ell$ desse fator,
                define-se o efeito como a diferença entre a média de S/N nesse nível
                e a média global:
                """
            )
            st.latex(
                r"\text{Efeito}(k,\ell)=\overline{\mathrm{S/N}}_{k,\ell}"
                r"-\overline{\mathrm{S/N}}_{\text{global}}"
            )
            st.markdown(
                r"""
                **em que,**  
                • $k \in \{1,\dots,K\}$ é o índice do fator (ex.: Temperatura, Pressão, ...),
                  sendo $K$ o número total de fatores.  

                • $\ell \in \{1,\dots,L_k\}$ representa o índice do nível do fator $k$,
                  sendo $L_k$ o número de níveis do respectivo fator. 

                • $\overline{\mathrm{S/N}}_{k,\ell}$: média da razão Sinal-Ruído considerando
                  apenas os ensaios em que o fator $k$ foi fixado no nível $\ell$.

                • $\overline{\mathrm{S/N}}_{\text{global}}$: média da razão Sinal-Ruído
                  considerando todos os ensaios do experimento.
                """
            )

        st.markdown("---")

        # Vamos usar o df_join com a coluna "_SN" (S/N das réplicas)
        df_effects = df_join.copy()
        sn_col = "_SN"

        # Tabelas formatadas por fator (para exibição)
        per_factor_tables = {}

        for fac in factor_cols:
            # níveis como string, ordenados naturalmente (1,2,3,...)
            lvls_in_plan = df_plan[fac].astype(str).unique().tolist()
            try:
                order_nat = sorted(lvls_in_plan, key=lambda s: int(s))
            except Exception:
                order_nat = sorted(lvls_in_plan)

            tmp = df_effects.copy()
            tmp[fac] = tmp[fac].astype(str)

            g = (
                tmp
                .groupby(fac, as_index=True)[sn_col]
                .mean()
                .reindex(order_nat)
            )

            fac_df = (
                pd.DataFrame({"Nível": g.index, "S/N médio (dB)": g.values})
                .reset_index(drop=True)
            )

            # Garante numérico e calcula Efeito (dB)
            fac_df["S/N médio (dB)"] = pd.to_numeric(
                fac_df["S/N médio (dB)"], errors="coerce"
            )
            fac_df["Efeito (dB)"] = fac_df["S/N médio (dB)"] - float(grand_mean)
            fac_df[["S/N médio (dB)", "Efeito (dB)"]] = fac_df[
                ["S/N médio (dB)", "Efeito (dB)"]
            ].round(3)

            per_factor_tables[fac] = fac_df

        # Renderiza as tabelas — até 4 fatores por linha
        COLS_PER_ROW = 4
        for i in range(0, len(factor_cols), COLS_PER_ROW):
            bloco = factor_cols[i:i + COLS_PER_ROW]
            cols = st.columns(len(bloco))
            for j, fac in enumerate(bloco):
                with cols[j]:
                    st.markdown(f"**Fator: {fac}**")
                    st.dataframe(
                        per_factor_tables[fac],
                        use_container_width=True,
                        hide_index=True,
                    )

        # Mantém o mesmo retorno de antes (para compatibilidade)
        return per_factor, grand_mean, factor_cols




    def mostrar_interacoes():
        if len(factor_cols) < 2:
            return
        st.subheader("🔗 Interações entre fatores")
        fac_x = st.selectbox("Fator no eixo X:", factor_cols)
        fac_l = st.selectbox("Fator para curvas:", [f for f in factor_cols if f != fac_x])

        df_tmp = df_join.copy()
        df_tmp[fac_x] = df_tmp[fac_x].astype(str)
        df_tmp[fac_l] = df_tmp[fac_l].astype(str)

        g = df_tmp.groupby([fac_x, fac_l])["_SN"].mean().reset_index()

        fig = go.Figure()
        for lvl in sorted(g[fac_l].unique()):
            sub = g[g[fac_l] == lvl]
            fig.add_trace(go.Scatter(
                x=sub[fac_x], y=sub["_SN"],
                mode="lines+markers",
                name=f"{fac_l}={lvl}"
            ))
        fig.update_layout(
            xaxis_title=f"Níveis de {fac_x}",
            yaxis_title="S/N médio (dB)"
        )
        st.plotly_chart(fig, use_container_width=True)

    def mostrar_superficie_3d():
        if len(factor_cols) < 2:
            return

        st.subheader("🌐 Superfície 3D (Y médio × 2 fatores)")

        fx = st.selectbox("Fator X (3D):", factor_cols)
        fy = st.selectbox("Fator Y (3D):", [f for f in factor_cols if f != fx])

        df_tmp = df_join.copy()
        df_tmp[fx] = df_tmp[fx].astype(str)
        df_tmp[fy] = df_tmp[fy].astype(str)

        grid = df_tmp.groupby([fx, fy])["_Ymean"].mean().reset_index()

        xs = sorted(grid[fx].unique(), key=lambda z: int(z))
        ys = sorted(grid[fy].unique(), key=lambda z: int(z))

        Z = np.zeros((len(ys), len(xs)))
        for i, yv in enumerate(ys):
            for j, xv in enumerate(xs):
                val = grid[(grid[fx] == xv) & (grid[fy] == yv)]["_Ymean"]
                Z[i, j] = float(val)

        fig = go.Figure(data=[go.Surface(
            x=list(range(len(xs))),
            y=list(range(len(ys))),
            z=Z,
            colorscale="Viridis"
        )])
        fig.update_layout(
            scene=dict(
                xaxis=dict(ticktext=xs, tickvals=list(range(len(xs)))),
                yaxis=dict(ticktext=ys, tickvals=list(range(len(ys)))),
                zaxis_title=var_label
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    def mostrar_regra_delta():
        st.subheader("📐 Regra Delta")

        rows = []
        for fac, g in per_factor.items():
            vals = g["S/N médio"]
            vmax = vals.max()
            vmin = vals.min()
            rows.append({
                "Fator": fac,
                "Delta": vmax - vmin,
                "Melhor nível": vals.idxmax()
            })

        df_delta = pd.DataFrame(rows).sort_values("Delta", ascending=False)
        st.dataframe(df_delta, use_container_width=True)

    def tabelas_observado_predito():
        st.subheader("📊 Observado × Predito")

        Y = df_join["_Ymean"].values
        SN = df_join["_SN"].values

        factor_cols_local = [c for c in df_plan.columns if c != "Experimento"]

        Y_bar = np.mean(Y)
        SN_bar = np.mean(SN)

        predY = []
        predSN = []

        # ATENÇÃO: aqui ainda falta você definir per_factor_Y em algum lugar
        for i in range(len(df_plan)):
            somaY = 0
            somaSN = 0
            for fac in factor_cols_local:
                lvl = str(df_join.loc[i, fac])
                somaY += per_factor_Y[fac][lvl]
                somaSN += per_factor[fac].loc[lvl, "S/N médio"]
            predY.append(somaY - (len(factor_cols_local) - 1) * Y_bar)
            predSN.append(somaSN - (len(factor_cols_local) - 1) * SN_bar)

        df_pred = pd.DataFrame({
            "Y_obs": Y,
            "Y_pred": predY,
            "SN_obs": SN,
            "SN_pred": predSN
        })

        st.dataframe(df_pred.round(3))

    def predicao_usuario():
        st.subheader("🧮 Predição para qualquer combinação")

        levels = {}
        for f in factor_cols:
            lvls = sorted(df_plan[f].astype(str).unique(), key=lambda z: int(z))
            levels[f] = st.selectbox(f"Nível para {f}", lvls)

        SN_bar = np.mean(SNR)
        somaSN = 0
        for f, lvl in levels.items():
            somaSN += per_factor[f].loc[lvl, "S/N médio"]
        pred_sn = somaSN - (len(factor_cols) - 1) * SN_bar

        st.success(f"**S/N predito = {pred_sn:.3f} dB**")

    def regressao_multipla():
        st.subheader("📉 Regressão múltipla (opcional)")
        ativar = st.checkbox("Ativar regressão múltipla", value=False)
        if not ativar:
            return

        bloco_regressao_multipla(
            df_plan=df_plan,
            df_design_cod=df_cod,
            mean_y=mean_y,
            df_effects=df_join,
            sn_col="_SN",
            var_label=var_label
        )

    # =============================================
    # 🔖 Abas de resultados
    # =============================================

    st.markdown("""
    <style>

    /* TODAS as abas (ativa e inativas) */
    button[role="tab"] {
        font-size: 24px !important;      /* fonte bem maior */
        font-weight: 900 !important;     /* negrito forte */
        padding: 14px 26px !important;   /* mais área clicável */
        margin-right: 10px !important;
        border-radius: 12px 12px 0 0 !important;
        border: none !important;
        background: #e3e9f7 !important;  /* cinza-azulado claro */
        color: #0a2d5c !important;       /* azul escuro */
    }

    /* ABA ATIVA */
    button[role="tab"][aria-selected="true"] {
        background: #ffffff !important;  /* fundo branco */
        color: #000000 !important;       /* texto preto */
        font-size: 28px !important;      /* ainda maior */
        font-weight: 900 !important;     /* super negrito */
    }

    </style>
    """, unsafe_allow_html=True)




    tab_efeitos, tab_inter, tab_3d, tab_pred, tab_reg = st.tabs(
        ["Efeitos principais & Delta", "Interações", "Superfície 3D", "Predições", "Regressão múltipla"]
    )

    with tab_efeitos:
        per_factor, grand_mean, factor_cols = mostrar_efeitos_e_graficos()
        mostrar_regra_delta()

    with tab_inter:
        mostrar_interacoes()

    with tab_3d:
        mostrar_superficie_3d()

    with tab_pred:
        tabelas_observado_predito()
        predicao_usuario()

    with tab_reg:
        regressao_multipla()



def main():
    # A parte de configuração da página e var_label já está no topo do arquivo,
    # então aqui só chamamos os blocos principais.
    section_factors_and_oa()
    section_results()


if __name__ == "__main__":
    main()

