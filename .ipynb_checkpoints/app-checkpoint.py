import streamlit as st
import numpy as np
import pandas as pd
import io
import math
import matplotlib.pyplot as plt

# =============================================
# Configuração
# =============================================
st.set_page_config(page_title="Plano Experimental Taguchi", layout="wide")
st.title("Plano Experimental Taguchi")
st.caption("Upload de fatores e réplicas, seleção de OA, plano, S/N (das réplicas), efeitos médios com Delta e gráficos no estilo Minitab.")

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
    st.write("**Variável definida:** Produção de H₂")  # Valor padrão inicial

# ---------------------------------------------
# Utilitários de OA
# ---------------------------------------------
def built_in_catalog():
    """Catálogo interno (nomes amigáveis)."""
    return {
        "L4(2^3)"     : {"cols2": 3,  "cols3": 0,  "n": 4},
        "L8(2^7)"     : {"cols2": 7,  "cols3": 0,  "n": 8},
        "L9(3^4)"     : {"cols2": 0,  "cols3": 4,  "n": 9},
        "L12(2^11)"   : {"cols2": 11, "cols3": 0,  "n": 12},
        "L16(2^15)"   : {"cols2": 15, "cols3": 0,  "n": 16},
        "L18(2^1 3^7)": {"cols2": 1,  "cols3": 7,  "n": 18},
        "L27(3^13)"   : {"cols2": 0,  "cols3": 13, "n": 27},
    }

# Mapear nomes amigáveis -> nomes do pyDOE3 (quando existir diferença)
PYDOE3_NAME_MAP = {
    "L18(2^1 3^7)": "L18(6^1 3^6)",  # variante isomorfa em pyDOE3
    "L27(3^13)":    "L27(2^1 3^12)",
}

def oa_from_name(name: str) -> np.ndarray:
    """Retorna OA com codificação 0‑based (0,1) ou (0,1,2).
    Prioriza pyDOE3; se não encontrar, usa fallbacks internos (L4, L8, L9, L16, L18, L27).
    """
    # 1) Tenta pyDOE3 com nome mapeado (se disponível)
    if HAS_PYDOE3 and get_orthogonal_array is not None:
        try:
            lookup = PYDOE3_NAME_MAP.get(name, name)
            arr = np.asarray(get_orthogonal_array(lookup), dtype=int)
            # Normaliza para 0‑based se vier 1‑based
            if arr.min() == 1:
                arr = arr - 1
            return arr
        except Exception:
            pass  # cai no fallback interno

    # 2) Fallbacks internos (0-based)
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
            [0,0,0,0],
            [0,1,1,1],
            [0,2,2,2],
            [1,0,1,2],
            [1,1,2,0],
            [1,2,0,1],
            [2,0,2,1],
            [2,1,0,2],
            [2,2,1,0]
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
            [1,1,1,1,1,1,1],
            [1,1,2,2,2,2,2],
            [1,1,3,3,3,3,3],
            [1,2,1,1,2,2,3],
            [1,2,2,2,3,3,1],
            [1,2,3,3,1,1,2],
            [1,3,1,2,1,3,2],
            [1,3,2,3,2,1,3],
            [1,3,3,1,3,2,1],
            [2,1,1,3,3,2,2],
            [2,1,2,1,1,3,3],
            [2,1,3,2,2,1,1],
            [2,2,1,2,3,1,3],
            [2,2,2,3,1,2,1],
            [2,2,3,1,2,3,2],
            [2,3,1,3,2,3,1],
            [2,3,2,1,3,1,2],
            [2,3,3,2,1,2,3],
        ], dtype=int)
        col8 = np.array([[1],[2],[3],[3],[1],[2],[3],[1],[2],[1],[2],[3],[2],[3],[1],[2],[3],[1]], dtype=int)
        return np.hstack([(part1[:,0:1]-1), (part1[:,1:]-1), (col8-1)])

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
        # Ler o arquivo Excel
        df_fatores = pd.read_excel(upl, sheet_name='Fatores')

        # Verificar colunas obrigatórias
        if 'Factor' not in df_fatores.columns:
            st.error("❌ Coluna 'Factor' não encontrada no arquivo.")
        else:
            # Preview
            st.success("✅ Arquivo carregado com sucesso!")
            st.dataframe(df_fatores, use_container_width=True, hide_index=True)

            # -----------------------------------------
            # Análise automática dos fatores
            # -----------------------------------------
            st.subheader("🔍 Análise Automática dos Fatores")

            fatores = df_fatores['Factor'].astype(str).tolist()
            num_fatores = len(fatores)

            level_cols = [col for col in df_fatores.columns if col.startswith('Level')]
            niveis_por_fator = []
            niveis_rotulos = []  # lista de listas com os rótulos de níveis por fator

            for _, row in df_fatores.iterrows():
                lvls = [row[col] for col in level_cols if pd.notna(row[col])]
                niveis_por_fator.append(len(lvls))
                niveis_rotulos.append([str(x) for x in lvls])

            niveis_unicos = list(set(niveis_por_fator))
            mesmo_numero_niveis = len(niveis_unicos) == 1
            dof_necessario = sum([n - 1 for n in niveis_por_fator])

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Número de Fatores", num_fatores)
            with col2:
                if mesmo_numero_niveis:
                    st.metric("Níveis por Fator", f"{niveis_unicos[0]}")
                else:
                    st.metric("Níveis por Fator", f"misto: {min(niveis_por_fator)}–{max(niveis_por_fator)}")
            with col3:
                st.metric("Graus de Liberdade Necessários", dof_necessario)
            with col4:
                st.metric("Experimentos no Fatorial Completo",
                          full_factorial_runs(niveis_por_fator))

            # -----------------------------------------
            # Sugerir matrizes L possíveis (catálogo interno)
            # -----------------------------------------
            st.subheader("🎯 Matrizes Ortogonais Recomendadas")

            catalog = built_in_catalog()
            matrizes_candidatas = []

            for nome, specs in catalog.items():
                # DoF do array = n - 1
                if specs['n'] - 1 < dof_necessario:
                    continue

                if mesmo_numero_niveis:
                    # Verificar se a matriz tem colunas suficientes para o tipo de nível
                    if niveis_unicos[0] == 2 and specs['cols2'] >= num_fatores:
                        matrizes_candidatas.append((nome, specs))
                    elif niveis_unicos[0] == 3 and specs['cols3'] >= num_fatores:
                        matrizes_candidatas.append((nome, specs))
                else:
                    # Mistos: verificar se a matriz suporta a combinação de fatores
                    f2 = sum(1 for n in niveis_por_fator if n == 2)
                    f3 = sum(1 for n in niveis_por_fator if n == 3)
                    if specs['cols2'] >= f2 and specs['cols3'] >= f3:
                        matrizes_candidatas.append((nome, specs))

            # Ordenar por n (menos experimentos primeiro)
            matrizes_candidatas.sort(key=lambda x: x[1]['n'])

            if matrizes_candidatas:
                st.success(f"🎉 Foram encontradas {len(matrizes_candidatas)} matrizes adequadas!")

                # Tabela de recomendações (eficiência vs. fatorial completo)
                total_full = full_factorial_runs(niveis_por_fator)
                recomendacoes = []
                for nome, specs in matrizes_candidatas:
                    eficiencia = (1 - specs['n'] / total_full) * 100 if total_full > 0 else 0.0
                    recomendacoes.append({
                        "Matriz": nome,
                        "Experimentos (n)": specs['n'],
                        "Colunas (2 níveis)": specs['cols2'],
                        "Colunas (3 níveis)": specs['cols3'],
                        "Economia de corridas (%)": f"{eficiencia:.1f}%"
                    })

                df_recomendacoes = pd.DataFrame(recomendacoes)
                st.dataframe(df_recomendacoes, use_container_width=True, hide_index=True)
                st.caption("ℹ️ Economia de corridas em relação ao fatorial completo")
                st.latex(r"\text{Economia (\%)} = \Bigg( 1 - \frac{n_{OA}}{n_{fatorial}} \Bigg) \times 100")

                st.markdown("""
                - $n_{OA}$: número de experimentos da matriz ortogonal selecionada  
                - $n_{fatorial}$: número total de experimentos no fatorial completo (produto dos níveis de todos os fatores)
                """)

                # Recomendação principal
                matriz_recomendada = matrizes_candidatas[0][0]
                st.info(f"**Recomendação principal:** {matriz_recomendada} - {matrizes_candidatas[0][1]['n']} experimentos")

                # Seleção da matriz
                matriz_opcoes = [m[0] for m in matrizes_candidatas]
                st.subheader("🎛️ Seleção da Matriz Ortogonal")
                matriz_selecionada = st.selectbox(
                    "Escolha a matriz para gerar o experimento:",
                    options=matriz_opcoes,
                    index=0,
                    help="A matriz mais eficiente é selecionada por padrão. Você pode escolher outra se desejar."
                )

                # Botão para gerar matriz experimental
                if st.button("🔄 Gerar Matriz Experimental", type="primary"):
                    try:
                        matriz_oa = oa_from_name(matriz_selecionada)  # 0-based codes
                        # Usar somente o número de colunas necessárias aos fatores
                        if matriz_oa.shape[1] < num_fatores:
                            st.error("❌ A OA selecionada tem menos colunas do que o número de fatores.")
                        else:
                            matriz_oa = matriz_oa[:, :num_fatores]

                            # Montar DF mapeando códigos -> rótulos dos níveis informados
                            df_codificada = pd.DataFrame(matriz_oa, columns=fatores)
                            df_niveis = pd.DataFrame(index=df_codificada.index)

                            for j, fator in enumerate(fatores):
                                rotulos = niveis_rotulos[j]
                                max_code = matriz_oa[:, j].max()
                                if max_code >= len(rotulos):
                                    st.warning(
                                        f"⚠️ Fator **{fator}** tem {len(rotulos)} níveis no arquivo, "
                                        f"mas a coluna da OA possui código até {int(max_code)}. "
                                        "Revise o número de níveis."
                                    )
                                # mapear cada código para rótulo (se faltar rótulo, mantém código)
                                df_niveis[fator] = [
                                    rotulos[c] if c < len(rotulos) else f"lvl{c+1}"
                                    for c in matriz_oa[:, j]
                                ]

                            # Adicionar coluna índice do experimento
                            df_niveis.insert(0, "Experimento", range(1, len(df_niveis) + 1))

                            # Armazenar no session_state
                            st.session_state['matriz_selecionada'] = matriz_selecionada
                            st.session_state['matriz_oa'] = matriz_oa
                            st.session_state['df_fatores'] = df_fatores
                            st.session_state['df_experimentos_cod'] = df_codificada
                            st.session_state['df_experimentos'] = df_niveis
                            st.session_state['var_label'] = var_label
                            st.session_state['step'] = 'results'

                            st.success(f"✅ Matriz {matriz_selecionada} gerada com sucesso!")

                            # (Removido) NÃO exibir a tabela aqui para evitar duplicidade
                            # st.subheader("📊 Matriz Experimental Gerada")
                            # st.dataframe(df_niveis, use_container_width=True, hide_index=True)

                            # Botão de download CSV
                            @st.cache_data
                            def convert_df_to_csv(df):
                                return df.to_csv(index=False, sep=';').encode('utf-8')

                            csv = convert_df_to_csv(df_niveis)
                            st.download_button(
                                label="📥 Download da Matriz Experimental (CSV)",
                                data=csv,
                                file_name=f"matriz_experimental_{matriz_selecionada}.csv",
                                mime="text/csv",
                            )

                    except Exception as e:
                        st.error(f"❌ Erro ao gerar a matriz: {str(e)}")

            else:
                st.warning("⚠️ Nenhuma matriz ortogonal padrão adequada foi encontrada.")
                st.markdown("""
                **Sugestões:**
                - Reduza o número de fatores;
                - Use o mesmo número de níveis para todos os fatores;
                - Considere usar um array personalizado.
                """)

    except Exception as e:
        st.error(f"❌ Erro ao processar o arquivo: {str(e)}")

# =========================
# Seção persistente de Resultados (fora do if do botão)
# Renderiza se já houver experimento gerado na sessão
# =========================
if st.session_state.get('df_experimentos') is not None:
    st.markdown(":heavy_check_mark: **Matriz já gerada.** Siga com o upload dos resultados abaixo.")

    st.subheader("📊 Matriz Experimental Gerada")
    st.dataframe(st.session_state['df_experimentos'], use_container_width=True, hide_index=True)

    st.subheader("📤 Upload de Resultados Experimentais")
    var_label = st.session_state.get('var_label', 'Variável de Interesse')
    st.info(f"Execute os experimentos conforme a matriz acima e faça o upload dos resultados para {var_label}.")

    # Configuração de S/N
    sn_tipo = st.selectbox(
        "Tipo de razão S/N (Taguchi)",
        options=["Larger-the-better", "Smaller-the-better", "Nominal-the-best"],
        index=0,
        help=(
            "Selecione a fórmula da razão S/N. O cálculo usa as réplicas por corrida "
            "(colunas numéricas no arquivo de resultados)."
        )
    )
    nominal_target = None
    if sn_tipo == "Nominal-the-best":
        nominal_target = st.number_input("Alvo (m)", value=0.0, help="Alvo m para Nominal-the-best")

    # --- Fórmula em LaTeX conforme seleção ---
    sn_formulas = {
        "Larger-the-better":  r"\mathrm{S/N} = -10 \log_{10}\!\left(\tfrac{1}{n}\sum_{i=1}^{n}\tfrac{1}{y_i^{2}}\right)",
        "Smaller-the-better": r"\mathrm{S/N} = -10 \log_{10}\!\left(\tfrac{1}{n}\sum_{i=1}^{n} y_i^{2}\right)",
        "Nominal-the-best":   r"\mathrm{S/N} = 10 \log_{10}\!\left(\tfrac{m^{2}}{s^{2}}\right)\quad(\,m=\bar{y}\text{ se alvo não informado})"
    }
    st.markdown("**Fórmula S/N selecionada:**")
    st.latex(sn_formulas[sn_tipo])
    
    result_upl = st.file_uploader(
        "**Carregar arquivo de resultados**",
        type=["xlsx", "csv"],
        key="resultados_upl",
        help=(
            "CSV (separador ';') ou Excel. Deve conter a coluna 'Experimento' (1..n). "
            "As demais colunas numéricas serão interpretadas como réplicas da resposta."
        )
    )

    if result_upl:
        try:
            if result_upl.name.endswith('.csv'):
                df_resultados = pd.read_csv(result_upl, sep=';')
            else:
                df_resultados = pd.read_excel(result_upl)

            # Normaliza nome da coluna Experimento
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

                # Identifica colunas numéricas (réplicas)
                num_cols = [c for c in df_res.columns if c != "Experimento" and pd.api.types.is_numeric_dtype(df_res[c])]
                if len(num_cols) == 0:
                    st.error("❌ Nenhuma coluna numérica de resposta encontrada. Inclua colunas de réplicas.")
                else:
                    # --- Validação: número de experimentos e índices 1..n ---
                    df_plan = st.session_state['df_experimentos']
                    n_exp_plan = len(df_plan)  # total de linhas da matriz experimental
                    n_exp_res  = df_res['Experimento'].nunique()

                    # Checa duplicatas explícitas na coluna Experimento
                    dups = df_res['Experimento'][df_res['Experimento'].duplicated()].unique()
                    if len(dups) > 0:
                        st.error(f"❌ Há experimentos repetidos no arquivo de resultados: {sorted(dups)}")
                        st.stop()

                    # Checa quantidade
                    if n_exp_res != n_exp_plan:
                        st.error(
                            f"❌ O arquivo de resultados possui {n_exp_res} experimentos, "
                            f"mas a matriz experimental tem {n_exp_plan}."
                        )
                        st.stop()

                    # Checa se todos os índices 1..n estão presentes
                    esperados = set(range(1, n_exp_plan + 1))
                    presentes = set(df_res['Experimento'])
                    faltando  = sorted(esperados - presentes)
                    if faltando:
                        st.error(f"❌ Estão faltando os experimentos: {faltando}")
                        st.stop()

                    # Se passou por todas as checagens, prossegue
                    st.success("✅ Número de experimentos confere com a matriz experimental!")

                    # Agora sim: exibe e segue
                    st.success("✅ Arquivo de resultados carregado com sucesso!")
                    st.dataframe(df_res, use_container_width=True, hide_index=True)

                    # Junta com o plano experimental (para saber fatores/níveis de cada Experimento)
                    df_join = pd.merge(df_plan, df_res, on='Experimento', how='left')

                    # Calcula estatísticas por corrida
                    rep_values = df_join[num_cols].to_numpy(dtype=float)
                    mean_y = np.nanmean(rep_values, axis=1)
                    std_y = np.nanstd(rep_values, axis=1, ddof=1)

                    # Razão S/N por corrida (padrões Taguchi)
                    def sn_ratio(vals: np.ndarray, tipo: str) -> float:
                        vals = vals[~np.isnan(vals)]
                        r = len(vals)
                        if r == 0:
                            return np.nan
                        if tipo == "Larger-the-better":
                            # S/N = -10 * log10( (1/r) * sum(1/y^2) )
                            return -10.0 * math.log10(np.mean(1.0 / (vals ** 2)))
                        elif tipo == "Smaller-the-better":
                            # S/N = -10 * log10( (1/r) * sum(y^2) )
                            return -10.0 * math.log10(np.mean(vals ** 2))
                        else:  # Nominal-the-best
                            m_hat = np.mean(vals)
                            s2 = np.var(vals, ddof=1) if r > 1 else 0.0
                            m_t = nominal_target if nominal_target is not None else m_hat
                            # S/N = 10 * log10( m^2 / s^2 ) (aprox), usa alvo m_t
                            # Se variância ~0, evita divisão por zero
                            if s2 <= 1e-12:
                                return 10.0 * math.log10((m_t ** 2) / 1e-12)
                            return 10.0 * math.log10((m_t ** 2) / s2)

                    sn_vals = np.array([sn_ratio(rep_values[i, :], sn_tipo) for i in range(rep_values.shape[0])])

                    df_join["Média"] = mean_y
                    df_join["Desvio-Padrão"] = std_y
                    df_join["S/N"] = sn_vals

        except Exception as e:
            st.error(f"❌ Erro ao processar o arquivo de resultados: {str(e)}")
