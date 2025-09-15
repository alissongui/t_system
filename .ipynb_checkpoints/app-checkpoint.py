import streamlit as st
import numpy as np
import pandas as pd

# =============================================
# Configuração
# =============================================
st.set_page_config(page_title="Plano Experimental Taguchi", layout="wide")
st.title("Plano Experimental Taguchi")
st.caption("Upload de fatores e réplicas, seleção de OA, plano, S/N (da média e das réplicas) e efeitos médios com Delta.")

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
var_label = st.text_input("Variável de interesse (ex.: Produção de H₂)", "Produção de H₂",
                          help="Digite o nome da variável de interesse. Tecle ENTER ao finalizar!")
if var_label:
    st.success(f"✅ **Variável definida:** {var_label}")
else:
    st.write("**Variável definida:** Produção de H₂")  # Valor padrão inicial

# ---------------------------------------------
# Utilitários de OA
# ---------------------------------------------
def built_in_catalog():
    return {
        "L4(2^3)":      {"cols2": 3,  "cols3": 0,  "n": 4},
        "L8(2^7)":      {"cols2": 7,  "cols3": 0,  "n": 8},
        "L9(3^4)":      {"cols2": 0,  "cols3": 4,  "n": 9},
        "L12(2^11)":    {"cols2": 11, "cols3": 0,  "n": 12},
        "L16(2^15)":    {"cols2": 15, "cols3": 0,  "n": 16},
        "L18(2^1 3^7)":{"cols2": 1,  "cols3": 7,  "n": 18},
        "L27(3^13)":    {"cols2": 0,  "cols3": 13, "n": 27},
    }

def oa_from_name(name: str) -> np.ndarray:
    """Retorna OA com codificação 0-based (0,1) ou (0,1,2).
    Prioriza pyDOE3; caso contrário, usa fallbacks internos (L4, L8, L9, L16, L18, L27).
    """
    if HAS_PYDOE3:
        return np.asarray(get_orthogonal_array(name), dtype=int)

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
            [3,1,3,2,1,3,2,1,3,2,1,3,2],
            [3,1,3,2,2,1,3,2,1,3,2,1,3],
            [3,1,3,2,3,2,1,3,2,1,3,2,1],
            [3,2,1,3,1,3,2,2,1,3,3,2,1],
            [3,2,1,3,2,1,3,3,2,1,1,3,2],
            [3,2,1,3,3,2,1,1,3,2,2,1,3],
            [3,3,2,1,1,3,2,3,2,1,2,1,3],
            [3,3,2,1,2,1,3,1,3,2,3,2,1],
            [3,3,2,1,3,2,1,2,1,3,1,3,2],
        ], dtype=int)
        return arr27 - 1

    raise RuntimeError(f"OA '{name}' não disponível sem pyDOE3.")

def full_factorial_runs(levels_by_factor: list[int]) -> int:
    runs = 1
    for n in levels_by_factor:
        runs *= int(n)
    return runs

# ---------------------------------------------
# Upload de fatores
# ---------------------------------------------
upl = st.file_uploader("**Carregar arquivo de fatores**", type=["xlsx"],
                       help="Selecione o arquivo Excel com a configuração dos fatores (aba 'Fatores').")

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
            st.dataframe(df_fatores, use_container_width=True)

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
                    # CORREÇÃO: Verificar se a matriz tem colunas suficientes para o tipo de nível
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
                st.dataframe(df_recomendacoes, use_container_width=True)
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

                            st.success(f"✅ Matriz {matriz_selecionada} gerada com sucesso!")
                            
                            # Exibir a matriz experimental gerada
                            st.subheader("📊 Matriz Experimental Gerada")
                            st.dataframe(df_niveis, use_container_width=True, hide_index=True)
                            
                            # Adicionar botão para download
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
                            
                            # Adicionar seção para upload de resultados
                            st.subheader("📤 Upload de Resultados Experimentais")
                            st.info(f"Execute os experimentos conforme a matriz acima e faça o upload dos resultados para {var_label}.")
                            
                            result_upl = st.file_uploader("**Carregar arquivo de resultados**", type=["xlsx", "csv"],
                                                         help="Selecione o arquivo com os resultados experimentais.")
                            
                            if result_upl:
                                try:
                                    if result_upl.name.endswith('.csv'):
                                        df_resultados = pd.read_csv(result_upl, sep=';')
                                    else:
                                        df_resultados = pd.read_excel(result_upl)
                                    
                                    st.success("✅ Arquivo de resultados carregado avec sucesso!")
                                    st.dataframe(df_resultados, use_container_width=True, hide_index=True)
                                    
                                    # Aqui você pode adicionar a lógica para processar os resultados
                                    # e calcular S/N ratio, efeitos médios, etc.
                                    
                                except Exception as e:
                                    st.error(f"❌ Erro ao processar o arquivo de resultados: {str(e)}")

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