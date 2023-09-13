import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Carregue seu modelo e scaler
model_dir = '../../../GitHub/Machine_learning/Aprendizado Supervisionado/Classificação/Novexus Churn/models/model_random_forest_novexus_churn.pkl'
scaler_dir = '../../../GitHub/Machine_learning/Aprendizado Supervisionado/Classificação/Novexus Churn/models/scaler_random_forest_novexus_churn.pkl'

with open(model_dir, 'rb') as f:
    model = pickle.load(f)

with open(scaler_dir, 'rb') as f:
    scaler = pickle.load(f)

# Título com emoji
st.title("📞 Previsão de Churn Telefônico 📞")

# Defina as variáveis categóricas e suas opções
variaveis_categoricas = {
    "Idoso:": ["Sim", "Não"],
    "Possui contrato ativo:": ["Sim", "Não"],
    "Selecione o sexo:": ["Masculino", "Feminino"],
    "Possui conjugue:": ["Sim", "Não"],
    "Possui dependentes:": ["Sim", "Não"],
    "Possui serviço telefonico:": ["Sim", "Não"],
    "Possui mais de uma linha:": ["Sim", "Não"],
    "Possui fibra Otica:": ["Sim", "Não"],
    "Possui internet:": ["Sim", "Não"],
    "Possui segurança online:": ["Sim", "Não"],
    "Possui backup:": ["Sim", "Não"],
    "Possui proteção dispositivo:": ["Sim", "Não"],
    "Possui suporte tecnico:": ["Sim", "Não"],
    "Possui stream TV:": ["Sim", "Não"],
    "Possui stream Filmes:": ["Sim", "Não"],
    "Escolha o tipo de contrato:": ["Contrato de Dois Anos", "Contrato Mensal"],
    "Qual o tipo de Fatura:": ['Fatura Online', 'Boleto', 'Eletronico', 'Transferência Automatica']
}

# Divida a tela em duas colunas
col1, col2 = st.columns(2)

# Crie widgets para as variáveis categóricas e numéricas na primeira coluna
valores_col1 = {}
for variavel, opcoes in variaveis_categoricas.items():
    if variavel == "Valor Mensal:" or variavel == "Valor Total:":
        valores_col1[variavel] = col1.number_input(variavel, min_value=0.0)
    else:
        valores_col1[variavel] = col1.selectbox(variavel, opcoes)

# Crie widgets para as variáveis categóricas e numéricas na segunda coluna
valores_col2 = {}
for variavel, opcoes in variaveis_categoricas.items():
    if variavel == "Valor Mensal:" or variavel == "Valor Total:":
        valores_col2[variavel] = col2.number_input(variavel, min_value=0.0)
    else:
        valores_col2[variavel] = col2.selectbox(variavel, opcoes)

# Combine os valores das duas colunas
valores = {**valores_col1, **valores_col2}

# Botão para fazer a previsão
if st.button("Obter Previsão"):
    # Mapeie as opções "Sim" e "Não" para 1 e 0
    mapeamento_sim_nao = {"Sim": 1, "Não": 0}
    valores_mapeados = {variavel: mapeamento_sim_nao.get(valor, valor) for variavel, valor in valores.items()}

    # Mapeie as opções do tipo de contrato
    contrato_encoded = {"Contrato de Dois Anos": (1, 0), "Contrato Mensal": (0, 1)}
    Contrato_Dois, Contrato_Mensal = contrato_encoded[valores["Escolha o tipo de contrato:"]]

    # Mapeie as opções do tipo de fatura
    tipo_fatura_encoded = {'Fatura Online': (1, 0, 0, 0), 'Boleto': (0, 1, 0, 0), 'Eletronico': (0, 0, 1, 0), 'Transferência Automatica': (0, 0, 0, 1)}
    fatura_online, forma_pagamento_correio, pagamento_eletronico, transferencia_aut = tipo_fatura_encoded[valores["Qual o tipo de Fatura:"]]

    # Crie o array de entrada
    input_data = np.array([
        [
            valores_mapeados["Idoso:"], valores_mapeados["Possui contrato ativo:"],
            valores["Valor Mensal:"], valores["Valor Total:"],
            valores_mapeados["Selecione o sexo:"], valores_mapeados["Possui conjugue:"],
            valores_mapeados["Possui dependentes:"], valores_mapeados["Possui serviço telefonico:"],
            valores_mapeados["Possui mais de uma linha:"], valores_mapeados["Possui fibra Otica:"],
            valores_mapeados["Possui internet:"], valores_mapeados["Possui segurança online:"],
            valores_mapeados["Possui backup:"], valores_mapeados["Possui proteção dispositivo:"],
            valores_mapeados["Possui suporte tecnico:"], valores_mapeados["Possui stream TV:"],
            valores_mapeados["Possui stream Filmes:"], Contrato_Dois, Contrato_Mensal,
            fatura_online, forma_pagamento_correio, pagamento_eletronico, transferencia_aut
        ]
    ])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    # Exibir o resultado da previsão
    if prediction == 0:
        result = "Cliente provavelmente não vai cancelar o serviço."
    else:
        result = "Cliente provavelmente vai cancelar o serviço."
    
    st.subheader("Resultado da Previsão:")
    st.write(result)
