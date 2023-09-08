from flask import Flask, request, jsonify
from textblob import TextBlob
import pickle
import numpy as np
from flask_basicauth import BasicAuth
import os

url_arquivo = "../../models/model_random_forest_novexus_churn.pkl"
url_scaler = "../../models/scaler_random_forest_novexus_churn.pkl"

colunas = ['Idoso', 'Contrato_Ativo', 'Valor_Mensal', 'Valor_Total', 'Genero',
            'Conjuge', 'Dependentes', 'Servico_Telefone', 'Mult_Linhas',
            'Servico_Internet_Fibra Otica', 'Servico_Internet', 'Seguranca_Online',
            'Backup_Online', 'Protecao_Disp', 'Suporte_Tecnico', 'Stream_Tv',
            'Stream_Filmes', 'Contrato_Dois anos', 'Contrato_Mensal',
            'Fatura_Online', 'Forma_Pagamento_Correio',
            'Forma_Pagamento_Pag. Eletronico', 'Forma_Pagamento_Transf. Aut.']

with open(url_arquivo, 'rb') as file:
    classificador_random = pickle.load(file)

with open(url_scaler, 'rb') as file:
    scaler = pickle.load(file)

    
app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')

BasicAuth = BasicAuth(app=app)

@app.route('/')
def home():
    return "Minha primeira API."

@app.route('/sentimento/<frase>')
@BasicAuth.required
def sentimento(frase):
    tb = TextBlob(frase)
    traducao = tb.translate(from_lang='pt_br', to='en')
    polaridade = traducao.sentiment.polarity
    return "Polaridade {}".format(polaridade)

@app.route('/novexus_churn/', methods=['post'])
@BasicAuth.required
def analisa_churn():
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    dados_input = np.array(dados_input).reshape(1,-1)

    dados_scaler = scaler.transform(dados_input)
    previsao = classificador_random.predict(dados_scaler)[0]
    # return jsonify(previsao=previsao[0])

    if previsao == 0:
        return 'Previsão de NÃO continuar como cliente!'
    else:
        return 'Previsão de CONTINUAR como cliente!' 

app.run(debug=True, host='0.0.0.0')


