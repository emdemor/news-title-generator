# Gerador de Títulos de Notícias com Rede Neural Recorrente (Experimental)

Este projeto consiste em um modelo seq2seq para gerar títulos a partir de um corpo de notícia, utilizando uma rede neural recorrente stacked com células LSTM. Os dados utilizados foram coletados através de webscrapping do site da Folha de São Paulo e estão disponíveis no repositório [News of the Brazilian Newspaper](https://github.com/emdemor/News-of-the-Brazilian-Newspaper).

O modelo utiliza um tokenizador BPEmb para dividir o texto em subpalavras e um modelo Word2Vec pré-treinado para representação de palavras. Essas representações são então alimentadas na rede neural, que é treinada para prever o título correto para cada notícia.

O modelo é capaz de aprender a sintetizar informações complexas e apresentá-las de forma resumida e clara. Além disso, a rede neural recorrente permite que o modelo capture a relação temporal entre as palavras, possibilitando uma melhor compreensão do contexto da notícia.

Vale ressaltar que este é um modelo experimental e que, futuramente, uma nova versão mais poderosa será construída utilizando LSTM+Attention e, posteriormente, utilizando Transformers, a fim de aprimorar a performance e a eficácia do modelo.

Como estamos utilizando uma rede neural recorrente, é esperado que o modelo perca eficiência para notícias maiores, uma vez que as informações mais antigas vão sendo diluídas entre os estados ocultos das muitas células LSTM subsequentes. Porém, é importante termos esse modelo configurado para depois inferir a influência do mecanismo de Attention nos modelos mais eficientes.

## Como utilizar o modelo

### Pré-requisitos

- Python 3.7 ou superior
- Bibliotecas Python necessárias (ver `requirements.txt`)

### Instalação

1. Clone este repositório em sua máquina local:

```bash
git clone https://github.com/emdemor/News-of-the-Brazilian-Newspaper.git
```


2. Instale as bibliotecas necessárias:

```bash
pip install -r requirements.txt
```

### Treinamento do modelo

Para treinar o modelo, execute o seguinte comando na linha de comando:

```bash
python train.py
```


### Execução do modelo

Para gerar um título para uma notícia específica, utilize o seguinte comando:

```bash
python predict.py --text "corpo da notícia"
```


Substitua "corpo da notícia" pelo corpo da notícia para a qual você deseja gerar um título. O modelo retornará o título gerado.

## Dados utilizados

Os dados utilizados neste projeto foram obtidos através de webscrapping do site da Folha de São Paulo.

## Créditos

- Este projeto foi criado por Eduardo Messias de Morais ( [email](emdemor415@gmail.com) | [github](https://github.com/emdemor) )
- As notícias foram obtidas por webscraping por Marlesson Santana ( [email](marlessonsa@gmail.com) | [github](https://github.com/marlesson) )
- O tokenizador BPEmb foi criado por [Benjamin Heinzerling e Michael Strube](https://github.com/bheinzerling/bpemb).
- O modelo Word2Vec foi treinado utilizando a biblioteca [gensim](https://github.com/RaRe-Technologies/gensim).
- A rede neural recorrente stacked com células LSTM foi construída utilizando a biblioteca [Keras](https://keras.io/).

