<h1 align="center">
    <img alt="GIF" src="https://miro.medium.com/max/1400/1*CniSdF4zewDrajSHwCekSQ.gif">
</h1>

<h2 align="center"> 
	🕹 E se pudessemos identificar minerais remotamente? 🕹
</h2>

<p align="center">
 
 <img src="http://img.shields.io/static/v1?label=STATUS&message=BUILDING&color=GREEN&style=for-the-badge"/>
 <img src="http://img.shields.io/static/v1?label=last_releasure&message=16_out_2022&color=GREEN&style=for-the-badge"/>
 <img src="http://img.shields.io/static/v1?label=license&message=MIT&color=blue&style=for-the-badge"/>
 <img src="http://img.shields.io/static/v1?label=LANGUAGES&message=1&color=red&style=for-the-badge"/>
 
</p>

Tabela de conteúdos
=================
<!--ts-->
   * [Sobre o projeto](#Sobre)
   * [Objetivos](#Objetivos)
   * [Demandas](#Demandas)
   * [Desafios](#Desafios)
   * [Primeira Etapa](#Firststep)
   * [Primeiro Estudo](#Firststudy)
<!--te-->

# 🔍 Sobre o projeto

Por meio de recursos e tecnologias das áreas de Computação, Matemática, Ciências da Natureza e Geologia, o objetivo do projeto é inovar e progredir dentro do segmento de inteligência arfiticial, processamento de imagens e visão computacional para a identificação e classificação remota de minerais e rochas.

Todos os arquivos utilizados no projeto se encontram no link abaixo. <br>
Link: https://drive.google.com/drive/folders/1pHI431Fdt6E6O9dDLIzFAh1tX9t5zS02?usp=sharing (conceder acesso)

# ⭐ Objetivos

A trilha desse projeto é árdua, mas sempre avante! A ideia é relativamente simples, embora trabalhosa, os objetivos que contemplam esse projeto em toda sua plenitude podem ser descritos logo abaixo:

- Identificar e classificar minerais e rochas remotamente por meio de imagens, filmagens ou, até mesmo, em tempo real com um aparelho eletrônico como o celular.
- Identificar e classificar rochas utilizando diagramas com base na porcentagem de minerais presentes.
- Identificar e classificar minerais utilizando diagramas com base na porcentagem de minerais presentes.
- Classificar minerais utilizando fórmula química real com base na porcentagem de minerais presentes.
- Ser capaz de utilizar técnicas além dos espectro eletromagnético do vísivel para classificação de minerais.
- Utilizar técnicas de sensoriamento remoto para identificação e classificação de minerais e rochas.

# ⚙ Demandas

O projeto é pioneiro, ou seja, há muito caminho para caminhar. Atualmente, existem algumas demandas para a conclusão da etapa vigente que serão atualizadas conforme o progresso, dentre as demandas, são apresentadas a seguir:

- Modelo de aprendizado mais eficiente.
- Obtenção de mais imagens.
- Estratégia para organizar o Dataset de imagens.

# 🥊 Desafios

**Primeira etapa**, existia o desafio de lidar com a obtenção de imagens que já foi solucionado com Webscraping, que coleta imagens e informações como nome e URL, entretanto, ainda existe a dúvida sobre a veracidade das imagens obtidas no site durante o Webscraping, gerando um desafio na hora de organizar o Dataset de imagens. Durante a obtenção, o algoritmo pode adquirir imagens de paisagens ou de conteúdo irrelevante para o projeto e, além disso, ele pode adquirir imagens de outros minerais que não estão no objeto de estudo atual, trazendo trabalho manual para organizar e selecionar as imagens de interesse.

# Primeira etapa

A primeira etapa será resumida em: ensinar a maquina a diferenciar 2 tipos de minerais, são eles: Quartzo e Pirita, utilizando somente imagens respectivas de cada mineral e, criar uma interface possibilita o usuário de subir imagens e receber uma mensagem da predição do modelo.

## Primeiro estudo:

O repositório referente ao primeiro estudo se encontra no repositório "Estudo_01.ipynb". <br>
Link: https://github.com/thisissamuca/EIAPI/blob/main/Estudo_01

### Estratégias:

- Utilizaremos um Dataset pequeno de **1000** imagens, sendo dividas em 400 para treino e 100 para validação em cada respectivo mineral;
- Importaremos um modelo pré-treinado "MobileNetV2";
- Analisaremos o aprendizado;

### Passos:

### 1) Instalando bibliotecas necessárias:

```Python

pip install tensorflow
p install keras
pip install Keras-Preprocessing

```

### 2) Importando todas as possíveis bibliotecas para utilização no estudo:

```Python

import tensorflow as tf 
import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import os
import random
import shutil
import PIL
import PIL.Image
from random import randint
from random import sample
from keras import preprocessing 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing import image
#from keras.preprocessing.image import load_img
from urllib.request import Request, urlopen
from google.colab import drive

```

### 3) Montando o Google Drive in colab:

```Python

drive.mount('/content/drive')

```

`Mounted at /content/drive` 

### 4) Criando diretórios dos arquivos que serão utilizados no estudo:

```Python

base_dir = '/content/drive/MyDrive/EIAPI/datasets/dataset_ML_02'
test_dir = "./test/"
train_dir = "./train/"

# Diretório de treinamento

dataset_train_dir = os.path.join(base_dir, 'train')
dataset_train_quartz = os.path.join(dataset_train_dir, 'quartz')
dataset_train_pyrite = os.path.join(dataset_train_dir, 'pyrite')

# Diretorio de teste

dataset_test_dir = os.path.join(base_dir, 'test')
dataset_test_quartz = os.path.join(dataset_test_dir, 'quartz')
dataset_test_pyrite = os.path.join(dataset_test_dir, 'pyrite')

# Quantidade de figuras treinamento

dataset_train_quartz_len = len(os.listdir(os.path.join(dataset_train_dir, 'quartz')))
dataset_train_pyrite_len = len(os.listdir(os.path.join(dataset_train_dir, 'pyrite')))

# Quantidade de figuras teste

dataset_test_quartz_len = len(os.listdir(os.path.join(dataset_test_dir, 'quartz')))
dataset_test_pyrite_len = len(os.listdir(os.path.join(dataset_test_dir, 'pyrite')))

print('Train quartz: %s' % dataset_train_quartz_len)
print('Validation quartz: %s' % dataset_test_quartz_len)
print('---')
print('Train pyrite: %s' % dataset_train_pyrite_len)
print('Validation pyrite: %s' % dataset_test_pyrite_len)
print('---')

```
`Train quartz: 400` <br>
`Validation quartz: 100` <br>
`---` <br>
`Train fluorite: 400` <br>
`Validation fluorite: 100` <br>
`---` 

### 5) Configurando parâmetros básicos; Fazendo o pré-processamento usando o Keras e separando parte do diretório para validação:

```Python

image_width = 300
image_height = 300
image_color_channel = 3
image_color_channel_size = 255
image_size = (image_width, image_height)
image_shape = image_size + (image_color_channel, )

batch_size = 32
epochs = 10
learning_rate = 0.0001

class_names = ['quartz', 'pyrite']

dataset_train = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_train_dir,
    seed=123,
    image_size = image_size,
    batch_size = batch_size,
    shuffle = True
)

# Dataset Test

dataset_test = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_test_dir,
    seed=123,
    image_size = image_size,
    batch_size = batch_size,
    shuffle = True
)

# Dataset Cardinality

dataset_test_cardinality = tf.data.experimental.cardinality(dataset_test)
dataset_test_batches = dataset_test_cardinality // 5

dataset_validation = dataset_test.take(dataset_test_batches)
dataset_test = dataset_test.skip(dataset_test_batches)

print('---')
print('Validation Dataset Cardinality: %d' % tf.data.experimental.cardinality(dataset_test))
print('Test Dataset Cardinality: %d' % tf.data.experimental.cardinality(dataset_validation))

class_names = dataset_train.class_names

print(class_names)

```

`Found 800 files belonging to 2 classes.` <br>
`Found 200 files belonging to 2 classes.` <br>
`---` <br>
`Validation Dataset Cardinality: 6` <br>
`Test Dataset Cardinality: 1` <br>
`['pyrite', 'quartz']`

### 6) Criando função de plotagem para verificar as imagens dos diretórios:

```Python

def plot_dataset(dataset):

    plt.gcf().clear()
    plt.figure(figsize = (15, 15))

    for features, labels in dataset.take(1):

        for i in range(9):

            plt.subplot(3, 3, i + 1)
            plt.axis('off')

            plt.imshow(features[i].numpy().astype('uint8'))
            plt.title(class_names[labels[i]])

#plot_dataset(dataset_train)
#plot_dataset(dataset_validation)
#plot_dataset(dataset_test)

```

### 7) Vamos importar um modelo neural pré-treinado "MobileNetV2". Vamos desativar a alteração dos layers e usar o pré-processamento mobilenet:

```Python

# Importando um modelo neural

model_transfer_learning = tf.keras.applications.MobileNetV2(input_shape = image_shape, include_top = False, weights = 'imagenet')
model_transfer_learning.trainable = False
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
rescaling = tf.keras.layers.experimental.preprocessing.Rescaling(1. / (image_color_channel_size / 2.), offset = -1, input_shape = image_shape)
#model_transfer_learning.summary()

```


### 8) Agora vamos otimizar o modelo com alguns parâmetros:

```Python

autotune = tf.data.AUTOTUNE

dataset_train = dataset_train.prefetch(buffer_size = autotune)
dataset_train = dataset_train.prefetch(buffer_size = autotune)
dataset_test = dataset_train.prefetch(buffer_size = autotune)

```

### 9) Vamos usar data augmentation para gerar imagens artificiais com operações como Flip, Rotation e Zoom para aumentar o Dataset e diminuir o Overfit:

```Python

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2)
    
])

```

### 10) Vamos criar uma função para fazer uma representação gráfica do aprendizado:

```Python

def plot_model():

  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  plt.figure(figsize=(8, 8))
  plt.subplot(2, 1, 1)
  plt.plot(acc, label='Training Accuracy')
  plt.plot(val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.ylabel('Accuracy')
  plt.ylim([min(plt.ylim()),1])
  plt.title('Training and Validation Accuracy')

  plt.subplot(2, 1, 2)
  plt.plot(loss, label='Training Loss')
  plt.plot(val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.ylabel('Cross Entropy')
  plt.ylim([0,1.0])
  plt.title('Training and Validation Loss')
  plt.xlabel('epoch')
  
  plt.show()
  
```

### 11) Faremos os últimos ajustes no modelo antes de começar a treina-lo.

Utilizaremos o compilador Adam e utilizaremos um dropout de 20%, isto é, 20% dos dados aprendidos durante o treinamento, aleatoriamente, serão desconsiderados, isso para diminuir ainda mais o Overfit.

```Python

# Definindo as últimas camadas do modelo

model = tf.keras.models.Sequential([
    rescaling,
    data_augmentation,
    model_transfer_learning,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

# Compilando o modelo com Adam

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics = ['accuracy']
)

#model.summary()

# Treinando o modelo

history = model.fit(
    dataset_train,
    validation_data = dataset_validation,
    epochs = epochs
)

# Avaliando o modelo

plot_model()

dataset_test_loss, dataset_test_accuracy = model.evaluate(dataset_test)

print('Dataset Test Loss:     %s' % dataset_test_loss)
print('Dataset Test Accuracy: %s' % dataset_test_accuracy)

features, labels = dataset_test.as_numpy_iterator().next()
predictions = model.predict_on_batch(features).flatten()
predictions = tf.where(predictions < 0.5, 0, 1)

print('Labels:      %s' % labels)
print('Predictions: %s' % predictions.numpy())

```

`Dataset Test Loss:     0.5312994122505188` <br>
`Dataset Test Accuracy: 0.7787500023841858` <br>
`Labels:      [0 1 1 0 1 1 1 0 1 0 0 1 0 1 1 0 0 1 0 1 1 0 0 0 0 0 1 0 0 0 0 1]` <br>
`Predictions: [0 0 1 0 1 1 1 0 1 0 0 1 0 0 1 0 0 1 0 1 0 1 0 0 1 1 1 0 0 0 1 1]` <br>

### 12) Criando função de predict image:

```Python

def predict(image_file):

    image = tf.keras.preprocessing.image.load_img(image_file, target_size = image_size)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, 0)

    prediction = model.predict(image)[0][0]

    print(f'{round(prediction, 3) * 100}% Quartz | {round(1 - prediction, 1) * 100}% Pyrite')
    
#predict('image_file.jpg')

```

### 13) Melhorando o modelo:

Vamos ativar alterações nos layers internos do modelo:

```Python

model_transfer_learning.trainable = True

# Vamos dar uma olhada para ver quantas camadas estão no modelo base

print("Number of layers in the base model: ", len(model_transfer_learning.layers))

# Ajuste fino desta camada em diante

fine_tune_at = 100

# Congele todas as camadas antes da camada `fine_tune_at`

for layer in model_transfer_learning.layers[:fine_tune_at]:

  layer.trainable = False
  
len(model.trainable_variables)

```

### 14) Re-compilando o modelo utilizando RMSprop com uma taxa de aprendizado diminuída:

```Python

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate/10),
              metrics=['accuracy'])
              
#model.summary()

```

### 15) Re-treinando o modelo:

```Python

fine_tune_epochs = 10

total_epochs =  epochs + fine_tune_epochs

history_fine = model.fit(dataset_train,
                         epochs = total_epochs,
                         initial_epoch = history.epoch[-1],
                         validation_data = dataset_validation)
                         
plot_model()

loss, accuracy = model.evaluate(dataset_test)

print('Test accuracy :', accuracy)

features, labels = dataset_test.as_numpy_iterator().next()
predictions = model.predict_on_batch(features).flatten()
predictions = tf.where(predictions < 0.5, 0, 1)

print('Labels:      %s' % labels)
print('Predictions: %s' % predictions.numpy())

``` 

`Test accuracy : 0.9024999737739563` <br>
`Labels:      [1 0 1 1 0 1 0 0 0 1 1 1 1 0 0 0 1 0 1 0 0 1 1 0 0 0 0 0 0 0 1 0]` <br>
`Predictions: [1 0 1 1 0 1 0 0 0 1 1 1 1 0 0 0 1 0 1 1 0 1 1 0 0 0 0 1 0 0 1 0]` <br>

### 16) Vamos agora salvar o modelo:

```Python

model.save('/content/drive/MyDrive/EIAPI/models/model_01')
#model.save_weights('model_02')

``` 

### Conclusões:

- O modelo conseguiu chegar em 77% de precisão com o re-treino, entretanto, embora parece uma taxa boa, posso exigir um pouco mais do modelo, pois se trata de dois minerais bem distintos e distingui-los é uma tarefa tanto fácil, então, podemos esperar por pelo menos 95% de precisão nesse caso.
- Suponho que o fator que prejudicou o aprendizado foi as imagens no Dataset, como dito anteriormente, existe uma falha na veracidade das imabens obitidas pelo algoritmo, isto é, imagens que não se trata do objeto de estudo foram interpretadas pelo modelo, gerando erros, uma opção é selecionar as imagens e conferir a integridade delas para que isso não ocorra.

## Segundo estudo:

Dividiremos esse estudo em dois casos: "Utilizando o modelo do estudo passado" e "Não utilizando o modelo do estudo passado". <br>
Link primeiro caso: https://github.com/thisissamuca/EIAPI/blob/main/Estudo_02.ipynb <br>
Link segundo caso: https://github.com/thisissamuca/EIAPI/blob/main/Estudo_03.ipynb

### Estratégias:

- Utilizaremos um Dataset maior de **1500** imagens, sendo dividas em 600 para treino e 150 para validação em cada respectivo mineral;
- Faremos algumas mudanças no Dataset, removendo imagens que inúteis.
- Utilizaremos dois modelos: o modelo anterior e um modelo pré-treinado "MobileNetV2";
- Analisaremos o aprendizado;

### Primeiro caso:

### Passos:

### 1) Além das bibliotecas anteriores, instalaremos mais uma:

```Python

pip install aspose-words

```

### 2) Além das bibliotecas anteriores, importaremos mais uma:

```Python

import aspose.words as aw

```

### 3) Montaremos um Google Drive in colab:

```Python

drive.mount('/content/drive')

```

### 4) Manteremos os mesmos parâmetros, exceto o número de epochs:

```Python

epochs = 5

```

### 5) Vamos selecionar o diretório com imagens autênticas para o nosso objeto de estudo:

```Python

base_dir = '/content/drive/MyDrive/EIAPI/datasets/dataset_ML_03'
test_dir = "./test/"
train_dir = "./train/"

```

Usaremos o mesmo código do estudo anterior para verificar a quantidade de arquivos no Dataset:

`Train quartz: 600` <br>
`Validation quartz: 150` <br>
`---` <br>
`Train pyrite: 600` <br>
`Validation pyrite: 150` <br>
`---`

### 8) Utilizaremos a mesma otimização e data augmentation, entretanto, vamos modificar alguns números:

```Python

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal_and_vertical'),
    tf.keras.layers.RandomRotation(0.9),
    tf.keras.layers.RandomZoom(0.35)
])

```

### 9) Vamos importar o mesmo modelo usado no estudo anterior, compila-lo e treina-lo novamente:

```Python

model = tf.keras.models.load_model('/content/drive/MyDrive/EIAPI/models/model_01/model_01')

model.compile(
    optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.0001),
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics = ['accuracy']
)

history = model.fit(
    dataset_train,
    validation_data = dataset_validation,
    epochs = epochs,
    callbacks = [early_stopping] 
)

plot_model()

dataset_test_loss, dataset_test_accuracy = model.evaluate(dataset_test)

print('Dataset Test Loss:     %s' % dataset_test_loss)
print('Dataset Test Accuracy: %s' % dataset_test_accuracy)

```

`Dataset Test Loss:     0.06267542392015457` <br>
`Dataset Test Accuracy: 0.9766666889190674`

### 10) Os próximos passos nos auxiliarão para visualizar o desempenho real do modelo.

Vamos selecionar um diretório com imagens que o modelo ainda não teve contato:

```Python

base_dir = '/content/drive/MyDrive/EIAPI/minerals'

dataset_dir = os.path.join(base_dir)

dataset_mineral_dir = os.path.join(dataset_dir, 'pyrite')
#dataset_mineral_dir = os.path.join(dataset_dir, 'quartz')

photos = os.listdir(dataset_mineral_dir)
```

```

# Lista genérica

x = []

```

```Python

# Vamos colocar parte (ou todos) os arquivos na lista "x"

#for i in range(len(photos) - 1):

for i in range(100):

  if re.search(r'Pyrite', photos[i]):

    x.append(photos[i])

  #if re.search(r'Quartz', photos[i]):

    #x.append(photos[i])
    
print(x)

```

`['EAIPI_Pyrite_8339095.jpg', 'EAIPI_Pyrite_8339149.jpg', 'EAIPI_Pyrite_8339164.jpg', 'EAIPI_Pyrite_8339180.jpg', 'EAIPI_Pyrite_1035089.jpg', 'EAIPI_Pyrite_8433319.jpg', 'EAIPI_Pyrite_8433395.jpg', 'EAIPI_Pyrite_8449490.jpg', 'EAIPI_Pyrite_8449777.jpg', 'EAIPI_Pyrite_8465689.jpg', 'EAIPI_Pyrite_8554198.jpg', ...]` 

```Python

# Lista genérica 

y = []

# Vamos fazer predições em cada um dos arquivos

for i in range(len(x) - 1):

  if re.search(r'Quartz', x[i]):

    image = tf.keras.preprocessing.image.load_img((f'/content/drive/MyDrive/EIAPI/minerals/quartz/{x[i]}') , target_size = image_size)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, 0)
    prediction = model.predict(image)[0][0]

    y.append([round(prediction, 3), round(1 - prediction, 3)])

  if re.search(r'Pyrite', x[i]):

    image = tf.keras.preprocessing.image.load_img((f'/content/drive/MyDrive/EIAPI/minerals/pyrite/{x[i]}') , target_size = image_size)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, 0)
    prediction = model.predict(image)[0][0]

    y.append([round(prediction, 3), round(1 - prediction, 3)])

print(y)

```

`[[0.0, 1.0], [0.0, 1.0], [0.001, 0.999], [0.003, 0.997], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], ...]` 

Vamos plotar todas as imagens em "x" com as predições em "y", salvar cada figura em um caminho de diretório, juntar todas em um PDF e salvar esse arquivo em um caminho de diretório.

```Python

doc = aw.Document()
builder = aw.DocumentBuilder(doc)

for i in range(len(y) - 1):

  if re.search(r'Quartz', x[i]):

    img = mpimg.imread(f'/content/drive/MyDrive/EIAPI/minerals/quartz/{x[i]}')

    fig = plt.figure(figsize = (12, 5))

    ax = fig.add_subplot(1, 2, 1)

    imgplot = plt.imshow(img)

    ax.set_title(f'{x[i]}')

    ax = fig.add_subplot(1, 2, 2)

    if y[i][0] < .5:

      plt.bar(class_names, y[i], width = 0.5, color ='maroon')

    else:

      plt.bar(class_names, y[i], width = 0.5)

    ax.set_title('Prediction')

  if re.search(r'Pyrite', x[i]):

    img = mpimg.imread(f'/content/drive/MyDrive/EIAPI/minerals/pyrite/{x[i]}')

    fig = plt.figure(figsize = (12, 5))

    ax = fig.add_subplot(1, 2, 1)

    imgplot = plt.imshow(img)

    ax.set_title(f'{x[i]}')

    ax = fig.add_subplot(1, 2, 2)

    if y[i][0] < .5:

      plt.bar(class_names, y[i], width = 0.5, color ='maroon')

    else:

      plt.bar(class_names, y[i], width = 0.5)

    ax.set_title('Prediction')

  plt.savefig(f'/content/drive/MyDrive/EIAPI/models/model_02/plots/plot_model_02_{i}.png')

  builder.insert_image(f'/content/drive/MyDrive/EIAPI/models/model_02/plots/plot_model_02_{i}.png')

doc.save('/content/drive/MyDrive/EIAPI/models/model_02/PDF/model_02_performance.pdf')

```

### 14) Vamos salvar o modelo

```Python

model.save('/content/drive/MyDrive/EIAPI/models/model_04')

```

Vale lembrar que, caso queira acessar o PDF para visualizar o desempenho do modelo, basta acessar o link abaixo. <br>
Link: https://drive.google.com/file/d/1BGjX15VAo_gSoLf_KtS0S2uac8W4DscH/view?usp=sharing

### Conclusões

- Embora a precisão do modelo seja de 97%, isso não necessariamente indica que o modelo está corretamente aparelhado com a realidade, pois, ao analisar as predições, percebi que há uma "generalização" nas imagens, ou seja, em imagens que apresentam parte Quartzo (silicato ou outro mineral semelhante ao Quartzo) o modelo parece "ignorar" essas feições e considerar ou Quartzo ou Pirita. Esse tipo de comportamento não é interessante, pois não coindiz com a realidade. Veja algumas representações. Veja que o modelo poderia ter considerado um pouco mais a possibilidade de ser um Quartzo, mas não é isso que ocorre:

<h3 align="center">
    <img alt="img" src="https://cdn.discordapp.com/attachments/947611804945776691/1031403451407675442/WhatsApp_Image_2022-10-17_at_00.07.52.jpeg">
</h3>

<h3 align="center">
    <img alt="img" src="https://cdn.discordapp.com/attachments/947611804945776691/1031403451629973564/WhatsApp_Image_2022-10-17_at_00.07.53.jpeg">
</h3>

### Segundo caso (Não utilizando o modelo do primeiro estudo):

### Passos:

### 1) Utilizaremos a maioria das linhas de código do caso anterior.

Vamos usar outro diretório:

```Python

base_dir = '/content/drive/MyDrive/EIAPI/datasets/dataset_ML_03'
test_dir = "./test/"
train_dir = "./train/"

```

Usaremos o mesmo código do estudo anterior para verificar a quantidade de arquivos no Dataset:

`Train quartz: 600` <br>
`Validation quartz: 150` <br>
`---` <br>
`Train pyrite: 600` <br>
`Validation pyrite: 150` <br>
`---`

### 2) Utilizaremos os mesmos parâmetros do estudo anterior. Iremos modificar apenas o número de epochs:

```Python

epochs = 15

```

### 3) Utilizaremos o mesmo código do estudo anterior para pré-processar os Datasets e selecionar arquivos para validação:

`Found 1200 files belonging to 2 classes.` <br>
`Found 300 files belonging to 2 classes.` <br>
`---` <br>
`Validation Dataset Cardinality: 8` <br>
`Test Dataset Cardinality: 2` <br>

### 4) Utilizaremos os mesmos códigos para importar, otimizar e preparar o modelo pré-treinado MobilenetV2

### 5) Vamos utilziar um data augmentation:

```Python

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal_and_vertical'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2)
])

```

### 6) Vamos treinar o modelo assim como já fizemos, utilizando os mesmos parâmetros do primeiro estudo. Também iremos re-treina-lo usando os mesmos parâmetros, exceto o número de epochs:

```Python

fine_tune_epochs = 15

```

### 7) Vamos verificar nosso modelo:

```Python

dataset_test_loss, dataset_test_accuracy = model.evaluate(dataset_test)

print('Dataset Test Loss:     %s' % dataset_test_loss)
print('Dataset Test Accuracy: %s' % dataset_test_accuracy)

```

`Dataset Test Loss:     0.1056627556681633` <br>
`Dataset Test Accuracy: 0.9591666460037231`

### 8) Vamos utilizar os mesmos códigos para gerar plotagens.

### 9) Vamos salvar o modelo:

```Python

model.save('/content/drive/MyDrive/EIAPI/models/model_03')

```

Vale lembrar que, caso queira acessar o PDF para visualizar o desempenho do modelo, basta acessar o link abaixo.
Link: https://drive.google.com/file/d/1bRIDQQXz6ISA0YboSQne-RFLD7xGpYPZ/view?usp=sharing

### Conclusões

- Embora o modelo tenha 95% de precisão (menor em relação ao primeiro caso) ele condiz mais com a realidade. Veja algumas representações:

<h3 align="center">
    <img alt="img" src="https://cdn.discordapp.com/attachments/947611804945776691/1031408708950962196/WhatsApp_Image_2022-10-17_at_00.03.30.jpeg">
</h3>

<h3 align="center">
    <img alt="img" src="https://cdn.discordapp.com/attachments/947611804945776691/1031408709357817937/WhatsApp_Image_2022-10-17_at_00.03.29.jpeg">
</h3>

- Imagens que podem apresentar ou parecer com Quartzo (silicato ou outro mineral semelhante ao Quartzo) o modelo não "ignora" essas feições e interpreta com a possível possibilidade de ser ou parecer com Quartzo, pois bem, isso é mais próximo da realidade em relação ao caso anterior que ignorava essas feições.
