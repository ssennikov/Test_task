# Face Segmentation

<img src="https://user-images.githubusercontent.com/109667895/206320957-07560012-51a4-4575-b167-6637db68595f.jpg" width="400"/>
<img src="https://user-images.githubusercontent.com/109667895/208003864-fe09e3f8-3024-4080-a4b0-d04d138a2a3b.gif" width="400"/>

Для обучения модели сегментации лиц была использована часть датасета<a href='https://github.com/switchablenorms/CelebAMask-HQ'> CelebAMASK-HQ</a> (5000изображений).

Модель обучалась 10 эпох с lr=0,0001 и batch_size = 4(обучал на своей видеокарте), 5000 изображений использовал для тренировки и 500 для валидации.

## Подготовка датасета

1. Скачайте уже предобработанный <a href='https://drive.google.com/file/d/152jPI7N1LrGjbe9kWDL6gzpcldj2Nlr2/view?usp=sharing'>датасет</a> и распакуйте архив в папке data

2. Либо скачайте полный датасет и воспользуйтесь написанным скриптом для преобразования датасета preprocess_dataset.py

## Обучение и получение предиктов
1. Для обучения модели запустите файл train.py

2. Для получения изображений с масками закиньте тестовые фото в папку test и запустите файл predict.py

3. const.py файл с путями

## Модель и итоговые метрики

В качестве финальной модели была использованна модель DeepLabV3 с головой ResNet50, а также попробовал Unet

c головой ResNet53.

Ссылка на архив с весами для модели DeepLab https://drive.google.com/file/d/1a1SwY0TG7bMusR2VAZcu9jFiW9SXFiv6/view?usp=share_link

Ссылка на логи в <a href='https://app.clear.ml/projects/46e0c21025b84f1bb7e6ec87c41cc802/experiments/ff7f692d790f4522abbed3d6947f80bc/output/execution'>ClearML</a> 

<table>
  <tr><th>model</th><th>Dice</th><th>IoU</th><th>Loss</th></tr>
  <tr><th>DeepLabV3</th><th>0,98</th><th>0,96</th><th>0,028</th></tr>
  <tr><th>UNet</th><th>0,96</th><th>0,93</th><th>0,053</th></tr>
</table>

Несмотря на то что DeepLab обучалась дольше UNet в 2 раза и сегма на тестовый изображений на первый взгляд практически не отличалась, когда стал
сравнивать фотографии лиц с перекрытиями DeepLab оказался лучше.
Так что для дальнейших экспериментов с аугументациями был сделан выбор в пользу DeepLab.

<table>
<tr>
   <td>DeepLabV3<th><img src="https://user-images.githubusercontent.com/109667895/206318901-58a6433b-18d3-42d6-ae7a-6d19900d97f8.jpg" width="300"/></th></td>
   <td>UNet<th><img src="https://user-images.githubusercontent.com/109667895/206318961-c858cd09-04af-4087-9867-838f4ba95f97.jpg" width="300"/></th></td>
</tr>
</table>

<table>
<tr>
   <td>DeepLabV3<th><img src="https://user-images.githubusercontent.com/109667895/206315552-84d667d9-1ecf-4e4d-9d57-d6cdc010a346.jpg" width="300"/></th></td>
   <td>UNet<th><img src="https://user-images.githubusercontent.com/109667895/206315756-b4829525-a65d-437d-9f38-9f9ae8afa81a.jpg" width="300"/></th></td>
</tr>
</table>


## Аугументации которые помогли улучшить сегму:

1. Из масок лиц были вырезаны очки

2. ColorJitter - Произвольно изменяет яркость, контрастность, насыщенность и оттенка изображения
  
3. CoarseDropout - Добавляет прямоугольники на изображение. После применения этой аугументации модель стала справляться с "перекрытиями"

Примеры с перекрытиями:

<table>
<tr>
   <td><th><img src="https://user-images.githubusercontent.com/109667895/208006644-e1b4ef39-f4f7-4445-b696-aefe2e2860e9.jpg" width="400"/></th></td>
   <td><th><img src="https://user-images.githubusercontent.com/109667895/208006684-6f5579bf-c87d-4da3-a99c-6fe7e48b00de.jpg" width="400"/></th></td>
   <td><th><img src="https://user-images.githubusercontent.com/109667895/208006755-86b77ec9-4af4-4a08-97c5-d46fcc432cfb.jpg" width="400"/></th></td>
   <td><th><img src="https://user-images.githubusercontent.com/109667895/208006786-3f6d9034-cd51-40d1-a373-a19d0907e65c.jpg" width="400"/></th></td>
</tr>
</table>
