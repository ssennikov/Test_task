# Face Segmentation

<img src="https://user-images.githubusercontent.com/109667895/206320957-07560012-51a4-4575-b167-6637db68595f.jpg" width="400"/>
<img src="https://user-images.githubusercontent.com/109667895/206535524-0e4e7e83-c4de-4e3d-a16f-a9fd16bd7cd7.gif" width="400"/>

Для обучения модели сегментации лиц была использована часть датасета<a href='https://github.com/switchablenorms/CelebAMask-HQ'> CelebAMASK-HQ</a> (2000изображений).

Модель обучалась 10 эпох с lr=0,0001 и batch_size = 4(обучал на своей видеокарте), 2000 изображений использовал для тренировки и 420 для валидации.

1. Распакуйте архив в папке data

2. Воспользуйтесь написанным скриптом для преобразования датасета preprocess_dataset.py

3. Для обучения модели запустите файл main.py

4. Для получения изображений с масками закиньте тестовые фото в папку test и запустите файл predict.py

5. const.py файл с путями

Ссылка на архив с весами модели DeepLab https://drive.google.com/file/d/1dqMPKW9F7WUc45aiCH_jFgeyZxhthJmv/view?usp=share_link

В качестве финальной модели была использованна модель DeepLabV3 с головой ResNet50, а также попоробовал Unet

c головой ResNet53.

<table>
  <tr><th>model</th><th>Dice</th><th>IoU</th><th>Loss</th></tr>
  <tr><th>DeepLabV3</th><th>0,97</th><th>0,95</th><th>0,039</th></tr>
  <tr><th>UNet</th><th>0,96</th><th>0,93</th><th>0,053</th></tr>
</table>

Несмотря на то что DeepLab обучалась дольше UNet в 2 раза и сегма на тестовый изображений на первый взгляд практически не отличалась, когда стал
сравнивать фотографии лиц с перекрытиями DeepLab оказался лучше.
Так что для дальнейших экспериментов с аугументациями был сделан выбор в пользу DeepLab.

<table>
<tr>
   <td>DeepLabV3<th><img src="https://user-images.githubusercontent.com/109667895/206318901-58a6433b-18d3-42d6-ae7a-6d19900d97f8.jpg" width="400"/></th></td>
   <td>UNet<th><img src="https://user-images.githubusercontent.com/109667895/206318961-c858cd09-04af-4087-9867-838f4ba95f97.jpg" width="400"/></th></td>
</tr>
</table>

<table>
<tr>
   <td>DeepLabV3<th><img src="https://user-images.githubusercontent.com/109667895/206315552-84d667d9-1ecf-4e4d-9d57-d6cdc010a346.jpg" width="400"/></th></td>
   <td>UNet<th><img src="https://user-images.githubusercontent.com/109667895/206315756-b4829525-a65d-437d-9f38-9f9ae8afa81a.jpg" width="400"/></th></td>
</tr>
</table>


Аугументации которые помогли улучшить сегму:

ColorJitter - Произвольно изменяет яркость, контрастность, насыщенность и оттенка изображения
  
HorizontalFlip - Отражает фото по горизонтали
  
CoarseDropout - Добавляет прямоугольники на изображение. После применения этой аугументации модель стала справляться с "перекрытиями"

Примеры с перекрытиями:

<table>
<tr>
   <td><th><img src="https://user-images.githubusercontent.com/109667895/206524818-91ffe7d8-5ff6-4303-b5c8-a1b004f21344.jpg" width="400"/></th></td>
   <td><th><img src="https://user-images.githubusercontent.com/109667895/206524912-f7b1d148-ae4a-4a47-918f-46fdcba66b56.jpg" width="400"/></th></td>
   <td><th><img src="https://user-images.githubusercontent.com/109667895/206529619-ddaa857f-8c27-40d7-9d8e-67c3c02ce2ae.jpg" width="400"/></th></td>
</tr>
</table>



