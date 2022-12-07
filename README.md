# Face Segmentation

<img src="https://user-images.githubusercontent.com/109667895/206320957-07560012-51a4-4575-b167-6637db68595f.jpg" width="400"/>

Для обучения модели сегментации лиц была использована часть датасета<a href='https://github.com/switchablenorms/CelebAMask-HQ'> CelebAMASK-HQ</a> (2000изображений).

Модель обучалась 10 эпох с lr=0,0001 и batch_size = 4(обучал на своей видеокарте), 2000 изображений использовал для тренировки и 420 для валидации.

1. Распакуйте архив в папке data

2. Воспользуйтесь написанным скриптом для преобразования датасета preprocess_dataset.py

3. Для обучения модели запустите файл main.py

4. Для получения изображений с масками закиньте тестовые фото в папку test и запустите файл predict.py

В качестве модели была использованна модель UNET с головой ResNet53, а также попоробовал DeepLabV3

c головой ResNet50.

<table>
  <tr><th>model</th><th>Dice</th><th>IoU</th><th>Loss</th></tr>
  <tr><th>DeepLabV3</th><th>0,97</th><th>0,95</th><th>0,039</th></tr>
  <tr><th>UNet</th><th>0,96</th><th>0,93</th><th>0,053</th></tr>
</table>

DeepLab обучалась дольше UNet в 2 раза и сегма на тестовый изображений практически не отличалась, 

так что для дальнейших экспериментов с аугументациями был сделан выбор в пользу UNet.

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

  A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3) - Произвольно изменяет яркость, контрастность, насыщенность и оттенка изображения
  
  A.HorizontalFlip() - отражает фото по горизонтали
  
  



