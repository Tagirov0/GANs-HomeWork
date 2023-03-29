# MAT and SimSwap

#### Попробовал SimSwap и MAT, обученный на датасете FFHQ на изображение после кропа получилось следующие:
<img src="https://github.com/Tagirov0/GANs-HomeWork/blob/main/inpainting%20%26%20swap/images/test1.png" width=80% height=60%>

#### Для улучшения результата пробовал, но не помогло:
 * Подовать на вход в MAT черно-белое изображение, после чего переводить его в цветное с помощью GPEN Colorization
 * После первого прогона, создовать небольшие маски в плохо сгенерированных областях
 
 #### Также попробовал создать меньшего размера, более-менее адекватных результат получается на масках такого размера:
 <img src="https://github.com/Tagirov0/GANs-HomeWork/blob/main/inpainting%20%26%20swap/images/test2.png" width=80% height=60%>

