## GAN with CSP blocks

### [Wandb](https://wandb.ai/tesareon/CSPGAN?workspace=user-tesareon)

#### Поптыка обучения без регуляризации (не считая batch normalization):
<img src="https://github.com/Tagirov0/GANs-HomeWork/blob/main/csp_gan/test/csp_gan_without_regularization.png" width=100% height=100%>

* Видно, что обучение расходится

#### Попробовал R1 регуляризацию с разными коэффициентами, процесс обучения начал сходится 
<img src="https://github.com/Tagirov0/GANs-HomeWork/blob/main/csp_gan/test/csp_gan_r1_regularization.png" width=100% height=100%>

* Остановился на R1 регуляризации с коэффициентом = 2, так сетка генерировала изображения более лучшего качества 

#### GAN с Residual блоками справился с задачей хуже, обучение расходится
<img src="https://github.com/Tagirov0/GANs-HomeWork/blob/main/csp_gan/test/simple_resnet_gan.png" width=100% height=100%>

* Возможно перед каждым upsampling изображения нужно добавлять по еще одному блоку без downsampling каналов

#### Также пробовал заменить BatchNorm на Dropout, но понял, что затея плохая)
<img src="https://github.com/Tagirov0/GANs-HomeWork/blob/main/csp_gan/test/csp_gan_dropout.png" width=100% height=100%>

