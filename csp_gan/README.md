## GAN with CSP blocks

### [Wandb](https://wandb.ai/tesareon/CSPGAN?workspace=user-tesareon)

#### Поптыка обучения без регуляризации (не считая batch normalization):
<img src="https://github.com/Tagirov0/GANs-HomeWork/blob/main/csp_gan/test/csp_gan_without_regularization.png" width=100% height=100%>

* Видно, что обучение расходится

#### Попробовал R1 регуляризацию с разными коэффициентами, процесс обучения начал сходится 
<img src="https://github.com/Tagirov0/GANs-HomeWork/blob/main/csp_gan/test/csp_gan_r1_regularization.png" width=100% height=100%>

* Остановился на R1 регуляризации с коэффициентом = 2, так сетка генерировала изображения более лучшего качества 

#### Также пробовал заменить BatchNorm на Dropout, но понял, что затея плохая)
<img src="https://github.com/Tagirov0/GANs-HomeWork/blob/main/csp_gan/test/csp_gan_dropout.png" width=100% height=100%>

### Сетка с Resnet блоками в процессе обучения
