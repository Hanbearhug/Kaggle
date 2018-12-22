# 问题描述
An existential problem for any major website today is how to handle toxic and divisive content. Quora wants to tackle this problem head-on to keep their platform a place where users can feel safe sharing their knowledge with the world.

Quora is a platform that empowers people to learn from each other. On Quora, people can ask questions and connect with others who contribute unique insights and quality answers. A key challenge is to weed out insincere questions -- those founded upon false premises, or that intend to make a statement rather than look for helpful answers.

In this competition, Kagglers will develop models that identify and flag insincere questions. To date, Quora has employed both machine learning and manual review to address this problem. With your help, they can develop more scalable methods to detect toxic and misleading content.

Here's your chance to combat online trolls at scale. Help Quora uphold their policy of “Be Nice, Be Respectful” and continue to be a place for sharing and growing the world’s knowledge.

## 引用库
现在版本的tensorflow已经将keras内嵌，因此可以直接调用了
```
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50,MobileNetV2
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

## Distribution of Labels
Majority of images only appear in training dataset once; this makes this situation a great candiate for a one shot learning simese network. 

![](https://github.com/Hanbearhug/Kaggle/blob/master/DistributionOfClassExcludingNewWhale.png)

## PreProccessing 

Training is performed on images subjected to the following operations:

* Transform to black and white;
* Normalized to 0 mean and unit variance

```
im_arrays = []
labels = []
fs = {} ##dictionary with original size of each photo 
for index, row in tqdm(train.iterrows()):
    im = cv2.imread(os.path.join(train_imgs,row['Image']),0)
    norm_image = cv2.normalize(im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    new_image = cv2.resize(norm_image,(resize,resize))
    new_image = np.reshape(new_image,[resize,resize,1])
    im_arrays.append(new_image)
    labels.append(d[row['Id']])
    fs[row['Image']] = norm_image.shape
train_ims = np.array(im_arrays)
train_labels = np.array(labels)
```

通过enumrate函数来将图片id转换成对应的label是一个pythonic的语法

```
d = {cat: k for k,cat in enumerate(train.Id.unique())}
labels.append(d[row['Id']])
```

通过tqdm函数可以显示进度条

```
for index, row in tqdm(train.iterrows()):
```

cv2.normalize函数用于图片的归一化，这里使用的是平移放缩归一化，除此以外还有四种归一化方式,alpha表示归一化范围的下界，beta表示归一化范围的上界
cv2.resize函数用于图片的比例放缩

```
keras.utils.to_categorical
```
用于将一维的label做onehot处理，其中维数要么明确指定，没有明确指定时使用最大值+1作为维数

```
gen =ImageDataGenerator(zoom_range = 0.2,
                            horizontal_flip = True
                       )
```

tf.keras.preprocessing.image.ImageDataGenerator:Generate minibatches of image data with real-time data augmentation.\
提供了数据输入的功能，其中提供了多种数据增强的操作，如，随机平移，翻转，旋转，剪切，放大缩小，ZCA白化操作，样本标准化操作，同时提供flow方法，可以用于生成图片的存储或者按照batch输入神经网络

```
reduceLROnPlat = ReduceLROnPlateau(monitor='val_top_5_accuracy',
                                      factor = 0.50,
                                      patience = 3,
                                      verbose = 1, 
                                      mode = 'max', 
                                      min_delta = .001,
                                      min_lr = 1e-5
                                  )

earlystop = EarlyStopping(monitor='val_top_5_accuracy',
                            mode= 'max',
                            patience= 5 )

callbacks = [earlystop, reduceLROnPlat]
```

tf.keras.callbacks.ReduceLROnPlateau:Reduce learning rate when a metric has stopped improving.

Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates. This callback monitors a quantity and if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced.\
提供了调节学习率的方法，包括监控对象，每次调整的幅度，上调或者是下调，最小的学习率下界等等。


tf.keras.callbacks.EarlyStopping:Stop training when a monitored quantity has stopped improving.

提供了停止训练的监控条件，如提升的判别条件，多少次迭代没有提升后停止训练等等。

```
model = ResNet50(input_shape=(resize, resize, 1),
                      weights=None, 
                      classes=num_classes)
```
keras提供了主流架构，同时可以决定权重使用随机初始化还是使用imagenet预训练过的权重，返回值是一个keras的model实例，这个实例既可以通过手动搭建神经网络的前向过程来初始化，也可以通过__init__方法进行初始化层，用call方法完成神经网络的前向传播过程。

```
model.compile(optimizer=Adam(lr = .005), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_5_accuracy])
print(model.summary())
```
compile函数用于神经网路的基础配置，例如反向传播中的优化算法，目标损失函数，分类任务中一般使用交叉熵函数，以及训练过程中的尺度函数（例如准确率等），对于多输出函数可以使用字典对每一个输出规定尺度量，以及样本权重等变量（当样本不平衡时也许有用？）\
summary()函数可以打印神经网络中的架构图以及可训练变量和不可训练变量等信息。

```
epochs = 50
history=model.fit_generator(generator=batches, 
                            steps_per_epoch=batches.n//batch_size, 
                            epochs=epochs, 
                            validation_data=val_batches, 
                            validation_steps=val_batches.n//batch_size,
                            callbacks = callbacks)
```
model实例中的fit_generator方法:Fits the model on data yielded batch-by-batch by a Python generator.

The generator is run in parallel to the model, for efficiency. For instance, this allows you to do real-time data augmentation on images on CPU in parallel to training your model on GPU.

The use of keras.utils.Sequence guarantees the ordering and guarantees the single use of every input per epoch when using use_multiprocessing=True.\

接受一个生成器函数(可以使用next方法迭代)或者一个keras.utils.Sequence对象作为输入，steps_per_epoch表示每一个epoch迭代的次数，一般表示dataset中元素的数量除以batch的大小，验证集（可以带权重），class_weight表示每一个类别附带的权重，指定多进程，是否随机打乱(只在step_per_epoch为None时有意义)，还可以在中断训练后重新指定训练的初始epoch（initial_epoch）

