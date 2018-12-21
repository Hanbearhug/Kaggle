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
