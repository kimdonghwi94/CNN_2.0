from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
img_width,img_height=34,34
batch_size=64
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        '/home/kim/server/alphabet/',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

# 픽셀 값을 0~1 사이로 정규화합니다.
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width,img_height, 3),padding='same',strides=1))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='same',strides=1))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(26, activation='softmax'))
model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(train_generator, epochs=20)
model.save("Alphabet")
