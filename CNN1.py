#Part-1 Building the  CNN
#Importing the keras libraries and packages
import tensorflow as tf

#building the neural network
classifier = tf.keras.models.Sequential(
        [tf.keras.layers.Conv2D(32,(3,3),input_shape = (64,64,3), activation = 'relu'),
         tf.keras.layers.MaxPooling2D(2,2),
         tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
         tf.keras.layers.MaxPooling2D(2,2),
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(128,activation = 'relu'),
         tf.keras.layers.Dense(1,activation = 'sigmoid')
        ])

#compiling the model
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
 
#fitting the CNN to the images (image augmentation)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25, 
                         validation_data = test_set,
                         validation_steps = 2000)