from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pickle

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001

def mobilenet_model():
    base_model = MobileNetV2(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
                             include_top=False,
                             weights='imagenet')
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def convert_to_hsv(image):
    hsv_image = tf.image.rgb_to_hsv(image)
    return hsv_image

def data_generators(train_dir, val_dir):
    train_datagen = ImageDataGenerator(rescale=1.0/255,
                                       rotation_range=20,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True
                                       )
    
    val_datagen = ImageDataGenerator(rescale=1.0/255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = IMG_SIZE,
        batch_size = BATCH_SIZE,
        class_mode='binary'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    return train_generator, val_generator

train_dir = './cat_eye_disease_training_dataset/blepharitis/'

val_dir = './cat_eye_disease_validation_dataset/blepharitis/'

train_generator, val_generator = data_generators(train_dir, val_dir)

model = mobilenet_model()

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

loss, accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

with open('./result_final/training_history_blepharitis.pkl', 'wb') as f:
    pickle.dump(history.history, f)

model.save('./saved_model_final/model_mobilenetv2_blepharitis.h5')
