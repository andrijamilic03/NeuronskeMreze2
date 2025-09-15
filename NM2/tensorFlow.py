import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split

# Učitavanje podataka
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Pretprocesiranje
train_images = train_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
test_images = test_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Podela trening skupa na trening i validaciju
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
)

# CNN arhitektura
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Kompilacija i treninga
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(val_images, val_labels))

# Evaluacija na test skupu
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")



# TensorFlow (Keras) koristi apstraktniji pristup kroz Sequential model i funkciju model.fit(),
# što omogućava bržu implementaciju bez brige o detaljima treniranja. Pogodniji je za brzi razvoj
# i manje kompleksne modele, ali manje fleksibilan ako je potrebna potpuna kontrola.