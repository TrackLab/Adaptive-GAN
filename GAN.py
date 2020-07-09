import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time, os, cv2
from tqdm import tqdm

if not os.path.exists('dataset/'):
    print("The folder 'dataset' does not exist. It has now been created. Please fill it with images to train on.")
    os.makedirs('dataset/')
    exit()

def make_nparray_from_images():
    images = []
    counter = 0

    # THIS CHECKS IF YOUR DATASET IS RGB OR GRAYSCALE
    for image in os.listdir("dataset/"):
        try:
            img = cv2.imread("dataset/"+image)
            b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
            if (b==g).all() and (b==r).all(): 
                color_mode = "GRAYSCALE"
            else:
                color_mode = "RGB"
            break
        except:
            pass
    color_mode = color_mode # YOU CAN OVERWRITE color_mode TO "RGB" OR "GRAYSCALE" MANUALLY IN CASE THIS CODE ABOVE DOES NOT CHOOSE THE CORRECT COLORSPACE FOR YOUR DATASET
    
    # CREATING THE DATASET AND CONVERTING THE COLORSPACE IF NECESSARY
    for image in tqdm(os.listdir("dataset/"), desc=f"Preprocessing dataset"):
        try:
            if color_mode == "GRAYSCALE":
                img = cv2.imread("dataset/"+image, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread("dataset/"+image)
            img = cv2.resize(img, (112, 112))
            images.append(np.array(img))
            counter += 1
        except:
            continue
    
    if counter < 1000:
        print("Dataset has less than 1000 images, which may result in low quality results.")
        print("Try to collect more images if you can. Training will run anyway.")

    images = np.array(images)
    print('Dataset shape:', images.shape)
    return images, color_mode

# DATASET CREATION, NORMALIZATION AND BATCH SPLITTING
train_images, color_mode = make_nparray_from_images()
train_images = (train_images-127.5)/127.5
buffer_size = train_images.shape[0]
batch_size = 32
generator_lr = 1e-4
discriminator_lr = 1e-3
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batch_size, drop_remainder=True)

print("Using color mode", color_mode)

if len(train_images) < batch_size:
    print(f"Amount of train images is less than batch size. {len(train_images)} images < {batch_size} batch size")
    print("Please adjust the batch size to atleast 1 and max the amount of images. Otherwise training is not possible")
    exit()

# TO SET THE INPUT SHAPE FOR GRAYSCALE OR RGB
try:
    # RGB (IMAGE_SIZE, IMAGE_SIZE, 3) 3 being the 3 color channels
    disc_input_shape = (train_images.shape[1], train_images.shape[2], train_images.shape[3])
except:
    # GRAYSCALE (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE) The color channel number is 1, its set by default
    disc_input_shape = (train_images.shape[1], train_images.shape[2], 1)
print("Discriminator Input Shape:", disc_input_shape)

# NETWORK SETUP
def make_discriminator():
    model = tf.keras.Sequential(name="Discriminator")
    model.add(tf.keras.layers.Conv2D(6, (16, 16), padding='same', input_shape=disc_input_shape))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.Conv2D(6, (8, 8), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.Conv2D(6, (4, 4), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    print("Discriminator Summary")
    print(model.summary())
    return model

def make_generator(color_mode):
    layers = tf.keras.layers
    model = tf.keras.Sequential(name="Generator")
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 28, 28, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 56, 56, 16)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 112, 112, 3)

    # THIS ADDS A LAST LAYER TO KEEP THE IMAGE SIZE BUT REDUCE THE COLOR SPACE TO GRAYSCALE IF GRAYSCALE IS BEING USED
    if color_mode == "GRAYSCALE":
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 112, 112, 1)

    print(model.summary())
    return model

# LOSS SETUP
def GeneratorLoss(fake_discrimination):
    fake_predictions = tf.sigmoid(fake_discrimination)
    fake_loss = tf.losses.binary_crossentropy(tf.ones_like(fake_predictions), fake_predictions) # Calculating the loss of the fake prediction
    return fake_loss

def DiscriminatorLoss(real_discrimination, fake_discrimination):
    real_predictions = tf.sigmoid(real_discrimination) # Activation value of the discriminators prediction on the real image
    fake_predictions = tf.sigmoid(fake_discrimination) # Activation value of the discriminators prediction on the fake image
    real_loss = tf.losses.binary_crossentropy(tf.ones_like(real_predictions), real_predictions) # Calculating the loss of the real prediction
    fake_loss = tf.losses.binary_crossentropy(tf.zeros_like(fake_predictions), fake_predictions) # Calculation the loss of the fake prediction
    return fake_loss + real_loss

# OPTIMIZER SETUP
generator_optimizer = tf.optimizers.Adam(generator_lr)
discriminator_optimizer = tf.optimizers.Adam(discriminator_lr)

# CREATING THE NETWORKS
discriminator = make_discriminator()
generator = make_generator(color_mode)

# AUTOMATIC FOLDER CREATION
if not os.path.exists('train_output_images/'):
    os.makedirs('train_output_images/')
if not os.path.exists('saved_models/'):
    os.makedirs('saved_models/')

train_step_counter = 0 # Do not change this number!
epochs = 2000 # How many epochs to train on?
train_images_interval = 100 # Change this number to anything from 1 to whatever. If it is set to 100, it will save a image after every 100 training steps
save_model_after_epochs = 250 # After how many epochs should the model be saved again during training?

# TRAINING
def train_step(images):

    global train_step_counter
    global train_images_interval
    fin = np.random.randn(batch_size, 100,).astype("float32")

    with tf.GradientTape() as GeneratorTape, tf.GradientTape() as DiscriminatorTape:

        generated_images = generator(fin)

        # SAVING IMAGES DURING TRAINING
        if train_step_counter % train_images_interval == 0:
            plt.axis('off')
            image = (np.array(generated_images[-1]) * 255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image)
            plt.savefig(f'train_output_images/{train_step_counter}.png', dpi=500)
            plt.clf()
        
        real_discrimination = discriminator(images)           # The discriminators rating for the real images
        fake_discrimination = discriminator(generated_images) # The discriminators rating for the fake images

        generator_loss     = GeneratorLoss(fake_discrimination) # To compute how badly the generator did to fool the discriminator
        discriminator_loss = DiscriminatorLoss(real_discrimination, fake_discrimination) # To compute how badly the discriminator actually performed to find the fake image

        generator_gradients = GeneratorTape.gradient(generator_loss, generator.trainable_variables) # Computing how the generator should be tweaked, ignoring the non trainable variables
        discriminator_gradients = DiscriminatorTape.gradient(discriminator_loss, discriminator.trainable_variables) # Computing how the discriminator should be tweaked, ignoring the non trainable variables

        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables)) # Applying the optimizations with our chosen optimizer, only applying it on trainable variables
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables)) # Applying the optimizations with our chosen optimizer, only applying it on trainable variables
        
        train_step_counter += 1
        return np.mean(generator_loss), np.mean(discriminator_loss)

def train(dataset, epochs=2000):
    for epoch in range(epochs):
        print('Epoch:', epoch+1, "/", epochs)
        for images in dataset:
            images = tf.cast(images, tf.dtypes.float32)
            gen_loss, disc_loss = train_step(images)
            print("Generator Loss:", gen_loss, "Discriminator Loss:", disc_loss)

        if epoch % save_model_after_epochs == 0:
            generator.save(f'saved_models/gen_model_{epoch+1}.h5')


train(train_dataset, epochs)
print("Saving final model")
generator.save(f'saved_models/gen_final_{epochs}.h5')

# THE FOLLOWING CODE CHECKS IF THE MODEL CAN BE LOADED AND USED CORRECTLY
del generator
generator = tf.keras.models.load_model(f'saved_models/gen_final_{epochs}.h5')
fin = np.random.randn(batch_size, 100,).astype("float32")
generated_images = generator(fin)
plt.axis('off')
plt.imshow(cv2.cvtColor((np.array(generated_images[-1]) * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
