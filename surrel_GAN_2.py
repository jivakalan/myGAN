from keras.layers import Input, BatchNormalization, Conv2D, Conv2DTranspose, Flatten, Dropout, Reshape
from keras.models import Model
from keras.layers.core import Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
#from keras import initializers
import matplotlib.pyplot as plt 
#import sys
#import os
import numpy as np
#from PIL import Image
import glob
#from IPython.display import display
#from IPython.display import Image as _Imgdis
#from scipy import ndimage
#from keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
from skimage.transform import resize

image_urls=glob.glob(r"C:\Users\jkalan\Downloads\Surrealism\*.jpg")

#
#for single_url in image_urls:
#	 surreal_img = load_img(single_url,(32,32))  
#	 x = np.array(surreal_img)  




plt.figure(figsize=(10, 10))
for i in range(20):
    img = plt.imread(image_urls[i])
    plt.subplot(4, 4, i+1)
    plt.imshow(img)
    plt.title(img.shape)
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()		



def load_image(image_urls, size=(32, 32)):
    img = plt.imread(image_urls)
    # crop
    rows, cols = img.shape[:2]
    crop_r, crop_c = 150, 150
    start_row, start_col = (rows - crop_r) // 2, (cols - crop_c) // 2
    end_row, end_col = rows - start_row, cols - start_row
    img = img[start_row:end_row, start_col:end_col, :]
    # resize
    img = resize(img, size)
    return img

def preprocess(x):
    return (x/255)*2-1

def deprocess(x):
    return np.uint8((x+1)/2*255)




#############GAN comes here

def surreal_generator(input_layer):
	generator = Dense(4*4*512,activation='relu')(input_layer)
	generator = BatchNormalization(momentum=0.9)(generator)
	generator = LeakyReLU(alpha=0.1)(generator)
	generator = Reshape((4, 4, 512))(generator)
	
	generator = Conv2DTranspose(256, 4, strides=2, padding='same')(generator)
	generator = BatchNormalization(momentum=0.9)(generator)
	generator = LeakyReLU(alpha=0.1)(generator)
	 
	generator = Conv2DTranspose(128, kernel_size=5, strides=1, padding='same')(generator)
	generator = BatchNormalization(momentum=0.9)(generator)
	generator = LeakyReLU(alpha=0.1)(generator)


	generator = Conv2DTranspose(3, kernel_size=5, strides=1, padding='same')(generator)      
	out = Activation("tanh")(generator)
	
	model = Model(input_layer, out)
	model.summary()
	
	return model, out

def surreal_discriminator(input_layer):
	discriminator =  Conv2D(64, kernel_size=3, strides=1,padding='same')(input_layer)
	discriminator = BatchNormalization(momentum=0.9)(discriminator)
	discriminator = LeakyReLU(alpha=0.1)(discriminator)
	
	discriminator =  Conv2D(128, kernel_size=4, strides=2,padding='same')(discriminator)
	discriminator = BatchNormalization(momentum=0.9)(discriminator)
	discriminator = LeakyReLU(alpha=0.1)(discriminator)	

	discriminator =  Conv2D(256, kernel_size=4, strides=2,padding='same')(discriminator)
	discriminator = BatchNormalization(momentum=0.9)(discriminator)
	discriminator = LeakyReLU(alpha=0.1)(discriminator)

	discriminator = Flatten()(discriminator)
	discriminator = Dropout(0.4)(discriminator)
	out = Dense(1,activation ='sigmoid')(discriminator)

	model = Model(input_layer,out)
	model.summary()

	return model, out	


batch_size = 16
num_batches = int(X_train.shape[0]/batch_size)



def generate_noise(n_samples, noise_dim):
  X = np.random.normal(0, 1, size=(n_samples, noise_dim))
  return X

def show_imgs(batchidx):
  noise = generate_noise(9, 100)
  gen_imgs = generator.predict(noise)

  fig, axs = plt.subplots(3, 3)
  count = 0
  for i in range(3):
    for j in range(3):
      # Dont scale the images back, let keras handle it
      img = image.array_to_img(gen_imgs[count], scale=True)
      axs[i,j].imshow(img)
      axs[i,j].axis('off')
      count += 1
  plt.show()
  plt.close()

##GAN creation
img_input = Input(shape=(32,32,3))
discriminator, disc_out = surreal_discriminator(img_input)
discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

discriminator.trainable = False

noise_input = Input(shape=(100,))
generator, gen_out = surreal_generator(noise_input)

gan_input = Input(shape=(100,))
x = generator(gan_input)
gan_out = discriminator(x)
gan = Model(gan_input, gan_out)
gan.summary()

gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')



##training of GAN
epochs = 20
for epoch in range(epochs):
	cum_d_loss = 0.
	cum_g_loss = 0.
	
	for batch_idx in range(num_batches): 
		 # Get the next set of real images to be used in this iteration
	   images = X_train[batch_idx*batch_size : (batch_idx+1)*batch_size]
	   noise_data = generate_noise(batch_size, 100)
	   generated_images = generator.predict(noise_data)

	    # Train on soft labels (add noise to labels as well)
	   noise_prop = 0.05 # Randomly flip 5% of labels
    
	    # Prepare labels for real data
	   true_labels = np.zeros((batch_size, 1)) + np.random.uniform(low=0.0, high=0.1, size=(batch_size, 1))
	   flipped_idx = np.random.choice(np.arange(len(true_labels)), size=int(noise_prop*len(true_labels)))
	   true_labels[flipped_idx] = 1 - true_labels[flipped_idx]
	   
	    # Train discriminator on real data
	   d_loss_true = discriminator.train_on_batch(images,true_labels)   
	   # Prepare labels for generated data
	   gene_labels = np.ones((batch_size, 1)) - np.random.uniform(low=0.0, high=0.1, size=(batch_size, 1))
	   flipped_idx = np.random.choice(np.arange(len(gene_labels)), size=int(noise_prop*len(gene_labels)))
	   gene_labels[flipped_idx] = 1 - gene_labels[flipped_idx]
	   # Train discriminator on generated data
	   d_loss_gene = discriminator.train_on_batch(generated_images, gene_labels)
	   d_loss = 0.5 * np.add(d_loss_true, d_loss_gene)
	   cum_d_loss += d_loss
	   # Train generator
	   noise_data = generate_noise(batch_size, 100)
	   g_loss = gan.train_on_batch(noise_data, np.zeros((batch_size, 1)))
	   cum_g_loss += g_loss
	   print('  Epoch: {}, Generator Loss: {}, Discriminator Loss: {}'.format(epoch+1, cum_g_loss/num_batches, cum_d_loss/num_batches))
	   show_imgs("epoch" + str(epoch))














