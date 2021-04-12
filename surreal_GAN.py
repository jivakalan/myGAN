from keras.layers import Input, BatchNormalization, Conv2D, Conv2DTranspose, Flatten, Dropout, Reshape
from keras.models import Model
from keras.layers.core import Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
#from keras import initializers
import matplotlib.pyplot as plt 
#import sys
import os
import numpy as np
#from PIL import Image
#import glob
from IPython.display import display
from IPython.display import Image as _Imgdis
#from scipy import ndimage
from keras.preprocessing.image import load_img, img_to_array, image
from sklearn.model_selection import train_test_split
#from __future__ import print_function, division


class Image_Processor():
	def __init__(self, image_location=r"C:\Users\jkalan\Downloads\Surrealism"):	
		self.image_location = image_location
		self.image_width = 640
		self.image_height = 480
		self.ratio = 4	
		self.channels = 3
		self.nb_classes = 1
		self.test_size=0.3

		self.train_files = []
		self.y_train = []
	
	def image_importer(self):
		surrealism_imgs = [f for f in os.listdir(self.image_location) if os.path.isfile(os.path.join(self.image_location,f))]
		print("Working with {0} images".format(len(surrealism_imgs)))
		print("Image examples: ")
		for i in range(1,3):
			print(surrealism_imgs[i])
			display(_Imgdis(filename=self.image_location + "/" + surrealism_imgs[i], width=240, height=320))


		i=0
		for _file in surrealism_imgs:
			self.train_files.append(_file)
			label_in_file = _file.find("_")
			self.y_train.append(str(_file[0:label_in_file]))
		print("Files in train_files: %d" % len(self.train_files))
		
		image_width = int(self.image_width / self.ratio)
		image_height = int(self.image_height / self.ratio)	
		self.dataset = np.ndarray(shape=(len(self.train_files), self.channels, image_height, image_width),
		                     dtype=np.float32)
		i = 0
		for _file in self.train_files:
		    img = load_img(self.image_location + "/" + _file)  
		    img.thumbnail((image_width, image_height))
		    # Convert to Numpy Array
		    x = img_to_array(img)  
		    x = x.reshape((3, 120, 160))
		    # Normalize
		    x = (x - 127.5) / 127.5
		    self.dataset[i] = x
		    i += 1
		    if i % 100 == 0:
		        print("%d images to array" % i)
		print("All images to array!")
		


r=Image_Processor()
r.image_importer()
dataset=np.asarray(r.dataset)
y_train = r.y_train

#Splitting
X_train, X_test, y_train, y_test = train_test_split(dataset, y_train, test_size=0.3, random_state=33)
x_test, x_val, y_test, y_val = train_test_split(X_train, y_test, test_size=0.3, random_state=33)
print("Train set size: {0}, Val set size: {1}, Test set size: {2}".format(len(X_train), len(x_val), len(X_test)))


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


#############GAN comes here

def surreal_generator(input_layer):
	generator = Dense(128*16*16,activation='relu')(input_layer)
	generator = BatchNormalization(momentum=0.9)(generator)
	generator = LeakyReLU(alpha=0.1)(generator)
	generator = Reshape((16, 16, 128))(generator)
	
	generator =  Conv2D(128, kernel_size=5, strides=1,padding='same')(generator)
	generator = BatchNormalization(momentum=0.9)(generator)
	generator = LeakyReLU(alpha=0.1)(generator)
	
	generator = Conv2DTranspose(128, 4, strides=2, padding='same')(generator)
	generator = BatchNormalization(momentum=0.9)(generator)
	generator = LeakyReLU(alpha=0.1)(generator)
	 
	generator = Conv2D(128, kernel_size=5, strides=1, padding='same')(generator)
	generator = BatchNormalization(momentum=0.9)(generator)
	generator = LeakyReLU(alpha=0.1)(generator)

	generator = Conv2D(128, kernel_size=5, strides=1, padding='same')(generator)
	generator = BatchNormalization(momentum=0.9)(generator)
	generator = LeakyReLU(alpha=0.1)(generator)

	generator = Conv2D(3, kernel_size=5, strides=1, padding='same')(generator)      
	out = Activation("tanh")(generator)
	
	model = Model(input_layer, out)
	model.summary()
	
	return model, out

def surreal_discriminator(input_layer):
	discriminator =  Conv2D(128, kernel_size=3, strides=1,padding='same')(input_layer)
	discriminator = BatchNormalization(momentum=0.9)(discriminator)
	discriminator = LeakyReLU(alpha=0.1)(discriminator)
	
	discriminator =  Conv2D(128, kernel_size=4, strides=2,padding='same')(discriminator)
	discriminator = BatchNormalization(momentum=0.9)(discriminator)
	discriminator = LeakyReLU(alpha=0.1)(discriminator)	

	discriminator =  Conv2D(128, kernel_size=4, strides=2,padding='same')(discriminator)
	discriminator = BatchNormalization(momentum=0.9)(discriminator)
	discriminator = LeakyReLU(alpha=0.1)(discriminator)

	discriminator =  Conv2D(128, kernel_size=4, strides=2,padding='same')(discriminator)
	discriminator = BatchNormalization(momentum=0.9)(discriminator)
	discriminator = LeakyReLU(alpha=0.1)(discriminator)	

	discriminator = Flatten()(discriminator)
	discriminator = Dropout(0.4)(discriminator)
	out = Dense(1,activation ='sigmoid')(discriminator)

	model = Model(input_layer,out)
	model.summary()

	return model, out	



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


