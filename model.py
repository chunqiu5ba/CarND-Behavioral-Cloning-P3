import csv
import cv2
import numpy as np
import sklearn
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Dropout

# Read image path csv file and skip header
lines = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader, None) # skip the header
	for line in reader:
		lines.append(line)

# Preprocess image, crop and resize to (66, 200)
def img_process(image):
	image = image[50:-20,:]
	image = cv2.resize(image, (200, 66), interpolation=cv2.INTER_AREA)
	
	return image

# Find Training image and steering data
# Using images from all 3 cameras
images, angles = [], []
steer_offset = 0.27 # Adjustment for left and righ camera
for line in lines:
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = "data/IMG/" + filename
		#print(current_path)
		img = cv2.imread(current_path)
		#print(img.shape)
		image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		images.append(img_process(image))
		if i == 1:
			angle = float(line[3]) + steer_offset
		elif i == 2:
			angle = float(line[3]) - steer_offset
		else:
			angle = float(line[3])
		angles.append(angle)

# Double the dataset with flipped images
augmented_images, augmented_angles = [], []
for image,angle in zip(images, angles):
	augmented_images.append(image)
	augmented_angles.append(angle)
	augmented_images.append(cv2.flip(image,1))
	augmented_angles.append(angle*-1.0)

# Split the dataset to 80% Trianning data and 20% Validation
from sklearn.model_selection import train_test_split
samples = list(zip(augmented_images, augmented_angles))
train_samples, validation_samples = train_test_split( \
	samples, test_size=0.2, random_state=42)

# Generator with 64 image per batch
def generator(samples, batch_size=64):
	num_samples = len(samples)
	sklearn.utils.shuffle(samples)
	X_samples, y_samples = zip(*samples)
	while 1:
		for offset in range(0, num_samples, batch_size):
			X_batch_samples = X_samples[offset:offset+batch_size]
			y_batch_samples = y_samples[offset:offset+batch_size]
			
			X_train = np.array(X_batch_samples)
			#print(X_train.shape)
			y_train = np.array(y_batch_samples)
			yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

# Implement NVIDIA Network
model = Sequential()
# Normalize and mean-center the data
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (66, 200, 3), name='Nor'))
# Convolutional layer 1 24@31x98 | 5x5 kernel | 2x2 stride | relu activation
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu', name='Con1'))
# Convolutional layer 1 36@14x47 | 5x5 kernel | 2x2 stride | relu activation
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu', name='Con2'))
# Convolutional layer 3 48@5x22  | 5x5 kernel | 2x2 stride | relu activation
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu', name='Con3'))
# Convolutional layer 4 64@3x20  | 3x3 kernel | 1x1 stride | relu activation
model.add(Convolution2D(64,3,3, activation='relu'))
# Convolutional layer 5 64@1x18  | 3x3 kernel | 1x1 stride | relu activation
model.add(Convolution2D(64,3,3, activation='relu'))
# Dropout 0.5
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
# Output
model.add(Dense(1))

# Compile and train the model with generator
model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), \
	validation_data=validation_generator, nb_val_samples=len(validation_samples), \
	nb_epoch=5, verbose=1)
			
model.save('model.h5')

print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])


if __name__ == '__main__':
	
	from keras import backend as K 
	K.clear_session()