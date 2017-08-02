import csv
import cv2
import numpy as np
import sklearn
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Cropping2D, Lambda, Convolution2D

lines = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile) #, delimiter='\t')
	next(reader, None) # skip the header
	for line in reader:
		lines.append(line)

#from sklearn.model_selection import train_test_split
#train_samples, validation_samples = train_test_split(lines, test_size=0.2)
		
images = []
angles = []
steer_offset = 0.25
for line in lines:
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = "data/IMG/" + filename
		#print(current_path)
		img = cv2.imread(current_path)
		#print(img.shape)
		image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		images.append(image)
		if i == 1:
			angle = float(line[3]) + steer_offset
		elif i == 2:
			angle = float(line[3]) - steer_offset
		else:
			angle = float(line[3])
		angles.append(angle)

augmented_images, augmented_angles = [], []
for image,angle in zip(images, angles):
	augmented_images.append(image)
	augmented_angles.append(angle)
	augmented_images.append(cv2.flip(image,1))
	augmented_angles.append(angle*-1.0)

"""
def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1:
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			
			images = []
			angles = []
			for batch_sample in batch_samples:
				name
"""


X_train = np.array(augmented_images)
y_train = np.array(augmented_angles)
print(X_train.shape)


model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
exit()
