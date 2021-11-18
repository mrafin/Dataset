import ctypes
hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\bin\\cudart64_110.dll")
hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\bin\\cublas64_11.dll")
hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\bin\\cublasLt64_11.dll")
hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\bin\\cufft64_10.dll")
hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\bin\\curand64_10.dll")
hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\bin\\cusolver64_11.dll")
hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\bin\\cusparse64_11.dll")
hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\bin\\cudnn64_8.dll")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import glob

# directory
image_path = 'Dataset\\contrast adjusment\\'
label_list = ['jeruk nipis','seledri']
data = []
labels =[]

# Pemasukan data dan label
for label in label_list:
    for imagePath in glob.glob(image_path +label+ '\\*.jpg'):
        # print(imagePath)
        image = cv2.imread(imagePath)
        image = cv2.resize(image,(32,32)).flatten()
        labels.append(label)
        data.append(image)

# Mengubah dari list ke numpy array
data = np.array(data, dtype=float) / 255.0
labels = np.array(labels)

# Perubahan label ke bentuk numeric
lb = LabelBinarizer()
labels =lb.fit_transform(labels)

# Split data ke train dan test
x_train,x_test,y_train,y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print('Ukuran data train =', x_train.shape)
print('Ukuran data test =', x_test.shape)

# penerapan ANN
model = Sequential()
model.add(Dense(512, input_shape=(3072,), activation="relu"))
model.add(Dense(1024, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

print(model.summary())

# tentukan hyperparameter
lr = 0.01
max_epochs = 100
opt_funct = SGD(learning_rate=lr)

# # compile arsitektur yang telah dibuat
model.compile(loss = 'binary_crossentropy', optimizer = opt_funct, metrics = ['accuracy'])
#from tensorflow.keras.optimizers import Adam
#model.compile(loss='binary_crossentropy',optimizer= SGD(lr=0.001),metrics=['acc'])

# # Train model
H = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=max_epochs, batch_size=32)

N = np.arange(0, max_epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.xlabel("Epoch #")
plt.legend()
plt.show()

# menghitung nilai akurasi model terhadap data test
predictions = model.predict(x_test, batch_size=32)
lbl = (predictions > 0.5).astype(np.int)
print(classification_report(y_test, lbl))


# uji model menggunakan image lain
queryPath = image_path+'002.jpg'
query = cv2.imread(queryPath)
output = query.copy()
query = cv2.resize(query, (32, 32)).flatten()
q = []
q.append(query)
q = np.array(q, dtype='float') / 255.0

q_pred = model.predict(q)
i = q_pred.argmax(axis=1)[0]
label = lb.classes_[i]

text = "{}: {:.2f}%".format(label, q_pred[0][i] * 100)
cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
# menampilkan output image
cv2.imshow('Output', output)
cv2.waitKey() # image tidak akan diclose,sebelum user menekan sembarang tombol
cv2.destroyWindow('Output') # image akan diclose