# Data science portfolio by Luke Monington

This portfolio is a compilation of AI notebooks which I wrote in order to explore machine learning and deep learning.

### Temperature Regulator - Reinforcement Learning [Completed]
In my current apartment, the shower temperature fluctuates frequently, especially 
if someone flushes the toilet. In this project, I simulated the fluctuating shower environment and 
built a reinforcement learning model with TensorFlow in order to keep the temperature within 
the desired range. To do this, I built a Sequential Neural Network and used a DQNAgent with a BoltzmannQPolicy.
[Github](https://github.com/lukemonington/shower_temp_reinforcement_learning)

### Landscape Recognition - CNN / Transfer Learning [Completed]
Image recognition using deep learning is a driving force behind many of today's AI advancements. The purpose 
of this project is to use deep learning in order to train a neural network to classify different types of 
landscape within images. The categories are: buildings, forest, glacier, mountain, sea, and street. 
I first did this by building my own Convolutional Neural Network using MaxPooling2D and Dropout. 
Then, I used transfer learning with the ResNet50V2 architecture to improve on those results.
[Github](https://github.com/lukemonington/landscape_classification)

### Yelp Reviews - NLP Binary Classifier [Completed]
In this project, I worked with the Yelp Reviews dataset, which contains over 7 million reviews. The purpose of this project 
is to use NLP to classify these Yelp reviews into positive reviews (4-5 stars) and negative reviews (1-2 stars). To do this, 
I experimented with a Conv1D NN Architecture and a Bidirectional LSTM NN Architecture. In the end, I found that the Conv1D 
NN Architecture achieved nearly the same results, but with a drastically lower training time.
[Github](https://github.com/lukemonington/yelp_reviews)

### Pokemon Images - Autoencoder / GAN [Completed]
An autoencoder is a type of NN which is used to learn efficient representations for a set of data by ignoring the noise.
The purpose of this project was to train an autoencoder to remove gaussian noise from pokemon images. From there, I
attempted to build a GAN to generate new pokemon images.
[Github](https://github.com/lukemonington/pokemon_images_gan)

### Facial Recognition - VGGFace / OpenCV [Completed]
OpenCV is a library of programming functions mainly aimed at real-time computer vision. One of its uses is for facial recognition.
In this project, I use OpenCV to detect faces and eyes in images using OpenCV's Haar feature-based cascade classifier. This classifier 
also returns the location, height, and width of the face. I use that information in order to crop out the face from images and videos.
From there, I use the VGGFace pretrained Neural Network to do One Shot Learning and perform real-time facial recognition with my webcam.
[Github](https://github.com/lukemonington/facial_recognition_opencv)

### Categorical Feature Encoding Challenge - Machine Learning [Completed]
In this project, I experimented with different types of categorical features, sampling methods, and predictive algorithms in order to 
implement machine learning. This required different methods of feature engineering in order to work with the dataset, which consisted of 
5 binary columns, 10 nominal columns, and 6 ordinal columns. 
[Github](https://github.com/lukemonington/Categorical-Feature-Encoding-Challenge)

### Household Electric Power Consumption
One promising use of neural networks is for time series forecasting. Here, I work with forecasting electric power consumption in
a single household, when given 4 years of data with a one-minute sampling rate. First, I try predicting a single time step into the future.
I build and compare models such as linear models, Neural Networks, CNNs, and LSTM Neural Networks. Then, I build and compare models that predict
24 time steps into the future.
[Github](https://github.com/lukemonington/household_electric_power_consumption)

### Super Resolution
[Github](https://github.com/lukemonington/super_resolution)

### Genetic Algorithm
[Github](https://github.com/lukemonington/genetic_algorithm)

### Don't Overfit AI Challenge - Machine Learning
[Github](https://github.com/lukemonington/Don-t-Overfit-AI-Challenge)

### Predicting Higgs Boson - Machine Learning
[Github](https://github.com/lukemonington/Higgs-Boson-machine-learning-challenge)

### Fraud Detection - Machine Learning
[Github](https://github.com/lukemonington/IEEE-CIS-Fraud-Detection-AI-Competition)

### Car Rental Company Relational Database
[Github](https://github.com/lukemonington/Car-Rental-Company-Relational-Database)
