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

### Household Electric Power Consumption - Neural Networks [Completed]
One promising use of neural networks is for time series forecasting. Here, I work with forecasting electric power consumption in
a single household, when given 4 years of data with a one-minute sampling rate. First, I try predicting a single time step into the future.
I build and compare models such as linear models, Neural Networks, CNNs, and LSTM Neural Networks. Then, I build and compare models that predict
24 time steps into the future.
[Github](https://github.com/lukemonington/household_electric_power_consumption)

### Hourly Staff Planning - Genetic Algorithms [Completed]
Staff planning is a topic of optimization research that comes back in many companies. As soon as a company has many employees, it becomes 
hard to find planning that suits the business needs while respecting certain constraints. In this project, I implement genetic 
algorithms to find an optimal hourly staff planning solution.
[Github](https://github.com/lukemonington/genetic_algorithm)

### Don't Overfit AI Challenge - Machine Learning [Completed]
In this project, I was challenged to not overfit to a dataset with only 300 features and 250 observations ot training data, while
predicting 19,750 rows of test data. The dataset was unbalanced, with almost twice as many positive examples as negative examples,
so I applied Synthetic Minority Over-sampling (SMOTE) to make the dataset more balanced. Then I compared the performance of different
classifiers such as RandomForestClassifier, SVC, and KNeighborsClassifier.
[Github](https://github.com/lukemonington/Don-t-Overfit-AI-Challenge)

### Car Rental Company Relational Database [Completed]
The challenge was to develop a relational database using Oracle SQL Developer for a hypothetical car rental business with several addresses. 
The business rents multiple different types of cars, which each have their own respective rental prices. The rental pricing is also dependent 
upon available promotions. Additionally, the business keeps track of all of its customers and employees. It is also possible to be both an 
employee and a customer at the same time.
[Github](https://github.com/lukemonington/Car-Rental-Company-Relational-Database)
