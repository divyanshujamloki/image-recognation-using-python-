#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


import numpy as np


# In[3]:


filename = 'C:\\Users\\91807\\Desktop\\data\\cars.jpg'


# In[4]:


filename = 'C:\\Users\\91807\\Desktop\\data\\bmw-m5-exterior0.jpeg'


# In[5]:


from tensorflow.keras.preprocessing import image
img = image.load_img(filename,target_size = (224,224))


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


plt.imshow(img)


# In[ ]:





# In[8]:


import cv2 


# In[9]:


imgg = cv2.imread(filename)


# In[10]:


plt.imshow(imgg)


# In[11]:


imgg = cv2.resize(imgg,(224,224))
plt.imshow(imgg)


# In[12]:


imgg = cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB )
plt.imshow(imgg)


# In[13]:


mobile= tf.keras.applications.mobilenet.MobileNet()


# In[14]:


mobile= tf.keras.applications.mobilenet_v2.MobileNetV2()


# In[15]:


from tensorflow.keras.preprocessing import image
img = image.load_img(filename,target_size = (224,224))


# In[16]:


plt.imshow(img)


# In[17]:


resized_img = image.img_to_array(img)
final_image= np.expand_dims(resized_img,axis=0)
final_image= tf.keras.applications.mobilenet.preprocess_input(final_image)


# In[18]:


print(resized_img)


# In[19]:


final_image.shape


# In[20]:


predictions = mobile.predict(final_image)


# In[21]:


#print(predictions)


# In[22]:


from  tensorflow.keras.applications import imagenet_utils


# In[23]:


results =  imagenet_utils.decode_predictions(predictions)


# In[24]:


print(results)


# In[25]:


plt.imshow(img)


# In[26]:


predictions = mobile.predict(final_image)


# In[27]:


results =  imagenet_utils.decode_predictions(predictions)


# In[28]:


print(results)


# In[ ]:





# In[ ]:




