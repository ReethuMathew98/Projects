#!/usr/bin/env python
# coding: utf-8

# # CODE : A Study on the Visual Elements of Book Covers using Machine Learning Techniques

# In[248]:


get_ipython().run_cell_magic('capture', '', 'import warnings\nwarnings.filterwarnings("ignore")\n')


# ## THE DATASET

# In[1]:


# Importing Required Libraries
import pandas as pd


# #### Loading the Dataset

# In[2]:


# Loading the dataset
# Loading the data
df_1 = pd.read_csv("./Data/book30-listing-train.csv", encoding = "ISO-8859-1")
df_2 = pd.read_csv("./Data/book30-listing-test.csv", encoding = "ISO-8859-1")
# Creating column headers
df_1.columns = ['Index', 'Filename', 'URL', 'Title', 'Author', 'Category_ID', 'Category']
df_2.columns = ['Index', 'Filename', 'URL', 'Title', 'Author', 'Category_ID', 'Category']
# combining the separate data
df = pd.concat([df_1, df_2], axis=0)


# In[3]:


# Folder containing the images
images_directory = r".\Data\Images\224x224"


# #### Dataset Description

# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


df.info()


# #### Checking for null values

# In[7]:


df.isna().sum()


# #### Checking for duplicates

# In[8]:


# Check for duplicate rows
duplicate_rows = df[df.duplicated()]
# Display the duplicate rows
print("Duplicate Rows except first occurrence:")
print(duplicate_rows)


# In[9]:


# Checking duplicate values in Title column
# Check for duplicates in the specified column
duplicates_in_column = df['Title'].duplicated().sum()

# Display the number of duplicates in the specified column
print(f"\nNumber of duplicates in column 'Title': {duplicates_in_column}")


# ## COLOR EXTRACTION

# In[10]:


# Importing required libraries
import os
import numpy as np
import matplotlib.image as img
from sklearn.cluster import KMeans


# In[11]:


# Create a function to extract colors
def extract_colours(image):
    # Get the dimensions
    w, h, d = tuple(image.shape)
    # Reshape the image into an array where each row represents a pixel
    pixel = np.reshape(image, (w * h, d))
    # Set the desired number of colors in the image
    n_colours = 5
    # Create a KMeans model with the specified number of clusters and fit it to the pixels
    model = KMeans(n_clusters=n_colours, random_state=42).fit(pixel)
    # Get the cluster centers from the model
    colour_palette = np.uint8(model.cluster_centers_)
    # Get the RGB codes from the clusters
    rgb_colours = colour_palette.tolist()
    return rgb_colours


# In[12]:


def apply_extraction(row):
    image_path = os.path.join(images_directory, row['Filename'])
    image = img.imread(image_path)
    colours = extract_colours(image)
    return colours


# In[13]:


# Applying the colour extraction to all the rows
df['Colours'] = df.apply(apply_extraction, axis=1)


# In[14]:


df.to_csv('df_colours.csv', index=False)


# In[15]:


df.head()


# Code Reference : https://www.geeksforgeeks.org/extract-dominant-colors-of-an-image-using-python/

# In[16]:


df.shape


# ## TEXT AREA CALCULATION

# In[17]:


# Importing required libraries
import easyocr
from PIL import Image


# In[18]:


def read_text(image):
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(image)
    return result


# In[19]:


def calculate_area(row):
    area = 0.0
    image_path = os.path.join(images_directory, row['Filename'])
    text_output = read_text(image_path)
    img = Image.open(image_path)
    w, h = img.size
    for detection in text_output:
        top_left = tuple([int(val) for val in detection[0][0]])
        x1, y1 = top_left
        bottom_right = tuple([int(val) for val in detection[0][2]])
        x2, y2 = bottom_right
        a = (((x2-x1)*(y2-y1))/(h*w))*100
        area += a
    return round(area, 2)


# In[20]:


df['Text_Area'] = df.apply(calculate_area, axis=1)


# In[21]:


df.to_csv('df_colours_text.csv', index=False)


# In[22]:


df.head()


# Ref: https://youtu.be/ZVKaWPW9oQY?si=UsCOzG5BFo2a_Ju_

# ## OBJECT RECOGNITION

# In[23]:


# Importing required libraries
import io
from google.cloud import vision
from google.cloud.vision_v1 import types


# In[24]:


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r".\Vision_API_credentials.json"


# In[25]:


client = vision.ImageAnnotatorClient()
def detect_objects(row):
    image_path = os.path.join(images_directory, row['Filename'])
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = types.Image(content=content)
    objects = client.object_localization(image=image)
    obj = objects.localized_object_annotations
    objects_in_image = []
    for o in obj:
        objects_in_image.append(o.name)
    return list(set(objects_in_image))


# In[26]:


df['Objects'] = df.apply(detect_objects, axis=1)


# In[27]:


df.to_csv('df_final.csv', index=False)


# In[28]:


df.head()


# In[29]:


# Replacing the empty strings with NaN
df['Objects'] = df['Objects'].apply(lambda x: np.nan if len(x) == 0 else x)


# In[30]:


# Final dataframe
df.head()


# In[31]:


# Creating a copy of the dataframe for machine learning methods
main_df = df.copy()


# In[32]:


main_df.head()


# Ref: https://youtu.be/zOKz0e8flTw?si=5qRdF86XlRfFxQJC

# ## PREDICTING CATEGORY FROM BOOK TITLE

# In[33]:


# Importing required libraries
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.neural_network import MLPClassifier


# In[34]:


books = pd.DataFrame(main_df['Title'])
genre = pd.DataFrame(main_df['Category'])


# In[35]:


# Unique values in the category column
genre['Category'].unique()


# In[36]:


# Encoding the category column
feat = ['Category']
for x in feat:
    le = LabelEncoder()
    le.fit(list(genre[x].values))
    genre[x] = le.transform(list(genre[x]))


# In[37]:


# Category values after encoding
genre['Category'].unique()


# In[38]:


# Getting the label from encoding
le.inverse_transform([0])[0]


# In[39]:


# downloading stopwords
nltk.download('stopwords')
stop = list(stopwords.words('English'))
stop[:10]


# In[40]:


# function to remove stopwords
def change(t):
    t = t.split()
    return ' '.join([(i) for (i) in t if i not in stop])


# In[45]:


# Removig stopwords from the title column
main_df['Title'] = main_df['Title'].apply(change)


# In[46]:


# Vectorizing the title column
vectorizer = TfidfVectorizer(min_df=2, max_features=70000, strip_accents='unicode', lowercase=True, analyzer='word', 
                             token_pattern=r'\w+', use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words='english')
vectors = vectorizer.fit_transform(main_df['Title'])
vectors.shape


# In[77]:


# Splitting the data into testing and training datasets
X_train, X_test, y_train, y_test = train_test_split(vectors, genre['Category'], test_size=0.2)


# ### Model 1 : Logistic Regression

# In[49]:


title_model_lr = linear_model.LogisticRegression(solver='sag', max_iter=200, random_state=450)
title_model_lr.fit(X_train, y_train)
pred = title_model_lr.predict(X_test)
print(metrics.f1_score(y_test, pred, average='macro'))
print(metrics.accuracy_score(y_test, pred))


# In[50]:


title_model_lr


# ### Model 2 : Neural Network MLPClassifier

# In[78]:


title_model_nn = MLPClassifier(activation='logistic', alpha=0.00003, batch_size='auto', beta_1=0.9, beta_2=0.999, 
                               early_stopping=False, epsilon=1e-08, hidden_layer_sizes=(20,), learning_rate='constant', 
                               learning_rate_init=0.003, max_iter=200, momentum=0.9, nesterovs_momentum=True, power_t=0.5, 
                               random_state=450, shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1, 
                               verbose=False, warm_start=False)
title_model_nn.fit(X_train, y_train)
pred = title_model_nn.predict(X_test)
print(metrics.f1_score(y_test, pred, average='macro'))
print(metrics.accuracy_score(y_test, pred))


# In[79]:


title_model_nn


# Choosing the logistic regression model as it has higher F1 score and accuracy.

# In[58]:


# Test book title
text = ['The Ballad of Songbirds and Snakes']
# Vectorizing the title
s = (vectorizer.transform(text))
# Predicting the category
d = (title_model_lr.predict(s))


# In[59]:


# Reverse encoding to reveal the category
le.inverse_transform(d)[0]


# Ref: https://github.com/akshaybhatia10/Book-Genre-Classification/blob/fb00e2477a8ba833f9a2de0b2c0a662e69af5f05/Best_TFIDF-Vectorizer_model.ipynb

# ## PREDICTING COLOURS FROM CATEGORY

# In[134]:


# Importing required libraries
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVR, SVC
import matplotlib.pyplot as plt
import colorsys


# In[60]:


main_df.head()


# In[61]:


# Splitting the Colours column into 5 separate columns
main_df[['Colour1', 'Colour2', 'Colour3', 'Colour4', 'Colour5']] = pd.DataFrame(main_df['Colours'].tolist(), index=main_df.index)


# In[62]:


main_df.head()


# The five colours are sorted in descending order based on their occurrence in the image. So for each colours the red, green and blue values will be predicted. First we'll take a look at different models we can use for this prediction. 
# 
# X is the category and y is the components of each color

# In[204]:


# Encoding the Category values
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(pd.DataFrame(main_df['Category']))


# In[81]:


# Colour1: RED component
y = main_df['Colour1'].apply(lambda x: x[0])


# In[84]:


# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=450)


# ### Linear Regression

# In[85]:


colour_model_lr = LinearRegression()
colour_model_lr.fit(X_train, y_train)
pred = colour_model_lr.predict(X_test)
mse_lr = mean_squared_error(y_test, pred)
mae_lr = mean_absolute_error(y_test, pred)
r2_lr = r2_score(y_test, pred)
print(f'Mean Squared Error (MSE): {mse_lr}')
print(f'Mean Absolute Error (MAE): {mae_lr}')
print(f'R-squared (R2) Score: {r2_lr}')


# In[86]:


colour_model_lr


# ### Decision Tree Regression

# In[87]:


colour_model_dt = DecisionTreeRegressor()
colour_model_dt.fit(X_train, y_train)
pred = colour_model_dt.predict(X_test)
mse_dt = mean_squared_error(y_test, pred)
mae_dt = mean_absolute_error(y_test, pred)
r2_dt = r2_score(y_test, pred)
print(f'Mean Squared Error (MSE): {mse_dt}')
print(f'Mean Absolute Error (MAE): {mae_dt}')
print(f'R-squared (R2) Score: {r2_dt}')


# In[88]:


colour_model_dt


# ### Random Forest Regression

# In[89]:


colour_model_rf = RandomForestRegressor()
colour_model_rf.fit(X_train, y_train)
pred = colour_model_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, pred)
mae_rf = mean_absolute_error(y_test, pred)
r2_rf = r2_score(y_test, pred)
print(f'Mean Squared Error (MSE): {mse_rf}')
print(f'Mean Absolute Error (MAE): {mae_rf}')
print(f'R-squared (R2) Score: {r2_rf}')


# In[90]:


colour_model_rf


# ### Support Vector Regression

# In[91]:


colour_model_svr = SVR()
colour_model_svr.fit(X_train, y_train)
pred = colour_model_svr.predict(X_test)
mse_svr = mean_squared_error(y_test, pred)
mae_svr= mean_absolute_error(y_test, pred)
r2_svr= r2_score(y_test, pred)
print(f'Mean Squared Error (MSE): {mse_svr}')
print(f'Mean Absolute Error (MAE): {mae_svr}')
print(f'R-squared (R2) Score: {r2_svr}')


# In[92]:


colour_model_svr


# While comparing the evaluation metrics, it is preferred to have lower MSE and MAE values and a higher R-Squared values. Even though all four models have similar values in all three metrics, model 4 seem to be least effective. So, out of the first three regression models, the decision tree regression model is chosen as the prediction model.

# In[93]:


# Category as input
new_X = pd.Series(["Children's Books"])
# Encoding the category value
new_X_encoded = onehotencoder.transform(pd.DataFrame(new_X))
# Prediction with the chosen model
red_pred_1 = colour_model_dt.predict(new_X_encoded)
print(f'Predicted R value for {new_X.iloc[0]}: {int(red_pred_1[0].round())}')


# We can now train the model on the blue and green values of Colour1 and display the colour.

# In[95]:


# Green values of Colour1
y = main_df['Colour1'].apply(lambda x: x[1])
# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=450)
# Training the model
colour_model_dt = DecisionTreeRegressor()
colour_model_dt.fit(X_train, y_train)
pred = colour_model_dt.predict(X_test)
mse_dt_g = mean_squared_error(y_test, pred)
mae_dt_g = mean_absolute_error(y_test, pred)
r2_dt_g = r2_score(y_test, pred)
print(f'Mean Squared Error (MSE): {mse_dt_g}')
print(f'Mean Absolute Error (MAE): {mae_dt_g}')
print(f'R-squared (R2) Score: {r2_dt_g}')
# Prediction
new_X = pd.Series(["Children's Books"])
new_X_encoded = onehotencoder.transform(pd.DataFrame(new_X))
green_pred_1 = colour_model_dt.predict(new_X_encoded)
print(f'Predicted G value for {new_X.iloc[0]}: {int(green_pred_1[0].round())}')


# In[96]:


# Blue values of Colour1
y = main_df['Colour1'].apply(lambda x: x[2])
# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=450)
# Training the model
colour_model_dt = DecisionTreeRegressor()
colour_model_dt.fit(X_train, y_train)
pred = colour_model_dt.predict(X_test)
mse_dt_b = mean_squared_error(y_test, pred)
mae_dt_b = mean_absolute_error(y_test, pred)
r2_dt_b = r2_score(y_test, pred)
print(f'Mean Squared Error (MSE): {mse_dt_b}')
print(f'Mean Absolute Error (MAE): {mae_dt_b}')
print(f'R-squared (R2) Score: {r2_dt_b}')
# Prediction
new_X = pd.Series(["Children's Books"])
new_X_encoded = onehotencoder.transform(pd.DataFrame(new_X))
blue_pred_1 = colour_model_dt.predict(new_X_encoded)
print(f'Predicted B value for {new_X.iloc[0]}: {int(blue_pred_1[0].round())}')


# In[99]:


# Displaying the colour
Colour1_prediction = (int(red_pred_1[0].round()), int(green_pred_1[0].round()), int(blue_pred_1[0].round()))
colour1_image = [[Colour1_prediction]]
plt.imshow(colour1_image)
plt.title(f'RGB: {Colour1_prediction}')
plt.show()


# In[101]:


# Code to input a column and predict the RGB values for a category based on that column
colour2_pred = []
for i in range(3):
    y = main_df['Colour2'].apply(lambda x: x[i])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=450)
    colour_model_dt = DecisionTreeRegressor()
    colour_model_dt.fit(X_train, y_train)
    new_X = pd.Series(["Children's Books"])
    new_X_encoded = onehotencoder.transform(pd.DataFrame(new_X))
    new_prediction = colour_model_dt.predict(new_X_encoded)
    colour2_pred.append(int(new_prediction[0].round()))
print(colour2_pred)


# In[102]:


# Code to input a Category and predict the five colours based on each five columns
predicted_top_5_colours = []
for j in range(5):
    column_name = f"Colour{j+1}"
    colour_pred = []
    for i in range(3):
        y = main_df[column_name].apply(lambda x: x[i])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=450)
        colour_model_dt = DecisionTreeRegressor()
        colour_model_dt.fit(X_train, y_train)
        new_X = pd.Series(["Children's Books"])
        new_X_encoded = onehotencoder.transform(pd.DataFrame(new_X))
        new_prediction = colour_model_dt.predict(new_X_encoded)
        colour_pred.append(int(new_prediction[0].round()))
    predicted_top_5_colours.append(colour_pred)


# In[103]:


predicted_top_5_colours


# In[110]:


# Displaying all five colours
fig, ax = plt.subplots(1, 1, figsize=(8,2))
for i, colour in enumerate(predicted_top_5_colours):
    ax.axvline(x=i, color=np.array(colour)/255.0, linewidth=100)
ax.set_yticks([])
ax.set_xticks(range(len(predicted_top_5_colours)))
ax.set_xticklabels([])
plt.show


# In[206]:


# Trying another category
predicted_top_5_colours = []
for j in range(5):
    column_name = f"Colour{j+1}"
    colour_pred = []
    for i in range(3):
        y = main_df[column_name].apply(lambda x: x[i])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=450)
        colour_model_dt = DecisionTreeRegressor()
        colour_model_dt.fit(X_train, y_train)
        new_X = pd.Series(["History"])
        new_X_encoded = onehotencoder.transform(pd.DataFrame(new_X))
        new_prediction = colour_model_dt.predict(new_X_encoded)
        colour_pred.append(int(new_prediction[0].round()))
    predicted_top_5_colours.append(colour_pred)
print(predicted_top_5_colours)
fig, ax = plt.subplots(1, 1, figsize=(8,2))
for i, colour in enumerate(predicted_top_5_colours):
    ax.axvline(x=i, color=np.array(colour)/255.0, linewidth=100)
ax.set_yticks([])
ax.set_xticks(range(len(predicted_top_5_colours)))
ax.set_xticklabels([])
plt.show


# In[191]:


# Getting all the unique categories into a list
categories = main_df['Category'].unique().tolist()
categories


# In[209]:


for category in categories:
    predicted_top_5_colours = []
    for j in range(5):
        column_name = f"Colour{j+1}"
        colour_pred = []
        for i in range(3):
            y = main_df[column_name].apply(lambda x: x[i])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=450)
            colour_model_dt = DecisionTreeRegressor()
            colour_model_dt.fit(X_train, y_train)
            new_X = pd.Series([category])
            new_X_encoded = onehotencoder.transform(pd.DataFrame(new_X))
            new_prediction = colour_model_dt.predict(new_X_encoded)
            colour_pred.append(int(new_prediction[0].round()))
        predicted_top_5_colours.append(colour_pred)
    print(category)
    print(predicted_top_5_colours)
    fig, ax = plt.subplots(1, 1, figsize=(8,2))
    for i, colour in enumerate(predicted_top_5_colours):
        ax.axvline(x=i, color=np.array(colour)/255.0, linewidth=100)
    ax.set_yticks([])
    ax.set_xticks(range(len(predicted_top_5_colours)))
    ax.set_xticklabels([])
    plt.show


# Regardless of the category, the output is very similar in colour. We can now try another form for colour representation.
# 
# HSL (Hue, Saturation, Lightness) values of colours are a different way of representing colours. We will now convert the RGB colours into their HSL values and train the models with those values.

# In[112]:


# Function to convert RGB to HSL
def rgb_to_hsl(rgb):
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0

    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin

    # Calculate Hue
    if delta == 0:
        h = 0
    elif cmax == r:
        h = 60 * (((g - b) / delta) % 6)
    elif cmax == g:
        h = 60 * (((b - r) / delta) + 2)
    elif cmax == b:
        h = 60 * (((r - g) / delta) + 4)

    # Calculate Lightness
    l = (cmax + cmin) / 2

    # Calculate Saturation
    s = delta / (1 - abs(2 * l - 1)) if delta != 0 else 0

    return int(h), round(s * 100, 1), round(l * 100, 1)


# In[113]:


# Example
test_rgb = [148, 140, 125]
test_hsl = rgb_to_hsl(test_rgb)
print(test_hsl)


# In[114]:


# Applying to all the five colour columns
main_df['hsl_1'] = main_df['Colour1'].apply(rgb_to_hsl)
main_df['hsl_2'] = main_df['Colour2'].apply(rgb_to_hsl)
main_df['hsl_3'] = main_df['Colour3'].apply(rgb_to_hsl)
main_df['hsl_4'] = main_df['Colour4'].apply(rgb_to_hsl)
main_df['hsl_5'] = main_df['Colour5'].apply(rgb_to_hsl)


# In[115]:


main_df.head()


# In[116]:


# Hue values of Colour1
y = main_df['hsl_1'].apply(lambda x: x[0])


# In[117]:


# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=450)


# ### Linear Regression

# In[118]:


hsl_model_lr = LinearRegression()
hsl_model_lr.fit(X_train, y_train)
pred = hsl_model_lr.predict(X_test)
mse_lr = mean_squared_error(y_test, pred)
mae_lr = mean_absolute_error(y_test, pred)
r2_lr = r2_score(y_test, pred)
print(f'Mean Squared Error (MSE): {mse_lr}')
print(f'Mean Absolute Error (MAE): {mae_lr}')
print(f'R-squared (R2) Score: {r2_lr}')


# ### Decision Tree Regression

# In[119]:


hsl_model_dt = DecisionTreeRegressor()
hsl_model_dt.fit(X_train, y_train)
pred = hsl_model_dt.predict(X_test)
mse_dt = mean_squared_error(y_test, pred)
mae_dt = mean_absolute_error(y_test, pred)
r2_dt = r2_score(y_test, pred)
print(f'Mean Squared Error (MSE): {mse_dt}')
print(f'Mean Absolute Error (MAE): {mae_dt}')
print(f'R-squared (R2) Score: {r2_dt}')


# ### Random Forest Regression

# In[120]:


hsl_model_rf = RandomForestRegressor()
hsl_model_rf.fit(X_train, y_train)
pred = hsl_model_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, pred)
mae_rf = mean_absolute_error(y_test, pred)
r2_rf = r2_score(y_test, pred)
print(f'Mean Squared Error (MSE): {mse_rf}')
print(f'Mean Absolute Error (MAE): {mae_rf}')
print(f'R-squared (R2) Score: {r2_rf}')


# ### Support Vector Regression

# In[121]:


hsl_model_svr = SVR()
hsl_model_svr.fit(X_train, y_train)
pred = hsl_model_svr.predict(X_test)
mse_svr = mean_squared_error(y_test, pred)
mae_svr= mean_absolute_error(y_test, pred)
r2_svr= r2_score(y_test, pred)
print(f'Mean Squared Error (MSE): {mse_svr}')
print(f'Mean Absolute Error (MAE): {mae_svr}')
print(f'R-squared (R2) Score: {r2_svr}')


# Out of the four models, the decision tree regressor has the lowest MSE and MAE values and highest R-sqaured values. So we can use this model to predict the values based on the category.

# In[249]:


predicted_top_5_colours = []
for j in range(5):
    column_name = f"hsl_{j+1}"
    colour_pred = []
    for i in range(3):
        y = main_df[column_name].apply(lambda x: x[i])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=450)
        hsl_model_dt = DecisionTreeRegressor()
        hsl_model_dt.fit(X_train, y_train)
        new_X = pd.Series(["History"])
        new_X_encoded = onehotencoder.transform(pd.DataFrame(new_X))
        new_prediction = hsl_model_dt.predict(new_X_encoded)
        colour_pred.append(int(new_prediction[0].round()))
    predicted_top_5_colours.append(colour_pred)
predicted_top_5_colours


# In[126]:


def convert_colour(h, s, l):
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return int(r * 255), int(g * 255), int(b * 255)


# In[127]:


fig, ax = plt.subplots(1, len(predicted_top_5_colours), figsize=(12,2))
for i, hsl in enumerate(predicted_top_5_colours):
    rgb = convert_colour(hsl[0]/360, hsl[1]/100, hsl[2]/100)
    ax[i].add_patch(plt.Rectangle((0, 0), 1, 1, color=[comp/255 for comp in rgb]))
    ax[i].set_title(f'HSL: {hsl}')
    ax[i].set_xticks([])
    ax[i].set_yticks([])
plt.show()


# In[250]:


# Trying another category
predicted_top_5_colours = []
for j in range(5):
    column_name = f"hsl_{j+1}"
    colour_pred = []
    for i in range(3):
        y = main_df[column_name].apply(lambda x: x[i])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=450)
        hsl_model_dt = DecisionTreeRegressor()
        hsl_model_dt.fit(X_train, y_train)
        new_X = pd.Series(["Romance"])
        new_X_encoded = onehotencoder.transform(pd.DataFrame(new_X))
        new_prediction = hsl_model_dt.predict(new_X_encoded)
        colour_pred.append(int(new_prediction[0].round()))
    predicted_top_5_colours.append(colour_pred)
fig, ax = plt.subplots(1, len(predicted_top_5_colours), figsize=(12,2))
for i, hsl in enumerate(predicted_top_5_colours):
    rgb = convert_colour(hsl[0]/360, hsl[1]/100, hsl[2]/100)
    ax[i].add_patch(plt.Rectangle((0, 0), 1, 1, color=[comp/255 for comp in rgb]))
    ax[i].set_title(f'HSL: {hsl}')
    ax[i].set_xticks([])
    ax[i].set_yticks([])
plt.show()


# In[210]:


categories


# In[251]:


for category in categories:
    print(category)
    predicted_top_5_colours = []
    for j in range(5):
        column_name = f"hsl_{j+1}"
        colour_pred = []
        for i in range(3):
            y = main_df[column_name].apply(lambda x: x[i])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=450)
            hsl_model_dt = DecisionTreeRegressor()
            hsl_model_dt.fit(X_train, y_train)
            new_X = pd.Series([category])
            new_X_encoded = onehotencoder.transform(pd.DataFrame(new_X))
            new_prediction = hsl_model_dt.predict(new_X_encoded)
            colour_pred.append(int(new_prediction[0].round()))
        predicted_top_5_colours.append(colour_pred)
    fig, ax = plt.subplots(1, len(predicted_top_5_colours), figsize=(12,2))
    for i, hsl in enumerate(predicted_top_5_colours):
        rgb = convert_colour(hsl[0]/360, hsl[1]/100, hsl[2]/100)
        ax[i].add_patch(plt.Rectangle((0, 0), 1, 1, color=[comp/255 for comp in rgb]))
        ax[i].set_title(f'HSL: {hsl}')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.show()


# This is also showing very similar colours for different categories. Since regression models are not working properly we will try some categorisation models. But before that we will convert these HSL values into categorical values. Each entry is assigned a dominant colour based on the hue and lightness. The saturation is omitted for this classification. The following are the criterion under which the colours were categorised:
# 
# ###### HUE:
# 
# Red 0-60 and 300-360
# 
# Green 60 - 180 
# 
# Blue 180 - 300
# 
# ###### LIGHTNESS:
# 
# Black 0-20 
# 
# White 80-100 
# 
# Ref : https://www.w3schools.com/html/html_colors_hsl.asp

# In[129]:


# Converting into labels
def label_value(values):
    hue, _, lightness = values

    # Lightness Check
    if lightness >= 0 and lightness <= 20:
        return "Black"
    elif lightness >= 80 and lightness <= 100:
        return "White"

    # Hue Check
    if (hue >= 0 and hue <= 60) or (hue >= 300 and hue <= 360):
        return "Red"
    elif hue >= 60 and hue < 180:
        return "Green"
    elif hue >= 180 and hue < 300:
        return "Blue"


# In[132]:


# Apply the label_value function to the 'colors' column
main_df['Colour1_labels'] = main_df['hsl_1'].apply(label_value)
main_df['Colour2_labels'] = main_df['hsl_2'].apply(label_value)
main_df['Colour3_labels'] = main_df['hsl_3'].apply(label_value)
main_df['Colour4_labels'] = main_df['hsl_4'].apply(label_value)
main_df['Colour5_labels'] = main_df['hsl_5'].apply(label_value)


# In[133]:


main_df.head()


# In[147]:


# Encoding the Category values
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(pd.DataFrame(main_df['Category']))


# In[148]:


y = main_df['Colour1_labels']


# In[149]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=450)


# ### Decision Tree Classifier

# In[150]:


label_model_dt = DecisionTreeClassifier()
label_model_dt.fit(X_train, y_train)
pred = label_model_dt.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print("Classification Report:\n", classification_report(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))


# ### Random Forest Classifier

# In[151]:


label_model_rf = RandomForestClassifier()
label_model_rf.fit(X_train, y_train)
pred = label_model_rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print("Classification Report:\n", classification_report(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))


# ### Logistic Regression

# In[152]:


label_model_lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=450)
label_model_lr.fit(X_train, y_train)
pred = label_model_lr.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print("Classification Report:\n", classification_report(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))


# ### Support Vector Machines

# In[153]:


label_model_svm = SVC(decision_function_shape='ovr', random_state=450)
label_model_svm.fit(X_train, y_train)
pred = label_model_svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print("Classification Report:\n", classification_report(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))


# Since all the models have the exact same accuracy, we'll use the logistic rergession model for prediction.

# In[155]:


new_X = pd.Series(["Children's Books"])
new_X_encoded = onehotencoder.transform(pd.DataFrame(new_X))
label1_pred = label_model_lr.predict(new_X_encoded)
print(f'Predicted label: {label1_pred[0]}')


# In[156]:


# Predicting colours from all five columns
predicted_labels = []
for j in range(5):
    column_name = f"Colour{j+1}_labels"
    y = main_df[column_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=450)
    label_model_lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=450)
    label_model_lr.fit(X_train, y_train)
    new_X = pd.Series(["Children's Books"])
    new_X_encoded = onehotencoder.transform(pd.DataFrame(new_X))
    new_prediction = label_model_lr.predict(new_X_encoded)
    predicted_labels.append(new_prediction[0])
predicted_labels


# In[157]:


# Trying another category
predicted_labels = []
for j in range(5):
    column_name = f"Colour{j+1}_labels"
    y = main_df[column_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=450)
    label_model_lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=450)
    label_model_lr.fit(X_train, y_train)
    new_X = pd.Series(["Health, Fitness & Dieting"])
    new_X_encoded = onehotencoder.transform(pd.DataFrame(new_X))
    new_prediction = label_model_lr.predict(new_X_encoded)
    predicted_labels.append(new_prediction[0])
predicted_labels


# Both the models output the same label(red). We'll compare the output with the actual distribution of the labels in these two categories.

# In[241]:


category_ = 'Health, Fitness & Dieting'
category_subset = main_df[main_df['Category'] == category_]

# Count the occurrences of each color label in the subset
color_counts = category_subset[['Colour1_labels', 'Colour2_labels', 'Colour3_labels', 'Colour4_labels', 'Colour5_labels']].apply(pd.value_counts).fillna(0)

# Plot the distribution of color labels for the specified category
plt.figure(figsize=(10, 6))
color_counts.T.plot(kind='bar', stacked=True, edgecolor='black')
plt.title(f'Category-wise Distribution of Color Labels for {category_of_interest}')
plt.xlabel('Color Labels')
plt.ylabel('Count')
plt.legend(title='Color Labels', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[253]:


category_ = "Children's Books"
category_subset = main_df[main_df['Category'] == category_]

# Count the occurrences of each color label in the subset
color_counts = category_subset[['Colour1_labels', 'Colour2_labels', 'Colour3_labels', 'Colour4_labels', 'Colour5_labels']].apply(pd.value_counts).fillna(0)

# Plot the distribution of color labels for the specified category
plt.figure(figsize=(10, 6))
color_counts.T.plot(kind='bar', stacked=True, edgecolor='black')
plt.title(f'Category-wise Distribution of Color Labels for {category_}')
plt.xlabel('Color Labels')
plt.ylabel('Count')
plt.legend(title='Color Labels', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# We can see that the data is heavily biased for the label 'Red' which could be causing the models to give the same colour regardless the category.

# In[212]:


categories


# In[214]:


for category in categories:
    predicted_labels = []
    for j in range(5):
        column_name = f"Colour{j+1}_labels"
        y = main_df[column_name]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=450)
        label_model_lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=450)
        label_model_lr.fit(X_train, y_train)
        new_X = pd.Series([category])
        new_X_encoded = onehotencoder.transform(pd.DataFrame(new_X))
        new_prediction = label_model_lr.predict(new_X_encoded)
        predicted_labels.append(new_prediction[0])
    print(f"{category}: {predicted_labels}")


# ## PREDICTING TEXT AREA FROM CATEGORY
# 
# Here the X value remains the same. y is the percentage area of text detected by easyOCR.

# In[158]:


y = main_df['Text_Area']


# In[159]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=450)


# ### Linear Regression

# In[160]:


area_model_lr = LinearRegression()
area_model_lr.fit(X_train, y_train)
pred = area_model_lr.predict(X_test)
mse_lr = mean_squared_error(y_test, pred)
mae_lr = mean_absolute_error(y_test, pred)
r2_lr = r2_score(y_test, pred)
print(f'Mean Squared Error (MSE): {mse_lr}')
print(f'Mean Absolute Error (MAE): {mae_lr}')
print(f'R-squared (R2) Score: {r2_lr}')


# ### Decision Tree Regression

# In[161]:


area_model_dt = DecisionTreeRegressor()
area_model_dt.fit(X_train, y_train)
pred = area_model_dt.predict(X_test)
mse_dt = mean_squared_error(y_test, pred)
mae_dt = mean_absolute_error(y_test, pred)
r2_dt = r2_score(y_test, pred)
print(f'Mean Squared Error (MSE): {mse_dt}')
print(f'Mean Absolute Error (MAE): {mae_dt}')
print(f'R-squared (R2) Score: {r2_dt}')


# ### Random Forest Regression

# In[162]:


area_model_rf = RandomForestRegressor()
area_model_rf.fit(X_train, y_train)
pred = area_model_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, pred)
mae_rf = mean_absolute_error(y_test, pred)
r2_rf = r2_score(y_test, pred)
print(f'Mean Squared Error (MSE): {mse_rf}')
print(f'Mean Absolute Error (MAE): {mae_rf}')
print(f'R-squared (R2) Score: {r2_rf}')


# ### Support Vector Regression

# In[165]:


area_model_svr = SVR()
area_model_svr.fit(X_train, y_train)
pred = area_model_svr.predict(X_test)
mse_svr = mean_squared_error(y_test, pred)
mae_svr = mean_absolute_error(y_test, pred)
r2_svr = r2_score(y_test, pred)
print(f'Mean Squared Error (MSE): {mse_svr}')
print(f'Mean Absolute Error (MAE): {mae_svr}')
print(f'R-squared (R2) Score: {r2_svr}')


# Out of the three better performing models, we'll choose the Random Forest regression model for the text area prediction.

# In[167]:


new_X = pd.Series(["Comics & Graphic Novels"])
new_X_encoded = onehotencoder.transform(pd.DataFrame(new_X))
new_prediction = area_model_rf.predict(new_X_encoded)
print(f'Predicted text area for {new_X.iloc[0]}: {new_prediction[0].round(2)}')


# In[168]:


# Testing another category
new_X = pd.Series(["Health, Fitness & Dieting"])
new_X_encoded = onehotencoder.transform(pd.DataFrame(new_X))
new_prediction = area_model_rf.predict(new_X_encoded)
print(f'Predicted text area for {new_X.iloc[0]}: {new_prediction[0].round(2)}')


# In[192]:


category_area = []
for category in categories:
    new_X = pd.Series([category])
    new_X_encoded = onehotencoder.transform(pd.DataFrame(new_X))
    new_prediction = area_model_rf.predict(new_X_encoded)
    category_area.append(new_prediction[0].round(2))
category_area


# In[193]:


area_df = pd.DataFrame({
    'Category': categories, 
    'Predicted Text Area': category_area
})
area_df


# In[196]:


# sorting dataframe based on the predicted text area
area_df = area_df.sort_values(by='Predicted Text Area', ascending=False).reset_index(drop=True)
area_df


# ## PREDICTING OBJECTS FROM CATEGORY

# In[178]:


# Importing required libraries
from sklearn.naive_bayes import MultinomialNB
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D


# In[169]:


main_df.head()


# As many rows in the dataframe does not contain any objects identified, we'll filter out these rows.

# In[171]:


main_df = main_df[main_df['Objects'].apply(lambda x: isinstance(x, list) and any(pd.notna(i) for i in x))]


# In[173]:


main_df.shape


# In[174]:


# Convert list values to strings
main_df['Objects'] = main_df['Objects'].apply(lambda x: ','.join(x))  


# In[175]:


main_df.head()


# In[176]:


# For rows containing more than one objects, only the first one is kept and others are removed as this has the highest
# confidence score by the vision API
# Keeping only the first object identified as it had the most score
main_df['Objects'] = main_df['Objects'].apply(lambda x: x.split(',')[0])


# In[177]:


main_df.head()


# ### Multinomial Naive Bayes Classification

# In[179]:


# Split the data into training and testing sets
train_data, test_data = train_test_split(main_df, test_size=0.2, random_state=450)
# Extract features using TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train = tfidf_vectorizer.fit_transform(train_data['Category'])
X_test = tfidf_vectorizer.transform(test_data['Category'])
# Use Multinomial Naive Bayes as the classifier
object_model_nb = MultinomialNB()
object_model_nb.fit(X_train, train_data['Objects'])
# Make predictions on the test set
predictions = object_model_nb.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(test_data['Objects'], predictions)
print(f'Accuracy: {accuracy:.2f}')
# Display classification report
print('\nClassification Report:\n', classification_report(test_data['Objects'], predictions))


# ### Random Forest Classification

# In[180]:


# Use Random Forest as the classifier
object_model_rf = RandomForestClassifier(n_estimators=100, random_state=450)
object_model_rf.fit(X_train, train_data['Objects'])

# Make predictions on the test set
pred = object_model_rf.predict(X_test)

# Evaluate the Random Forest model
rf_accuracy = accuracy_score(test_data['Objects'], pred)
print(f'Random Forest Accuracy: {rf_accuracy:.2f}')

# Display classification report
print('\nRandom Forest Classification Report:\n', classification_report(test_data['Objects'], pred))


# ## Tensorflow Sequential

# In[182]:


# Tokenize and pad sequences
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(train_data['Category'])
X_train = tokenizer.texts_to_sequences(train_data['Category'])
X_test = tokenizer.texts_to_sequences(test_data['Category'])
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=X_train.shape[1])

# Convert the 'Object' column to numerical labels
label_mapping = {label: idx for idx, label in enumerate(main_df['Objects'].unique())}
train_data['Objects'] = train_data['Objects'].map(label_mapping)
test_data['Objects'] = test_data['Objects'].map(label_mapping)

# Define the model
embedding_dim = 50
model_tf = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=X_train.shape[1]),
    GlobalAveragePooling1D(),
    Dense(128, activation='relu'),
    Dense(len(main_df['Objects'].unique()), activation='softmax')
])

# Compile the model
model_tf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model_tf.fit(X_train, train_data['Objects'], epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
_, nn_accuracy = model_tf.evaluate(X_test, test_data['Objects'])
print(f'Neural Network Accuracy: {nn_accuracy:.2f}')


# Since all three models have the same accuracy, we'll test prediction from two models: MultinomialNB and TensorFlow Sequential.

# In[183]:


category_input = ["Humour & Entertainment"]
# Transforming the input category using the same TF-IDF vectorizer used during training
category_input_transformed = tfidf_vectorizer.transform(category_input)
# Using the Naive Bayes model to predict the object
nb_prediction = object_model_nb.predict(category_input_transformed)
print(f"Naive Bayes Prediction: {nb_prediction[0]}")


# In[184]:


category_input = ["Humour & Entertainment"]
# Tokenizing and padding the input category
category_input_tokenized = tokenizer.texts_to_sequences(category_input)
category_input_padded = tf.keras.preprocessing.sequence.pad_sequences(category_input_tokenized, maxlen=X_train.shape[1])
# Using the Neural Network model to predict the object
nn_prediction = model_tf.predict(category_input_padded)
predicted_class_index = tf.argmax(nn_prediction, axis=1).numpy()[0]
predicted_object = list(label_mapping.keys())[list(label_mapping.values()).index(predicted_class_index)]
print(f"Neural Network Prediction: {predicted_object}")


# We'll get the area and object predictions for all the categories.

# In[197]:


# Predicting objects for all the categories by both naive bayes and tensorflow sequential model
nb_objects = []
tf_objects = []
for category in categories:
    category_input = [category]
    category_input_transformed = tfidf_vectorizer.transform(category_input)
    nb_prediction = object_model_nb.predict(category_input_transformed)
    nb_objects.append(nb_prediction[0])
    category_input_tokenized = tokenizer.texts_to_sequences(category_input)
    category_input_padded = tf.keras.preprocessing.sequence.pad_sequences(category_input_tokenized, maxlen=X_train.shape[1])
    nn_prediction = model_tf.predict(category_input_padded)
    predicted_class_index = tf.argmax(nn_prediction, axis=1).numpy()[0]
    predicted_object = list(label_mapping.keys())[list(label_mapping.values()).index(predicted_class_index)]
    tf_objects.append(predicted_object)


# In[198]:


nb_objects


# In[199]:


tf_objects


# In[200]:


object_df = pd.DataFrame({
    'Category': categories,
    'MultinomialNB Predictions': nb_objects,
    'TF Sequential Predictions': tf_objects
})


# In[201]:


object_df


# In[ ]:




