import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np

classifier = pickle.load(open("trait.pkl", 'rb'))
input_data = [1,1,0,0,0,1,1,0,0,0,6,3,4,1,1,1,1,1,1,2,1,'M',1,0]

# Define label encoders for categorical variables
le_sex = LabelEncoder()

# Assuming 'Sex' is the 22nd feature in your data
input_data[21] = le_sex.fit_transform([input_data[21]])[0]

# Convert to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Make predictions
prediction = classifier.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print("No possibility of Autism")
else:
    print("Autism Spectrum possibility")


