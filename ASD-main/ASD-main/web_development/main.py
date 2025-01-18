from flask import Flask, Response, app, redirect, render_template, request,jsonify, session, url_for
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
import keras
from PIL import Image
import numpy as np
from skimage import transform
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import tensorflow as tf
import pdfkit
from flask import make_response



#from keras.preprocessing import image
import pickle

#from pickle.chatbot import generate_response, preprocess_text

# classifier = pickle.load(open("C:\\Users\\PRIYANKA A H\\Downloads\\Deployment\\pickle\\trait.pkl", 'rb'))
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.static_folder = r'C:\Users\PRIYANKA A H\Downloads\Deployment\static'


# path_to_wkhtmltopdf = r'C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe'  # Adjust path as necessary
# config = pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf)  Adjust path as necessary


@app.route('/')
def hello():
    return render_template("home.html")

@app.route('/home')
def home():
    return render_template("home.html")

@app.route("/ntest")
def ntest():
    return render_template("ntest.html")

@app.route("/image_dataset")
def image_dataset():
    return render_template("image_dataset.html")

@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contacts")
def contacts():
    return render_template("contacts.html")

# @app.route("/pdf")
# def pdf():
#     return render_template("pdf.html")


@app.route("/report")
def report():
    form_predictions = request.args.get('form_predictions')
    image_predictions = request.args.get('image_predictions')
    # print("report")
    # Render the report template with the prediction results
    rendered_template=render_template("report.html", form_predictions=form_predictions, image_predictions=image_predictions)
    # print("read")
    return rendered_template

# @app.route('/download_report')
# def download_report():
#     # Example HTML content
#     html_content = '<html><head><title>Report</title></head><body><h1>Report Title</h1><p>Report content here...</p></body></html>'
    
#     # Generate PDF from HTML string
#     pdf = pdfkit.from_string(html_content, False, configuration=config)

#     # Generate response
#     response = Response(pdf)
#     response.headers['Content-Type'] = 'application/pdf'
#     response.headers['Content-Disposition'] = 'attachment; filename=report.pdf'
    
#     return response

# @app.route('/download_report')
# def download_report():
#     # You can access predictions stored in session or passed through other means
#     form_predictions = session.get('numerical_predictions')
#     image_predictions = session.get('image_predictions')

    # data = {
    #     "report_title": "Assessment Report",
    #     "form_predictions": form_predictions,
    #     "image_predictions": image_predictions
    # }

    # Render the HTML template with data
    # rendered_html = render_template('pdf.html',form_predictions = form_predictions ,image_predictions= image_predictions)

    # Generate PDF from the rendered HTML string
    pdf = pdfkit.from_string(rendered_html, False, configuration=config)

    # Generate response
    response = Response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=assessment_report.pdf'
    
    return response



#@app.route("/submit_numerical",methods=['post'])
@app.route("/submit_numerical", methods=['POST'])
    
def submit_application():
    classifier = pickle.load(open("C:\\Users\\PRIYANKA A H\\Downloads\\Deployment\\pickle\\trait.pkl", 'rb'))
    A1 = request.form.get("A1")
    A2 = request.form.get("A2")
    A3 = request.form.get("A3")
    A4 = request.form.get("A4")
    A5= request.form.get("A5")
    A6 = request.form.get("A6")
    A7 = request.form.get("A7")
    A8 = request.form.get("A8")
    A9 = request.form.get("A9")
    A10 = request.form.get("A10_Autism_Spectrum_Quotient")
    Social_Responsiveness_Scale = request.form.get("Social_Responsiveness_Scale")
    Age_Years = request.form.get("Age_Years")
    Qchat_10_Scor = request.form.get("Qchat_10_Score")
    Speech_Delay = request.form.get("Speech Delay/Language Disorder")
    Learning_disorder = request.form.get("Learning disorder")
    Genetic_Disorders = request.form.get("Genetic_Disorders")
    Depression = request.form.get("Depression")
    Global_developmental_delay= request.form.get("Global developmental delay/intellectual disability")
    Social_or_Behavioural_Issues = request.form.get("Social/Behavioural Issues")
    Childhood_Autism_Rating_Scale = request.form.get("Childhood Autism Rating Scale")
    Anxiety_disorder = request.form.get("Anxiety_disorder")
    Sex = request.form.get("Sex")
    Jaundice = request.form.get("Jaundice")
    Family_mem_with_ASD = request.form.get("Family_mem_with_ASD")


    input_data=[A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,Social_Responsiveness_Scale,Age_Years,Qchat_10_Scor,Speech_Delay,Learning_disorder,Genetic_Disorders,Depression,Global_developmental_delay,Social_or_Behavioural_Issues,Childhood_Autism_Rating_Scale,Anxiety_disorder,Sex,Jaundice,Family_mem_with_ASD]
    
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

    # if prediction[0] == 0:
    #   return "No possibility of Autism"
    # else:
    #   return "Autism Spectrum possibility"

    # if prediction == 1:
    #   form_predictions = "possibility"
    # else:
    #   form_predictions = "No_possibility"

    # report = {
    #     'form_predictions': form_predictions
    # }
    # return render_template("report.html", report=report)

    if prediction == 1:
        session['numerical_predictions'] = "Based on these results, there is a possibility of autism. Please consult a healthcare professional for further evaluation."
    else:
        session['numerical_predictions'] = "Based on these results, there is no indication of autism. However, if you have concerns, please consult a healthcare professional for further evaluation."

    # Redirect to the image input form
    return redirect(url_for('image_dataset'))

    
photo_size=224
def load_image_from_path(filename):
    #img = mpimg.imread(filename)
    #imgplot = plt.imshow(img)
    #plt.show()
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32') / 255
    np_image = transform.resize(np_image, (photo_size, photo_size, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = tf.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)
    
@app.route("/submit_image", methods=['POST'])
def submit_image():
    # print(request.files)
    model_path = 'C:\\Users\\PRIYANKA A H\\Downloads\\Deployment\\pickle\\inception_model.h5'
    with tf.keras.utils.custom_object_scope({'FixedDropout': FixedDropout}):
        inception = load_model(model_path)
    # print("loaded")

    # Get the image file from the request
    file = request.files['file']
    # print("received:", file.filename)

    # Load and preprocess the uploaded image
    image=load_image_from_path(file)
    # print("image loaded")

    # np_image = np.array(np_image).astype('float32') / 255
    # np_image = transform.resize(np_image, (224, 224, 3))
    # np_image = np.expand_dims(np_image, axis=0)


    # Make predictions on the preprocessed image
    prediction = inception.predict(image).argmax()
    # print(prediction)

    if prediction == 1:
        # print("p")
        image_predictions = "Based on these results, there is a possibility of autism. Please consult a healthcare professional for further evaluation."
    else:
        # print('np')
        image_predictions = "Based on these results, there is no indication of autism. However, if you have concerns, please consult a healthcare professional for further evaluation."
    
    # Retrieve numerical predictions from session
    numerical_predictions = session.get('numerical_predictions')

    return jsonify({'form_predictions':numerical_predictions, 'image_predictions':image_predictions})


# import re
# import os
# import pickle
# from flask import Flask, request, jsonify

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# app = Flask(__name__)

# # Load the trained model from the pickle file
# with open('C:\\Users\\PRIYANKA A H\\Downloads\\Deployment\\feedback_data.pickle', 'rb') as file:
#     pickle.load((np.vectorize, tfidf_matrix_user_inputs, tfidf_matrix_chatbot_outputs), file)


# # Initialize the feedback data dictionary
# feedback_data = {}

# # Check if feedback_data.pickle file exists and load the data
# feedback_file_path = 'feedback_data.pickle'
# if os.path.exists(feedback_file_path):
#     with open(feedback_file_path, 'rb') as handle:
#         feedback_data = pickle.load(handle)

# # Preprocess text function
# def preprocess_text(text):
#     text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
#     return text

# # Generate response to user input function
# def generate_response(user_input, chatbot_outputs):
#     user_input = preprocess_text(user_input)
#     if "asd" in user_input:
#         user_input = user_input.replace("asd", "autism spectrum disorder")

#     user_tfidf = vectorizer.transform([user_input])
#     similarity_scores = cosine_similarity(user_tfidf, tfidf_matrix_user_inputs)
#     max_sim_idx = similarity_scores.argmax()
#     threshold = 0.2
#     if similarity_scores[0][max_sim_idx] < threshold:
#         return "Sorry, I couldn't understand. Could you please rephrase your question?"
#     else:
#         return chatbot_outputs[max_sim_idx]

# # Endpoint for chatbot
# @app.route('/chatbot', methods=['POST'])
# def chatbot():
#     data = request.get_json()
#     user_input = data['user_input']
#     response = generate_response(user_input)
#     return jsonify({'response': response})

# # Endpoint for collecting feedback
# @app.route('/feedback', methods=['POST'])
# def collect_feedback():
#     data = request.get_json()
#     user_input = preprocess_text(data['user_input'])
#     bot_response = preprocess_text(data['bot_response'])
#     feedback = data['feedback']

#     if user_input not in feedback_data:
#         feedback_data[user_input] = {'bot_responses': [], 'feedbacks': []}
#     feedback_data[user_input]['bot_responses'].append(bot_response)
#     feedback_data[user_input]['feedbacks'].append(feedback)

#     # Save updated feedback data to pickle file
#     with open(feedback_file_path, 'wb') as handle:
#         pickle.dump(feedback_data, handle, protocol=pickle.HIGHEST_PROTOCOL/)

#     return jsonify({'message': 'Feedback received successfully'})




app.run(debug=True)