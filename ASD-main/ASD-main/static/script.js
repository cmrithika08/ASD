document.getElementById('predictButton').onclick = function() {
    var imageInput = document.getElementById('imageInput');
    if(imageInput.files.length > 0) {
        var file = imageInput.files[0];
        
        // Example: Display the file name in the console
        console.log("File chosen:", file.name);

        // Here you'd typically use FormData to append the file
        // and then send it to your server for prediction via fetch() or XMLHttpRequest
        // This is a placeholder for the server interaction
        // Replace URL with your actual endpoint
        var formData = new FormData();
        formData.append('file', file);

        fetch('/submit_image', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // console.log(data);
            // // Display the prediction result
            // document.getElementById('predictionResult').innerText = "Numerical Prediction: " + data.form_predictions + "<br>Image Predictions: "+data.image_predictions;
            window.location.href = '/report?form_predictions=' + data.form_predictions + '&image_predictions=' + data.image_predictions;
        })
        .catch(error => console.error(error));
    } else {
        alert("Please select an image first.");
    }
};

document.getElementById('imageInput').addEventListener('change', function(event) {
    const file = event.target.files[0];
    const reader = new FileReader();

    reader.onload = function(e) {
        const imgElement = document.getElementById('uploadedImage');
        imgElement.style.display = 'block';
        imgElement.src = e.target.result;
        document.getElementById('uploadText').textContent = 'Change Image';
    };

    reader.readAsDataURL(file);
});

function sendMessage() {
    var userInput = $('#user_input').val();
    $('#chat-list').append('<li>You: ' + userInput + '</li>');
    $('#user_input').val('');

    $.post('/chatbot', { user_input: userInput }, function(data) {
        $('#chat-list').append('<li>Bot: ' + data.response + '</li>');
    });
}

// document.addEventListener("DOMContentLoaded", function() {
//     document.getElementById("predictButton").addEventListener("click", function() {
//         // Show the loading bar
//         document.getElementById("loadingBar").style.display = "block";

//         // Make an AJAX request to submit the image and get predictions
//         var formData = new FormData();
//         var imageInput = document.getElementById("imageInput").files[0];
//         formData.append("file", imageInput);

//         var xhr = new XMLHttpRequest();
//         xhr.open("POST", "/submit_image");
//         xhr.onload = function() {
//             if (xhr.status === 200) {
//                 var response = JSON.parse(xhr.responseText);
//                 // Hide the loading bar
//                 document.getElementById("loadingBar").style.display = "none";
//                 // Display the prediction results
//                 document.getElementById("predictionResult").innerHTML = "<p>" + response.form_predictions + "</p><p>" + response.image_predictions + "</p>";
//             } else {
//                 // Handle errors
//                 console.error("Request failed. Status: " + xhr.status);
//             }
//         };
//         xhr.send(formData);
//     });
// });