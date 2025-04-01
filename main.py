from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import numpy as np
import shutil
import os

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

fire_detection = load_model('FireDetection.h5')

@app.get("/", response_class=HTMLResponse)
async def upload_form():
    content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Intelligent Fire Detector</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/toastify-js/src/toastify.min.css">
    <link rel="icon" href="/static/favicon.ico" type="image/x-icon">
    <style>
        body {
            background-image: url('/static/bg1.png');
            background-size: cover;
            background-position: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        @media (max-width: 1050px) { body { background-image: url('/static/bg2.png'); } }
        @media (max-width: 500px) { body { background-image: url('/static/bg3.png'); } }
        .navbar {
            width: 100%;
            padding: 15px;
            background: rgba(52, 58, 64, 0.9);
            backdrop-filter: blur(5px);
        }
        .upload-container {
            background: rgba(255, 255, 255, 0.9);
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
            animation: fadeIn 1s ease-in-out;
            max-width: 500px;
            width: 100%;
        }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        .btn {
            border-radius: 25px;
            padding: 10px 15px;
        }
        #preview {
            max-width: 100%;
            border-radius: 10px;
            display: none;
        }
        .loader {
            display: none;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-dark">
        <div class="container text-center">
            <a href="/" class="navbar-brand">Intelligent Fire Detector</a>
        </div>
    </nav>
    <div class="container d-flex justify-content-center align-items-center flex-grow-1">
        <div class="upload-container text-center">
            <h3>Upload an Image</h3>
            <p>Accepted: PNG, JPG, JPEG, WEBP</p>
            <form id="upload-form" enctype="multipart/form-data">
                <input type="file" class="form-control mb-3" id="file" name="file" accept="image/*" required>
                <img id="preview" src="#" alt="Preview" class="mb-3"/>
                <div class="loader spinner-border text-primary" role="status"></div>
                <button type="submit" class="btn btn-primary w-100">Upload</button>
                <button type="button" class="btn btn-secondary w-100 mt-2 d-none" id="clear-btn">Clear</button>
            </form>
            <div id="response-message" class="alert mt-3 d-none"></div>
        </div>
    </div>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/toastify-js"></script>
    <script>
        $(document).ready(function() {
            $('#file').change(function() {
                let file = this.files[0];
                if (file) {
                    let reader = new FileReader();
                    reader.onload = function(e) {
                        $('#preview').attr('src', e.target.result).show();
                    }
                    reader.readAsDataURL(file);
                }
            });
            $('#upload-form').submit(function(e) {
                e.preventDefault();
                let formData = new FormData(this);
                $('.loader').show();
                $.ajax({
                    url: '/upload/',
                    type: 'POST',
                    data: formData,
                    contentType: false,
                    processData: false,
                    success: function(response) {
                        $('.loader').hide();
                        let msg = response.predictions.classification + ' detected';
                        let alertClass = response.predictions.classification === 'No Fire' ? 'alert-success' : 'alert-danger';
                        $('#response-message').removeClass('d-none alert-success alert-danger').addClass(alertClass).text(msg);
                        Toastify({
                            text: msg,
                            backgroundColor: alertClass === 'alert-success' ? '#28a745' : '#dc3545',
                            duration: 3000
                        }).showToast();
                    },
                    error: function() {
                        $('.loader').hide();
                        $('#response-message').removeClass('d-none').addClass('alert-danger').text('Error uploading image');
                    }
                });
            });
            $('#clear-btn').click(function() {
                $('#file').val('');
                $('#preview').hide();
                $('#response-message').addClass('d-none');
            });
        });
    </script>
</body>
</html>

    """
    return HTMLResponse(content=content)



@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload an image file.")
    
    temp_file_path = "temp_uploaded_image.png"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        img = Image.open(temp_file_path).convert('RGB')
        img = img.resize((256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
    except Exception as e:
        os.remove(temp_file_path)  # Clean up even if an error occurs
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")
    
    try:
        fire_detection_pred = fire_detection.predict(img_array)[0][0]
    
        fireOrNofire = 'No Fire' if fire_detection_pred > 0.5 else 'Fire'

        predictions_list = {"classification": fireOrNofire}

    except Exception as e:
        os.remove(temp_file_path)  # Ensure cleanup on prediction error
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")
        
    os.remove(temp_file_path)  # Delete the temporary file after processing

    return {"predictions": predictions_list}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8100)
