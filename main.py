from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status,Header
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import StreamingResponse
from datetime import datetime, timedelta, timezone
from typing import Union
import jwt
from jwt import PyJWTError
from passlib.context import CryptContext
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import io
import torch
import timm
import torchvision.transforms as transforms
from PIL import ImageFont, ImageDraw, Image
from torchvision import transforms
import pandas as pd
from fastapi.responses import FileResponse

# JWT settings
SECRET_KEY = "fdb3e44ba75f4d770ee8de98e488bc3ebcf64dc3066c8140a1ae620c30964454"  # Replace with your own secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Define the mapping of predicted class indices to names
class_mapping = {
    0: 'Angelina Jolie', 1: 'Brad Pitt', 2: 'Denzel Washington',
    3: 'Hugh Jackman', 4: 'Jennifer Lawrence', 5: 'Johnny Depp',
    6: 'Kate Winslet', 7: 'Leonardo DiCaprio', 8: 'Megan Fox',
    9: 'Natalie Portman', 10: 'Nicole Kidman', 11: 'Robert Downey Jr',
    12: 'Sandra Bullock', 13: 'Scarlett Johansson', 14: 'Tom Cruise',
    15: 'Tom Hanks', 16: 'Will Smith'
}

# User database (mock)
users_db = {
    "admin": {"username": "admin", "password": pwd_context.hash("adminpass"), "role": "admin"},
    "user": {"username": "user", "password": pwd_context.hash("userpass"), "role": "user"},
}

# Load the face detection model (YOLOv8)
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
face_model = YOLO(model_path)

# Load the face classification model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("rexnet_150", pretrained=False, num_classes=len(class_mapping))
model.load_state_dict(torch.load('./models/faces_best_model.pth', map_location=device))
model.eval()
model.to(device)

# Define image transformation for classification
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
])

app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# Helper functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def authenticate_user(username: str, password: str):
    user = users_db.get(username)
    if not user or not verify_password(password, user["password"]):
        return False
    return user


def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return username
    except PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


@app.get("/")
def index():
    return {"message": "Welcome to the Face Recognition FastAPI app!"}


# Token Generation Endpoint
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/predict/{door_number}")
async def face_recognition(door_number, file: UploadFile = File(...), token: str = Depends(oauth2_scheme)):
    # Load the image
    image = Image.open(file.file).convert("RGB")
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(image)

    results = face_model(image)

    if not results or len(results[0].boxes) == 0:
        raise HTTPException(status_code=400, detail="No faces detected in the image.")

    try:
        font_path = "/Library/Fonts/Arial.ttf"  # Update this path according to your system
        font = ImageFont.truetype(font_path, size=24)
    except IOError:
        font = ImageFont.load_default()  

    # Process only if faces are detected
    draw = ImageDraw.Draw(image)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  
        
        # Crop the detected face from the image
        face_image = image.crop((x1, y1, x2, y2))
        face_tensor = transform(face_image).unsqueeze(0).to(device)

        with torch.no_grad():
            # Get the raw model output (logits)
            output = model(face_tensor)
            
            output = torch.exp(output - torch.max(output))  
            output = output / output.sum(dim=1, keepdim=True) 
            predicted_class = torch.argmax(output, dim=1).item()

            confidence, _ = torch.max(output, dim=1)
            confidence = confidence.item()  


        if confidence <0.95:
            predicted_name = "Uknown Face"
        else:
            predicted_name = class_mapping.get(predicted_class)
        
        # Draw bounding box and label on the original image
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1), predicted_name, fill="white", font=font)

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)  

    return StreamingResponse(
    img_byte_arr, 
    media_type="image/jpeg", 
    headers={
        "Predicted-Door-Num": str(predicted_class),  
        "X-Confidence": str(confidence)  
    }
)


