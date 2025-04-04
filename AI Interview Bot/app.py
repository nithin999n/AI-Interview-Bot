from fastapi import FastAPI, WebSocket, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import cv2
import mediapipe as mp
import numpy as np
import json
import asyncio
from typing import List
import base64
import time

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Initialize MediaPipe Face Mesh with improved settings
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True
)

# Key facial landmarks
MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
MOUTH_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYEBROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
RIGHT_EYEBROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]

# Interview questions
QUESTIONS = [
    "Tell me about yourself.",
    "What are your greatest strengths?",
    "Where do you see yourself in five years?",
    "Why should we hire you?",
    "What are your career goals?"
]

class InterviewManager:
    def __init__(self):
        self.current_question = -1
        self.name = None
        self.waiting_for_name = True
        self.last_response_time = time.time()
        self.last_question_time = time.time()
        self.processing_response = False
        self.last_emotion = None
        self.emotion_count = {}
        self.last_eye_contact = 50.0
        self.speaking = False
        self.face_detected = False

interview_manager = InterviewManager()

def get_landmark_points(face_landmarks, indices, image_shape):
    points = []
    for idx in indices:
        landmark = face_landmarks.landmark[idx]
        x = int(landmark.x * image_shape[1])
        y = int(landmark.y * image_shape[0])
        points.append((x, y))
    return np.array(points)

def calculate_mouth_aspect_ratio(mouth_points):
    # Calculate vertical distances
    vert_dists = []
    for i in range(3):
        vert_dists.append(np.linalg.norm(mouth_points[i] - mouth_points[-i-1]))
    
    # Calculate horizontal distances
    horz_dists = []
    for i in range(5, 8):
        horz_dists.append(np.linalg.norm(mouth_points[i] - mouth_points[-i-1]))
    
    mar = np.mean(vert_dists) / np.mean(horz_dists)
    return mar

def calculate_eye_aspect_ratio(eye_points):
    # Calculate vertical distances
    vert_dist1 = np.linalg.norm(eye_points[1] - eye_points[-1])
    vert_dist2 = np.linalg.norm(eye_points[2] - eye_points[-2])
    
    # Calculate horizontal distance
    horz_dist = np.linalg.norm(eye_points[0] - eye_points[8])
    
    ear = (vert_dist1 + vert_dist2) / (2.0 * horz_dist)
    return ear

def calculate_emotion(face_landmarks, image_shape):
    try:
        # Get facial feature points
        mouth_outer = get_landmark_points(face_landmarks, MOUTH_OUTER, image_shape)
        mouth_inner = get_landmark_points(face_landmarks, MOUTH_INNER, image_shape)
        left_eye = get_landmark_points(face_landmarks, LEFT_EYE, image_shape)
        right_eye = get_landmark_points(face_landmarks, RIGHT_EYE, image_shape)
        left_brow = get_landmark_points(face_landmarks, LEFT_EYEBROW, image_shape)
        right_brow = get_landmark_points(face_landmarks, RIGHT_EYEBROW, image_shape)

        # Calculate metrics
        mouth_ratio = calculate_mouth_aspect_ratio(mouth_outer)
        inner_mouth_ratio = calculate_mouth_aspect_ratio(mouth_inner)
        left_eye_ratio = calculate_eye_aspect_ratio(left_eye)
        right_eye_ratio = calculate_eye_aspect_ratio(right_eye)
        
        # Calculate eyebrow position and asymmetry
        left_brow_height = np.mean(left_brow[:, 1])
        right_brow_height = np.mean(right_brow[:, 1])
        brow_asymmetry = abs(left_brow_height - right_brow_height)
        avg_brow_height = (left_brow_height + right_brow_height) / 2
        
        # Emotion classification with improved thresholds
        if mouth_ratio > 0.7 and inner_mouth_ratio > 0.5:  # Wide open mouth
            if avg_brow_height < np.mean(image_shape) * 0.3:
                return "Surprised"
        elif mouth_ratio > 0.5:  # Smile
            if inner_mouth_ratio < 0.3:  # Closed smile
                return "Happy"
            else:  # Open smile
                return "Happy"
        elif brow_asymmetry > 10:  # Asymmetric expression
            return "Confused"
        elif avg_brow_height < np.mean(image_shape) * 0.25:  # Low brows
            if mouth_ratio < 0.3:  # Tight mouth
                return "Angry"
        elif mouth_ratio < 0.3:  # Downturned mouth
            return "Sad"
        
        return "Neutral"
    except Exception as e:
        print(f"Emotion calculation error: {e}")
        return None

def calculate_eye_gaze(face_landmarks, image_shape):
    try:
        # Get eye points
        left_eye = get_landmark_points(face_landmarks, LEFT_EYE, image_shape)
        right_eye = get_landmark_points(face_landmarks, RIGHT_EYE, image_shape)
        
        # Calculate eye centers
        left_center = np.mean(left_eye, axis=0)
        right_center = np.mean(right_eye, axis=0)
        
        # Calculate iris positions
        left_iris = np.mean(left_eye[8:12], axis=0)  # Using inner eye points
        right_iris = np.mean(right_eye[8:12], axis=0)
        
        # Calculate relative iris positions
        left_rel_x = (left_iris[0] - left_center[0]) / (np.max(left_eye[:, 0]) - np.min(left_eye[:, 0]))
        right_rel_x = (right_iris[0] - right_center[0]) / (np.max(right_eye[:, 0]) - np.min(right_eye[:, 0]))
        
        left_rel_y = (left_iris[1] - left_center[1]) / (np.max(left_eye[:, 1]) - np.min(left_eye[:, 1]))
        right_rel_y = (right_iris[1] - right_center[1]) / (np.max(right_eye[:, 1]) - np.min(right_eye[:, 1]))
        
        # Calculate gaze scores
        horizontal_score = 100 - (abs(left_rel_x - 0.5) + abs(right_rel_x - 0.5)) * 100
        vertical_score = 100 - (abs(left_rel_y - 0.5) + abs(right_rel_y - 0.5)) * 100
        
        # Combine scores with emphasis on horizontal gaze
        gaze_score = horizontal_score * 0.7 + vertical_score * 0.3
        
        # Normalize and smooth
        gaze_score = max(0, min(100, gaze_score))
        return gaze_score
        
    except Exception as e:
        print(f"Eye gaze calculation error: {e}")
        return None

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Reset interview state
    interview_manager.current_question = -1
    interview_manager.waiting_for_name = True
    interview_manager.name = None
    interview_manager.last_response_time = time.time()
    interview_manager.last_question_time = time.time()
    interview_manager.processing_response = False
    interview_manager.last_emotion = None
    interview_manager.emotion_count = {}
    interview_manager.last_eye_contact = 50.0
    interview_manager.speaking = False
    interview_manager.face_detected = False
    
    try:
        # Send initial greeting
        interview_manager.speaking = True
        await websocket.send_json({
            "message": "Hello! Please tell me your name to begin the interview."
        })
        await asyncio.sleep(2)  # Wait for TTS to finish
        interview_manager.speaking = False
        
        while True:
            data = await websocket.receive()
            current_time = time.time()
            
            if "text" in str(data):
                # Handle speech input
                message_data = json.loads(data["text"])
                if message_data["type"] == "speech" and not interview_manager.speaking:
                    text = message_data["text"].strip()
                    
                    # Prevent processing multiple responses too quickly
                    if current_time - interview_manager.last_response_time < 2:
                        continue
                        
                    interview_manager.last_response_time = current_time
                    
                    if interview_manager.waiting_for_name:
                        interview_manager.name = text
                        interview_manager.waiting_for_name = False
                        interview_manager.speaking = True
                        
                        # Add delay before first question
                        await asyncio.sleep(1)
                        await websocket.send_json({
                            "message": f"Nice to meet you, {interview_manager.name}! Let's begin the interview. Here's your first question: {QUESTIONS[0]}"
                        })
                        await asyncio.sleep(2)  # Wait for TTS to finish
                        
                        interview_manager.current_question = 0
                        interview_manager.last_question_time = current_time
                        interview_manager.speaking = False
                    else:
                        # Only process response if enough time has passed
                        if not interview_manager.processing_response and current_time - interview_manager.last_question_time > 3:
                            interview_manager.processing_response = True
                            interview_manager.speaking = True
                            
                            # Add delay to simulate thinking
                            await asyncio.sleep(1.5)
                            
                            # Move to next question
                            interview_manager.current_question += 1
                            if interview_manager.current_question < len(QUESTIONS):
                                await websocket.send_json({
                                    "message": f"Thank you for your answer. Next question: {QUESTIONS[interview_manager.current_question]}"
                                })
                            else:
                                await websocket.send_json({
                                    "message": "Thank you for completing the interview! You did great!"
                                })
                            
                            await asyncio.sleep(2)  # Wait for TTS to finish
                            interview_manager.last_question_time = current_time
                            interview_manager.processing_response = False
                            interview_manager.speaking = False
            
            elif "bytes" in str(data):
                # Handle video frame
                frame_data = data["bytes"]
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Process with MediaPipe
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb_frame)
                    
                    if results.multi_face_landmarks:
                        face_landmarks = results.multi_face_landmarks[0]
                        interview_manager.face_detected = True
                        
                        # Calculate emotion with smoothing
                        emotion = calculate_emotion(face_landmarks, frame.shape)
                        if emotion:
                            if emotion not in interview_manager.emotion_count:
                                interview_manager.emotion_count[emotion] = 0
                            interview_manager.emotion_count[emotion] += 1
                            
                            # Only update emotion if it's stable for a few frames
                            max_emotion = max(interview_manager.emotion_count.items(), key=lambda x: x[1])[0]
                            if interview_manager.emotion_count[max_emotion] > 3:  # Reduced stability requirement
                                if max_emotion != interview_manager.last_emotion:
                                    interview_manager.last_emotion = max_emotion
                                    await websocket.send_json({"emotion": max_emotion})
                                interview_manager.emotion_count.clear()
                        
                        # Calculate eye contact with smoothing
                        eye_contact = calculate_eye_gaze(face_landmarks, frame.shape)
                        if eye_contact is not None:
                            # Smooth eye contact value with more weight on new values
                            interview_manager.last_eye_contact = (
                                0.6 * interview_manager.last_eye_contact +
                                0.4 * eye_contact
                            )
                            await websocket.send_json({
                                "eye_contact": round(interview_manager.last_eye_contact, 1)
                            })
                    else:
                        if interview_manager.face_detected:
                            interview_manager.face_detected = False
                            await websocket.send_json({"emotion": "Unknown"})
                            await websocket.send_json({"eye_contact": 0})
    
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print("WebSocket connection closed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
