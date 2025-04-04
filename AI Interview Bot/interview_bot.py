import cv2
import mediapipe as mp
import pyttsx3
import speech_recognition as sr
import random
import time
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
import numpy as np
import math

class InterviewBot:
    def __init__(self):
        # Initialize main window
        self.window = ctk.CTk()
        self.window.title("AI Interview Bot")
        self.window.geometry("1200x800")
        ctk.set_appearance_mode("dark")
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # Initialize speech recognition with more sensitive settings
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold = 100
        self.recognizer.pause_threshold = 1.0
        self.recognizer.phrase_threshold = 0.3
        self.recognizer.non_speaking_duration = 0.5
        
        # Initialize face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define landmark indices
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.MOUTH = [61, 291, 0, 17, 269, 405, 321, 375, 78, 308, 324, 318, 402, 317, 14, 87]
        self.LEFT_EYEBROW = [276, 283, 282, 295, 285]
        self.RIGHT_EYEBROW = [46, 53, 52, 65, 55]
        
        # Initialize variables for metrics
        self.emotion = "Unknown"
        self.eye_gaze_percentage = 0
        self.last_emotion_time = time.time()
        self.emotion_cooldown = 0.2  # Seconds between emotion updates
        
        # Interview questions
        self.questions = [
            "Tell me about yourself.",
            "What are your greatest strengths?",
            "Where do you see yourself in five years?",
            "Why should we hire you?",
            "What are your career goals?"
        ]
        self.current_question = 0
        self.name = ""
        self.running = True
        
        # Initialize microphone
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
        except Exception as e:
            print(f"Microphone initialization error: {e}")
        
        # Initialize UI elements
        self.setup_ui()
        
    def setup_ui(self):
        # Create main frames
        self.left_frame = ctk.CTkFrame(self.window, width=800)
        self.left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        self.right_frame = ctk.CTkFrame(self.window, width=400)
        self.right_frame.pack(side="right", fill="both", padx=10, pady=10)
        
        # Video frame
        self.video_label = ctk.CTkLabel(self.left_frame, text="")
        self.video_label.pack(padx=10, pady=10)
        
        # Metrics frame
        self.metrics_frame = ctk.CTkFrame(self.left_frame)
        self.metrics_frame.pack(fill="x", padx=10, pady=5)
        
        # Emotion label
        self.emotion_label = ctk.CTkLabel(
            self.metrics_frame,
            text="Emotion: Unknown",
            font=("Helvetica", 16)
        )
        self.emotion_label.pack(pady=5)
        
        # Eye gaze progress bar
        self.gaze_label = ctk.CTkLabel(
            self.metrics_frame,
            text="Eye Contact:",
            font=("Helvetica", 16)
        )
        self.gaze_label.pack(pady=2)
        
        self.gaze_progress = ctk.CTkProgressBar(
            self.metrics_frame,
            width=300,
            height=20,
            border_width=2,
            progress_color="green"
        )
        self.gaze_progress.pack(pady=5)
        self.gaze_progress.set(0)
        
        # Question display
        self.question_label = ctk.CTkLabel(
            self.right_frame,
            text="Welcome to AI Interview",
            font=("Helvetica", 16),
            wraplength=350
        )
        self.question_label.pack(padx=20, pady=20)
        
        # Response display
        self.response_text = ctk.CTkTextbox(
            self.right_frame,
            width=350,
            height=300,
            font=("Helvetica", 14)
        )
        self.response_text.pack(padx=20, pady=10)
        
        # Status label
        self.status_label = ctk.CTkLabel(
            self.right_frame,
            text="Status: Ready",
            font=("Helvetica", 14)
        )
        self.status_label.pack(padx=20, pady=10)
        
        # Control buttons
        self.start_button = ctk.CTkButton(
            self.right_frame,
            text="Start Interview",
            command=self.start_interview_thread,
            font=("Helvetica", 14)
        )
        self.start_button.pack(padx=20, pady=10)
        
        self.quit_button = ctk.CTkButton(
            self.right_frame,
            text="Quit",
            command=self.quit_app,
            fg_color="red",
            hover_color="darkred",
            font=("Helvetica", 14)
        )
        self.quit_button.pack(padx=20, pady=10)
        
    def calculate_emotion(self, face_landmarks, image_shape):
        # Get facial feature points
        def get_points(indices):
            points = []
            for idx in indices:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * image_shape[1])
                y = int(landmark.y * image_shape[0])
                points.append((x, y))
            return np.array(points)
        
        mouth_points = get_points(self.MOUTH)
        left_eyebrow = get_points(self.LEFT_EYEBROW)
        right_eyebrow = get_points(self.RIGHT_EYEBROW)
        
        # Calculate metrics
        mouth_height = np.mean([abs(mouth_points[i][1] - mouth_points[i+8][1]) for i in range(8)])
        mouth_width = abs(mouth_points[0][0] - mouth_points[6][0])
        mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
        
        eyebrow_height_left = np.mean([p[1] for p in left_eyebrow])
        eyebrow_height_right = np.mean([p[1] for p in right_eyebrow])
        eyebrow_height = (eyebrow_height_left + eyebrow_height_right) / 2
        
        # Determine emotion based on facial metrics
        if mouth_ratio > 0.5:  # Open mouth
            if eyebrow_height < image_shape[0] * 0.3:  # Raised eyebrows
                return "Surprised"
            else:
                return "Happy"
        elif mouth_ratio < 0.2:  # Tight mouth
            if eyebrow_height < image_shape[0] * 0.3:  # Raised eyebrows
                return "Angry"
            else:
                return "Sad"
        else:
            if abs(eyebrow_height_left - eyebrow_height_right) > 10:  # Asymmetric eyebrows
                return "Confused"
            else:
                return "Neutral"
    
    def calculate_eye_gaze(self, face_landmarks, image_shape):
        def get_eye_coordinates(eye_points):
            coordinates = []
            for point in eye_points:
                landmark = face_landmarks.landmark[point]
                x = int(landmark.x * image_shape[1])
                y = int(landmark.y * image_shape[0])
                coordinates.append((x, y))
            return np.array(coordinates)
        
        left_eye = get_eye_coordinates(self.LEFT_EYE)
        right_eye = get_eye_coordinates(self.RIGHT_EYE)
        
        # Calculate eye centers
        left_center = np.mean(left_eye, axis=0)
        right_center = np.mean(right_eye, axis=0)
        
        # Calculate eye direction vector
        eye_vector = right_center - left_center
        
        # Calculate gaze percentage based on eye symmetry and position
        ideal_ratio = 0.9
        current_ratio = abs(eye_vector[1] / eye_vector[0]) if eye_vector[0] != 0 else 1
        gaze_percentage = max(0, min(100, (1 - abs(current_ratio - ideal_ratio)) * 100))
        
        return gaze_percentage
    
    def update_video_feed(self):
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process face mesh
                results = self.face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Draw face mesh points
                        for landmark in face_landmarks.landmark:
                            h, w, _ = frame.shape
                            x, y = int(landmark.x * w), int(landmark.y * h)
                            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                        
                        # Calculate metrics
                        self.eye_gaze_percentage = self.calculate_eye_gaze(face_landmarks, frame.shape)
                        self.gaze_progress.set(self.eye_gaze_percentage / 100)
                        
                        if time.time() - self.last_emotion_time > self.emotion_cooldown:
                            self.emotion = self.calculate_emotion(face_landmarks, frame.shape)
                            self.last_emotion_time = time.time()
                        
                        # Update UI
                        self.emotion_label.configure(text=f"Emotion: {self.emotion}")
                        
                        # Draw metrics on frame
                        cv2.putText(frame, f"Emotion: {self.emotion}", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(frame, f"Eye Contact: {int(self.eye_gaze_percentage)}%",
                                  (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Convert to PIL format
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image = image.resize((780, 580))
                photo = ImageTk.PhotoImage(image=image)
                self.video_label.configure(image=photo)
                self.video_label.image = photo
            
            self.window.after(10, self.update_video_feed)
    
    def speak(self, text):
        self.status_label.configure(text="Status: Speaking...")
        self.response_text.insert("end", f"Bot: {text}\n")
        self.response_text.see("end")
        self.engine.say(text)
        self.engine.runAndWait()
        self.status_label.configure(text="Status: Listening...")
        
    def listen(self):
        try:
            with sr.Microphone() as source:
                self.status_label.configure(text="Status: Listening...")
                self.response_text.insert("end", "Listening... (Please speak clearly)\n")
                self.response_text.see("end")
                
                # Quick ambient noise adjustment before each listen
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Increased timeout and phrase_time_limit for more lenient listening
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=15)
                
                self.status_label.configure(text="Status: Processing speech...")
                # Using a more lenient recognition model
                text = self.recognizer.recognize_google(audio, language='en-US')
                
                if text:
                    self.response_text.insert("end", f"Recognized: {text}\n")
                    self.response_text.see("end")
                    return text.lower()
                return ""
                
        except sr.WaitTimeoutError:
            self.response_text.insert("end", "No speech detected. Please speak louder and try again.\n")
            self.response_text.see("end")
            return ""
        except sr.UnknownValueError:
            self.response_text.insert("end", "Could not understand audio. Please speak more clearly.\n")
            self.response_text.see("end")
            return ""
        except sr.RequestError as e:
            self.response_text.insert("end", f"Could not request results; {e}\n")
            self.response_text.see("end")
            return ""
        except Exception as e:
            self.response_text.insert("end", f"Error: {str(e)}\n")
            self.response_text.see("end")
            return ""

    def get_name(self):
        max_attempts = 5  # Increased number of attempts
        attempts = 0
        
        self.speak("I'll be listening for your name. Please speak clearly and a bit louder than normal.")
        time.sleep(1)  # Give a moment before starting to listen
        
        while attempts < max_attempts:
            self.speak("Please tell me your name.")
            name = self.listen()
            
            if name:
                self.name = name.title()
                self.speak(f"Hello {self.name}! Nice to meet you. Let's begin the interview.")
                return True
            
            attempts += 1
            if attempts < max_attempts:
                self.speak("I didn't catch that. Please speak a bit louder and more clearly.")
                time.sleep(0.5)  # Brief pause before next attempt
        
        self.speak("I'm having trouble hearing your voice. Please check your microphone settings and make sure it's not muted.")
        return False
    
    def start_interview_thread(self):
        self.start_button.configure(state="disabled")
        threading.Thread(target=self.run_interview).start()
    
    def run_interview(self):
        # Start camera
        self.cap = cv2.VideoCapture(0)
        self.update_video_feed()
        
        if not self.get_name():
            self.speak("I couldn't hear your name. Let's try again later.")
            self.quit_app()
            return
        
        while self.running and self.current_question < len(self.questions):
            self.question_label.configure(text=self.questions[self.current_question])
            self.speak(self.questions[self.current_question])
            response = self.listen()
            
            if response:
                self.response_text.insert("end", f"{self.name}: {response}\n")
                self.response_text.see("end")
                self.current_question += 1
                time.sleep(1)
        
        if self.running:
            self.speak("Thank you for completing the interview. Have a great day!")
        
        self.quit_app()
    
    def quit_app(self):
        self.running = False
        if hasattr(self, 'cap'):
            self.cap.release()
        self.window.quit()
    
    def start(self):
        self.window.mainloop()

if __name__ == "__main__":
    bot = InterviewBot()
    bot.start()
