import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf  # Needed for TFLite interpreter
import pickle
import time
from scipy.spatial import distance as dist

# Load TFLite model and tokenizer
interpreter = tf.lite.Interpreter(model_path="model2.tflite")
interpreter.allocate_tensors()

with open('tokenizer2.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Get input/output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

EYE_AR_THRESH = 0.3
DOT_DURATION = 0.2    #Initially 0.75
DASH_DURATION = 0.7   #Intially 1.5
NEXT_CHAR_DELAY = 2.5   #Initally 3


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_blinks_and_gaze():
    cap = cv2.VideoCapture(0)
    blink_sequence = ""
    decoded_text = ""
    final_text = ""
    blink_start = None
    last_blink_time = time.time()
    blink_sequence_2 = ""
    selected_word = ""
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                left_eye = np.array([(landmarks[i].x, landmarks[i].y) for i in [362, 385, 387, 263, 373, 380]])
                right_eye = np.array([(landmarks[i].x, landmarks[i].y) for i in [33, 160, 158, 133, 153, 144]])
                
                left_EAR = eye_aspect_ratio(left_eye)
                right_EAR = eye_aspect_ratio(right_eye)
                ear = (left_EAR + right_EAR) / 2.0
                
                if ear < EYE_AR_THRESH:
                    if blink_start is None:
                        blink_start = time.time()
                else:
                    if blink_start is not None:
                        blink_duration = time.time() - blink_start
                        if blink_duration >= DASH_DURATION:
                            blink_sequence += "-"
                            blink_sequence_2 += "-"
                        elif blink_duration >= DOT_DURATION:
                            blink_sequence += "."
                            blink_sequence_2 += "."
                        blink_start = None
                        last_blink_time = time.time()
                
                if time.time() - last_blink_time >= NEXT_CHAR_DELAY and ('.' in blink_sequence or '-' in blink_sequence):
                    if len(blink_sequence) > 0:
                        final_text += decode_morse(blink_sequence)
                        blink_sequence = ""
                        last_blink_time = time.time()
                
                
                decoded_text = decode_morse(blink_sequence)
                
                if len(final_text) > 0 and final_text[-1] == ' ':
                    predicted_word = Predict_Next_Words(interpreter, tokenizer, final_text)
                    selected_word = predicted_word
                
                if blink_sequence == ".....":
                    final_text += selected_word + " "
                    predicted_word = Predict_Next_Words(interpreter, tokenizer, final_text)
                    selected_word = predicted_word
                    blink_sequence = ""
                    blink_sequence_2 = ""
                
                if blink_sequence == "....-":
                    final_text += " "
                    blink_sequence = ""
                    blink_sequence_2 = " "
        
        cv2.putText(frame, f"Morse: {blink_sequence}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Text: {decoded_text}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Predicted: {selected_word}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Final Text: {final_text}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return blink_sequence

MORSE_CODE_DICT = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E', '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O', '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y', '--..': 'Z',
    '....-': ' ', '.....': 'SELECT',
}

def decode_morse(morse_sequence):
    words = morse_sequence.split("$")  # Split words using '$'
    decoded_text = " ".join(
        "".join(MORSE_CODE_DICT.get(char, "") for char in word.split("#")) for word in words
    )
    return decoded_text

# def Predict_Next_Words(interpreter, tokenizer, text):
#     sequence = tokenizer.texts_to_sequences([text])
#     if len(sequence[0]) == 0:
#         return None
#     sequence = np.array(sequence, dtype=np.float32).reshape((1, -1))
#     interpreter.set_tensor(input_details[0]['index'], sequence)
#     interpreter.invoke()
#     output_data = interpreter.get_tensor(output_details[0]['index'])
#     return tokenizer.index_word.get(np.argmax(output_data), 'the')

def Predict_Next_Words(interpreter, tokenizer, text):
    # Split the text and get the last word
    text = text.rstrip()
    words = text.split(" ")
    last_word = words[-1] if words else ""
    print(last_word)

    # Tokenize the last word
    sequence = tokenizer.texts_to_sequences([last_word])

    if len(sequence[0]) == 0:
        return None

    # Convert sequence to a NumPy array
    sequence = np.array(sequence, dtype=np.float32)

    # Ensure the input tensor has shape (1, 1), as expected by the model
    sequence = sequence.reshape(1, 1)  # Reshape to match the expected input shape of (1, 1)

    interpreter.set_tensor(input_details[0]['index'], sequence)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    #print("hello" + tokenizer.index_word.get(np.argmax(output_data), 'the'))

    return tokenizer.index_word.get(np.argmax(output_data), 'the')


morse_sequence = detect_blinks_and_gaze()
decoded_text = decode_morse(morse_sequence)
print("Decoded Text:", decoded_text)