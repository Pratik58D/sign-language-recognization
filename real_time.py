import cv2
import mediapipe as mp
import joblib
import numpy as np

# Load the trained SVM model and label encoder
svm_model = joblib.load('svm_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Initialize MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize VideoCapture
cap = cv2.VideoCapture(0)

# Real-time hand gesture recognition
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to capture image.")
        break

    # Convert the BGR image to RGB and process with MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Collect hand landmarks (21 points with (x, y, z) coordinates)
            hand_data = []
            for landmark in hand_landmarks.landmark:
                hand_data.extend([landmark.x, landmark.y, landmark.z])

            # Prepare the data for the model
            hand_data = np.array(hand_data).reshape(1, -1)

            # Make a prediction using the trained SVM model
            predicted_label = svm_model.predict(hand_data)
            predicted_label = label_encoder.inverse_transform(predicted_label)

            # Print the predicted label to the console
            print(f"Predicted: {predicted_label[0]}")

            # Display the predicted label on the image
            cv2.putText(image, f"Predicted: {predicted_label[0]}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the result
    cv2.imshow('Hand Gesture Recognition', image)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
