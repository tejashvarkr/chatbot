#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import mediapipe as mp
import numpy as np

mpHands = mp.solutions.hands
hands = mpHands.Hands()

cap = cv2.VideoCapture(0)

# Define a dictionary to map hand landmarks to characters
hand_landmark_to_char = {
    (0, 1, 2, 3, 4): 'A',  # Thumb and index finger
    (0, 2, 3, 4): 'B',  # Thumb and middle finger
    (0, 1, 3, 4): 'C',  # Thumb and ring finger
    (0, 1, 2): 'D',  # Thumb and pinky finger
    (1, 2, 3, 4): 'E',  # Index and middle fingers
    (1, 2, 3): 'F',  # Index and ring fingers
    (1, 2): 'G',  # Index and pinky fingers
    (2, 3, 4): 'H',  # Middle and ring fingers
    (2, 3): 'I',  # Middle and pinky fingers
    (3, 4): 'J',  # Ring and pinky fingers
    (0): '0',  # Thumb only
    (1): '1',  # Index finger only
    (2): '2',  # Middle finger only
    (3): '3',  # Ring finger only
    (4): '4',  # Pinky finger only
    (): ' '  # No fingers
}

while True:
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            hand_landmarks = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                hand_landmarks.append((cx, cy))
                cv2.circle(frame, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

            # Extract the fingers that are extended
            extended_fingers = [i for i, landmark in enumerate(hand_landmarks) if landmark[1] < hand_landmarks[0][1]]

            # Get the character corresponding to the hand gesture
            char = hand_landmark_to_char.get(tuple(extended_fingers), '')

            # Print the character
            print(char, end='', flush=True)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




