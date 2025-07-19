import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Screen size for mouse mapping
screen_width, screen_height = pyautogui.size()

# Start webcam
cap = cv2.VideoCapture(0)

# Initialize finger coordinates
x1 = y1 = x2 = y2 = 0

while True:
    success, img = cap.read()
    if not success:
        break

    # Flip the image horizontally for natural feel
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    img_height, img_width, _ = img.shape

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Iterate through landmarks
            for id, lm in enumerate(hand_landmarks.landmark):
                x = int(lm.x * img_width)
                y = int(lm.y * img_height)

                # Index finger tip (id=8)
                if id == 8:
                    cv2.circle(img, (x, y), 10, (255, 0, 0), cv2.FILLED)
                    mouse_x = screen_width / img_width * x
                    mouse_y = screen_height / img_height * y
                    pyautogui.moveTo(mouse_x, mouse_y)
                    x1, y1 = x, y

                # Thumb tip (id=4)
                if id == 4:
                    cv2.circle(img, (x, y), 10, (0, 255, 0), cv2.FILLED)
                    x2, y2 = x, y

            # Check for click gesture (thumb close to index finger)
            distance = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
            if distance < 40:
                pyautogui.click()
                cv2.putText(img, "Click", (x1, y1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Hand Mouse Control", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
