import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees


def calculate_distance(x1, y1, x2, y2):
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    return np.linalg.norm(p1 - p2)


def detect_finger_down(hand_landmarks):
    finger_down = False
    y_wrist0 = int(hand_landmarks.landmark[0].y * height)
    y_indexT0 = int(hand_landmarks.landmark[8].y * height)
    if y_indexT0 > y_wrist0:
        finger_down = True
    return finger_down


def detect_finger_up(hand_landmarks):
    finger_up = False
    y_wrist1 = int(hand_landmarks.landmark[0].y * height)
    y_indexT1 = int(hand_landmarks.landmark[8].y * height)
    y_middleT0 = int(hand_landmarks.landmark[12].y * height)
    if y_indexT1 < y_wrist1 and y_indexT1 < y_middleT0:
        finger_up = True
    return finger_up


def detect_fist(hand_landmarks):
    fist = False
    y_indexT2 = int(hand_landmarks.landmark[8].y * height)
    x_indexT2 = int(hand_landmarks.landmark[8].x * width)
    y_wrist2 = int(hand_landmarks.landmark[0].y * height)
    x_wrist2 = int(hand_landmarks.landmark[0].x * width)
    d_index = calculate_distance(x_wrist2, y_wrist2, x_indexT2, y_indexT2)
    y_middleT1 = int(hand_landmarks.landmark[12].y * height)
    x_middleT1 = int(hand_landmarks.landmark[12].x * width)
    y_middleM0 = int(hand_landmarks.landmark[9].y * height)
    x_middleM0 = int(hand_landmarks.landmark[9].x * width)
    d_middle = calculate_distance(x_wrist2, y_wrist2, x_middleT1, y_middleT1)
    x_ringT0 = int(hand_landmarks.landmark[16].x * width)
    y_ringT0 = int(hand_landmarks.landmark[16].y * height)
    d_ring = calculate_distance(x_wrist2, y_wrist2, x_ringT0, y_ringT0)
    x_pinkyT0 = int(hand_landmarks.landmark[20].x * width)
    y_pinkyT0 = int(hand_landmarks.landmark[20].y * height)
    d_pinky = calculate_distance(x_wrist2, y_wrist2, x_pinkyT0, y_pinkyT0)
    d_base = calculate_distance(x_wrist2, y_wrist2, x_middleM0, y_middleM0)
    if d_index < d_base and d_middle < d_base and d_ring < d_base and d_pinky < d_base:
        fist = True
    return fist


def detect_join_finger(hand_landmarks):
    join_finger = False
    x_indexT3 = int(hand_landmarks.landmark[8].x * width)
    y_indexT3 = int(hand_landmarks.landmark[8].y * height)

    x_thumbT0 = int(hand_landmarks.landmark[4].x * width)
    y_thumbT0 = int(hand_landmarks.landmark[4].y * height)

    d_join = int(calculate_distance(x_indexT3, y_indexT3, x_thumbT0, y_thumbT0))

    lista1 = []
    for i in enumerate([d_join]):
        lista1.append(d_join)

        if lista1 >= [5] and lista1 <= [24]:
            join_finger = True
    return join_finger


def detect_spread_finger(hand_landmarks):
    spread_finger = False

    x_indexT4 = int(hand_landmarks.landmark[8].x * width)
    y_indexT4 = int(hand_landmarks.landmark[8].y * height)

    x_thumbT1 = int(hand_landmarks.landmark[4].x * width)
    y_thumbT1 = int(hand_landmarks.landmark[4].y * height)

    d_spread = int(calculate_distance(x_indexT4, y_indexT4, x_thumbT1, y_thumbT1))

    lista2 = []
    for x in enumerate([d_spread]):
        lista2.append(d_spread)

        if [25] <= lista2 <= [55]:
            spread_finger = True
    return spread_finger


def detect_open_hand(hand_landmarks):
    finger_down = False
    finger_up = False

    x_base1 = int(hand_landmarks.landmark[4].x * width)
    y_base1 = int(hand_landmarks.landmark[4].y * height)

    x_base2 = int(hand_landmarks.landmark[20].x * width)
    y_base2 = int(hand_landmarks.landmark[20].y * height)

    p1 = np.array([x_base1, y_base1])
    p2 = np.array([x_base2, y_base2])

    Calcular = np.linalg.norm(p1 - p2)

    if Calcular >= 100:
        finger_up = True
    if finger_up == True and finger_down == False and Calcular <= 10:
        finger_down = True
    if finger_down == True and Calcular >= 100:
        finger_up = False
    return finger_up


def detect_close_hand(hand_landmarks):
    finger_down = False
    finger_up = False

    x_base1 = int(hand_landmarks.landmark[4].x * width)
    y_base1 = int(hand_landmarks.landmark[4].y * height)

    x_base2 = int(hand_landmarks.landmark[20].x * width)
    y_base2 = int(hand_landmarks.landmark[20].y * height)

    p1 = np.array([x_base1, y_base1])
    p2 = np.array([x_base2, y_base2])

    Calcular2 = np.linalg.norm(p1 - p2)

    if Calcular2 >= 10:
        finger_down = True
    if finger_down == True and finger_up == False and Calcular2 <= 100:
        finger_up = True
    if finger_up == True and Calcular2 >= 10:
        finger_down = False
    return finger_up


def move_right(hand_landmarks):
    move_right1 = False
    x_indexT5 = int(hand_landmarks.landmark[8].x * width)
    x_indexD0 = int(hand_landmarks.landmark[7].x * width)
    if x_indexT5 < 320 and x_indexD0 < 320:
        move_right1 = True
    return move_right1


def move_left(hand_landmarks):
    move_left1 = False
    x_indexT5 = int(hand_landmarks.landmark[8].x * width)
    x_indexD0 = int(hand_landmarks.landmark[7].x * width)
    if x_indexT5 > 320 and x_indexD0 > 320:
        move_left1 = True
    return move_left1


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
clase_manos = mp.solutions.hands
dibujo = mp.solutions.drawing_utils
manos = clase_manos.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8,
                          min_tracking_confidence=0.8)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    height, width, _ = frame.shape
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    copia = frame.copy()
    resultado = manos.process(color)
    posiciones = []

    if resultado.multi_hand_landmarks:
        for mano in resultado.multi_hand_landmarks:
            coordsx = int(mano.landmark[clase_manos.HandLandmark.WRIST].x * width)
            coordsy = int(mano.landmark[clase_manos.HandLandmark.WRIST].y * height)
            x = int(mano.landmark[9].x * width)
            y = int(mano.landmark[9].y * height)
            for id, lm in enumerate(mano.landmark):
                alto, ancho, c = frame.shape
                corx, cory = int(lm.x * ancho), int(lm.y * alto)
                posiciones.append([id, corx, cory])
                dibujo.draw_landmarks(frame, mano, clase_manos.HAND_CONNECTIONS)
            if len(posiciones) != 0:
                pto_i1 = posiciones[4]
                pto_i2 = posiciones[20]
                pto_i3 = posiciones[12]
                pto_i4 = posiciones[0]
                pto_i5 = posiciones[9]
                x1, y1 = (pto_i5[1] - 85), (pto_i5[2] - 85)
                ancho, alto = (x1 + 85), (y1 + 85)
                x2, y2 = x1 + ancho, y1 + alto
                dedos_reg = copia[y1:y2, x1:x2]
                #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                if detect_finger_down(mano):
                    cv2.putText(frame, '{}'.format("Abajo"), (x1 - 30, y1 - 30), 1, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
                    break

                if detect_finger_up(mano):
                    cv2.putText(frame, '{}'.format("Arriba"), (x1 - 30, y1 - 30), 1, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
                    break

                if detect_fist(mano):
                    cv2.putText(frame, '{}'.format("Mano cerrada"), (x1 - 30, y1 - 30), 1, 1.5, (0, 0, 255), 2,
                                cv2.LINE_AA)
                    break
                if detect_join_finger(mano):
                    cv2.putText(frame, '{}'.format("Disminuir"), (x1 - 30, y1 - 30), 1, 1.5, (0, 0, 255), 2,
                                cv2.LINE_AA)
                    break
                if detect_spread_finger(mano):
                    cv2.putText(frame, '{}'.format("Ampliar"), (x1 - 30, y1 - 30), 1, 1.5, (0, 0, 255), 2,
                                cv2.LINE_AA)
                    break
                if detect_open_hand(mano):
                    cv2.putText(frame, '{}'.format("Reabrir"), (x1 - 30, y1 - 30), 1, 1.5, (0, 0, 255), 2,
                                cv2.LINE_AA)
                    break
                if detect_close_hand(mano):
                    cv2.putText(frame, '{}'.format("Minimizar"), (x1 - 30, y1 - 30), 1, 1.5, (0, 0, 255), 2,
                                cv2.LINE_AA)
                    break
                if move_right(mano):
                    cv2.putText(frame, '{}'.format("Deslizar derecha"), (x1 + 175, y1 - 30), 1, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)
                    break

                elif move_left(mano):
                    cv2.putText(frame, '{}'.format("Deslizar izquierda"), (x1 + 175, y1 - 30), 1, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)

    cv2.imshow("Video", frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()