import cv2
import mediapipe as mp
import math
import numpy as np
import time
import os
import pyautogui


class principal:
    def __init__(self, modo=False, model_complex=1, max_cant_manos=1, min_confianza_deteccion=0.8,
                 min_confianza_rastreo=0.8):  # Si no se ingresa valores, estos seran los valores predeterminados
        self.modo = modo  # para ingresar el modo de la imagen, si es estatico o en tiempo real
        self.maxmanos = max_cant_manos  # ingresa la cantidad de manos a detectar
        self.mindeteccion = min_confianza_deteccion  # ingresa la confianza de deteccion de la mano
        self.minrastreo = min_confianza_rastreo  # ingresa la confianza de rastreo de la mano en el frame
        self.complejidad_modelo = model_complex  # ingresa la complejidad del modelo

        self.clase_manos = mp.solutions.hands
        self.puntos_interes = mp.solutions.drawing_utils  # coloca los puntos rojos o los 21 puntos de interes
        self.manos = self.clase_manos.Hands(self.modo, self.complejidad_modelo, self.maxmanos, self.mindeteccion,
                                            self.minrastreo)
        self.dedos_punta = [4, 8, 12, 16, 20]
        self.resultados = None
        self.coord = None

    def detectar_manos(self, frame, dibujar=True):  # funcion para detectar las manos con el opencv

        color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convierte la entrada BGR a RGB en opencv
        self.resultados = self.manos.process(color)  # se crea una variable para almacenar los resultados

        if self.resultados.multi_hand_landmarks:
            for mano in self.resultados.multi_hand_landmarks:
                if dibujar:
                    self.puntos_interes.draw_landmarks(frame, mano,
                                                       self.clase_manos.HAND_CONNECTIONS)  # se dibujan las conexiones entre los 21 puntos
        return frame

    def rastreo_posicion(self, frame, numero_mano=0, dibujar=True, texto="", texto2=""):
        coord_x = []
        coord_y = []
        rect_box = []  # dibujar el rectangulo que sigue a la mano
        self.coord = []
        if self.resultados.multi_hand_landmarks:
            hande = self.resultados.multi_hand_landmarks[numero_mano]
            for id, lm in enumerate(hande.landmark):
                alto, ancho, c = frame.shape  # se extraen las dimensiones de los frames por segundo
                cx, cy = int(lm.x * ancho), int(lm.y * alto)  # se convierte a pixeles
                coord_x.append(cx)
                coord_y.append(cy)
                self.coord.append([id, cx, cy])

            xmin, xmax = min(coord_x), max(coord_x)
            ymin, ymax = min(coord_y), max(coord_y)
            rect_box = xmin, ymin, xmax, ymax
            if dibujar:
                # cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)
                cv2.putText(frame, '{}'.format(texto), (xmin - 20, ymin - 25), 5, 1.0, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, '{}'.format(texto2), (xmin - 20, ymin - 50), 5, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
        return self.coord, rect_box

    def dedos_levantados(self):
        dedos = []
        if self.coord[self.dedos_punta[0]][1] > self.coord[self.dedos_punta[0] - 1][1]:
            dedos.append(1)
        else:
            dedos.append(0)

        for id in range(1, 5):
            if self.coord[self.dedos_punta[id]][2] < self.coord[self.dedos_punta[id] - 2][2]:
                dedos.append(1)
            else:
                dedos.append(0)
        return dedos

    def distancia(self, punto1, punto2, frame, radio=15, colorCirculo=(10, 24, 243), dibujar=True, grosor=3):
        x1, y1 = self.coord[punto1][1:]
        x2, y2 = self.coord[punto2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if dibujar:
            cv2.circle(frame, (x1, y1), radio, colorCirculo, cv2.FILLED)
            cv2.circle(frame, (x2, y2), radio, colorCirculo, cv2.FILLED)
            cv2.line(frame, (x1, y1), (x2, y2), (28, 243, 28), grosor)
            cv2.circle(frame, (cx, cy), radio, colorCirculo, cv2.FILLED)
        longitud = math.hypot(x2 - x1, y2 - y1)
        return longitud, frame, [x1, y1, x2, y2, cx, cy]


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    texto = ""
    texto2 = ""

    detector = principal(max_cant_manos=1, min_confianza_deteccion=0.7, min_confianza_rastreo=0.7)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detector.detectar_manos(frame)
        coord, rect_box = detector.rastreo_posicion(frame,
                                                    texto=texto,
                                                    texto2=texto2)  # rect_box es el rectangulo que sigue a la mano y almacena las coordenadas minimas y maximas de cada punto
        # coord es una lista que almacena las coordenadas de cada punto de interes
        # print(coord)
        if len(coord) != 0:  # len(coord) cuenta la cantidad de puntos de interes
            x1, y1 = coord[4][1], coord[4][2]  # Se extraen las coordenadas del punto 4
            x2, y2 = coord[8][1], coord[8][2]  # Se extraen las coordenadas del punto 8
            x3, y3 = coord[0][1], coord[0][2]  # se extraen las coordenas del punto 0
            x4, y4 = coord[6][1], coord[6][2]  # se extraen las coordenas del punto 6
            x5, y5 = coord[5][1], coord[5][2]  # se extraen las coordenas del punto 5
            dist_8_6 = math.hypot(x2 - x4, y2 - y4)  # se calcula la distancia entre los puntos 8 y 6
            dist_8_0 = math.hypot(x3 - x1, y3 - y1)  # se calcula la distancia entre los puntos 8 y 0
            dist_8_4 = math.hypot(x2 - x1, y2 - y1)  # se calcula la distancia entre los puntos 8 y 4

            dedos = detector.dedos_levantados()

            #cv2.line(frame, (290, 0), (290, 480), (255, 0, 0), 4)  # linea vertical
            #cv2.line(frame, (340, 0), (340, 480), (255, 0, 0), 4)  # linea vertical
            # cv2.line(frame, (0, 240), (640, 240), (255, 0, 0), 4)#linea horizontal

            if dedos[0] == 1 and dedos[1] == 1 and dedos[2] == 1 and dedos[3] == 1 and dedos[4] == 1:
                texto = "Posicion inicial"

            elif (dedos[0] == 0 and dedos[1] == 1 and dedos[2] == 0 and dedos[3] == 0 and dedos[4] == 0):

                if dedos[1] == 1:
                    texto = "Posicion 1"
                    if x2 > 290 and x2 < 340:
                        texto2 = ""

                    if x2 < 290:
                        texto2 = "izquierda"
                        pyautogui.hotkey('alt', 'left', interval=(0.5))
                    if x2 > 340:
                        texto2 = "derecha"
                        pyautogui.hotkey('alt', 'right', interval=(0.5))

                    if dist_8_6 < 30:
                        texto2 = "abajo"
                        screenshot = pyautogui.screenshot()
                        screenshot.save("screenshot.png")

            elif dedos[0] == 0 and dedos[1] == 1 and dedos[2] == 1 and dedos[3] == 0 and dedos[4] == 0:

                if dedos[1] == 1 and dedos[2] == 1:
                    texto = "Posicion 2"

                    if x2 > 290 and x2 < 340:
                        texto2 = ""

                    if x2 < 290:
                        texto2 = "izquierda"
                        pyautogui.hotkey('ctrl', 'z', interval=(0.5))
                    if x2 > 340:
                        texto2 = "derecha"
                        pyautogui.hotkey('ctrl', 'y', interval=(0.5))
            elif dedos[0] == 1 and dedos[1] == 1 and dedos[2] == 0 and dedos[3] == 0 and dedos[4] == 0:

                if dedos[0] == 1 and dedos[1] == 1:
                    texto = "Posicion 3"
                    longitud, frame, linea = detector.distancia(4, 8, frame, radio=4, grosor=2)

                    if dist_8_4 > 100:
                        texto2 = ""
                    elif dist_8_4 > 75 and dist_8_4 < 105:
                        texto2 = "abierto"
                        pyautogui.hotkey('ctrl', '+', interval=(0.5))
                    else:
                        texto2="cerrado"
                        pyautogui.hotkey('ctrl', '-', interval=(0.5))
            if dedos[0] == 0 and dedos[1] == 0 and dedos[2] == 0 and dedos[3] == 0 and dedos[4] == 0:
                texto2 = ""

        cv2.imshow("Video", frame)
        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
