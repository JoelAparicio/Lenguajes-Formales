import cv2
import mediapipe as mp
import math
import numpy as np
from math import acos, degrees
import time


class principal():
    def __init__(self, modo=False, model_complex=1, max_cant_manos=1, min_confianza_deteccion=0.8,
                 min_confianza_rastreo=0.8):  # Si no se ingresa valores, estos seran los valores predeterminados
        self.modo = modo  # para ingresar el modo de la imagen, si es estatico o en tiempo real
        self.maxmanos = max_cant_manos  # ingresa la cantidad de manos a detectar
        self.mindeteccion = min_confianza_deteccion  # ingresa la confianza de deteccion de la mano
        self.minrastreo = min_confianza_rastreo  # ingresa la confianza de rastreo de la mano en el frame
        self.complejidad_modelo = model_complex  # ingresa la complejidad del modelo

        self.clase_manos = mp.solutions.hands
        self.puntos_interes = mp.solutions.drawing_utils  # coloca los puntos rojos o los 21 puntos de interes
        self.manos = self.clase_manos.Hands(self.modo, self.complejidad_modelo, self.maxmanos, self.mindeteccion, self.minrastreo)
        self.dedos_punta = [4, 8, 12, 16, 20]

    def detectar_manos(self, frame, dibujar=True):  # funcion para detectar las manos con el opencv

        color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convierte la entrada BGR a RGB en opencv
        self.resultados = self.manos.process(color)  # se crea una variable para

        if self.resultados.multi_hand_landmarks:
            for mano in self.resultados.multi_hand_landmarks:
                if dibujar:
                    self.puntos_interes.draw_landmarks(frame, mano, self.clase_manos.HAND_CONNECTIONS)  # se dibujan las conexiones entre los 21 puntos
        return frame

    def rastreo_posicion(self, frame, numero_mano=0, dibujar=True, texto=""):
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
                #cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)
                cv2.putText(frame, '{}'.format(texto), (xmin - 20, ymin - 25), 5, 1.5, (0, 0, 255), 1, cv2.LINE_AA)
        return self.coord, rect_box

    def dedos_levantados(self):
        dedos = []
        if self.coord[self.dedos_punta[0]][1] > self.coord[self.dedos_punta[0] - 1][0]:
            dedos.append(1)
        else:
            dedos.append(0)

        for id in range(1, 5):
            if self.coord[self.dedos_punta[id]][2] < self.coord[self.dedos_punta[id] - 2][2]:
                dedos.append(1)
            else:
                dedos.append(0)
        return dedos


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    detector = principal(max_cant_manos=1, min_confianza_deteccion=0.9, min_confianza_rastreo=0.9)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detector.detectar_manos(frame)
        coord, rect_box = detector.rastreo_posicion(frame, texto="Mano")#rect_box es el rectangulo que sigue a la mano y almacena las coordenadas minimas y maximas de cada punto
        #coord es una lista que almacena las coordenadas de cada punto de interes
        print(rect_box)
        #dedos = detector.dedos_levantados()
        #print(dedos)
        #if dedos[0] == 1 and dedos[1] == 1:
            #print("Pulgar y indice levantados")
        cv2.imshow("Video", frame)
        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
