# Progetto Computer Vision
# Titolo: Mosaicing
# Autore: Emanuele Venturelli - matricola VR509826
# Ultima modifica: 23/02/2025

###############################################################################################################################

# Import pacchetti e dipendenze

import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
import sys
from posixpath import dirname
from RANSAC_HM import ransac_homography

# Impostazione della directory di input ed output
img_folder_output = (dirname(sys.argv[0]))
img_folder_input = os.path.join(img_folder_output, 'input')

# Lista delle immagini nella cartella di input
lst = os.listdir(img_folder_input) 
num_immagini_input = len(lst) # Numero totale di immagini di input
img_0 = np.floor(num_immagini_input/2) # Selezione dell'immagine centrale

# Inizializzazione delle variabili
descr_post_SIFT = []
immagini_pre_SIFT = []
immagini_post_SIFT = []
kp_post_SIFT = []
immagini_input = []
imgg = []
contatore = 0
contatore_ind = 1
img_temp = np.zeros(3)
sift = cv.SIFT_create() # Creazione dell'oggetto SIFT
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False) # Creazione dell'oggetto BFMatcher

H = []
scale_percent = 10  # Riduci la dimensione al 15%

# Caricamento delle immagini di input da cartella
for filename in os.listdir(img_folder_input):
        if filename.endswith('.jpg') or filename.endswith('.png'): # carico solo i file con estensione .jpg o .png
                img_path = os.path.join(img_folder_input, filename)
                imgn = cv.imread(img_path)
                # Riduzione risoluzione dell'immagine per calcolo più veloce
                width = int(imgn.shape[1] * scale_percent / 100)
                height = int(imgn.shape[0] * scale_percent / 100)
                dim = (width, height)
                immagini_input.append(cv.resize(imgn, dim, interpolation=cv.INTER_AREA))


################################  ---  ALGORITMO DA SX A DX  ---  #################################################

contatore = (img_0.copy()).astype('int')  #inizializzazione contatori
contatorepiu1 = (img_0.copy()).astype('int') + 1

while contatore < (len(immagini_input)-1):

        if contatore < contatorepiu1:    # Prima iterazione 

                # Conversione in scala di grigi delle immagini
                gray_A= cv.cvtColor(immagini_input[contatore].copy(), cv.COLOR_BGR2GRAY)
                gray_B= cv.cvtColor(immagini_input[contatore+1].copy(), cv.COLOR_BGR2GRAY)

                # Estrazione dei keypoint e descrittori usando SIFT
                kp_A, descriptor_A = sift.detectAndCompute(gray_A,None)
                kp_B, descriptor_B = sift.detectAndCompute(gray_B,None)

                # Match dei descrittori
                matches = sorted((bf.match(descriptor_A, descriptor_B)), key = lambda x:x.distance)

                # Conversione dei keypoint in float32 per calcoli geometrici, estrazione delle coordinate
                keypoints_train_image = np.float32([keypoint.pt for keypoint in kp_A])
                keypoints_query_image = np.float32([keypoint.pt for keypoint in kp_B])

                if len(matches) >= 4:
                        pointsBB = np.float32([keypoints_train_image[m.queryIdx] for m in matches])
                        pointsAA = np.float32([keypoints_query_image[m.trainIdx] for m in matches])

                        # Calcolo dell'omografia usando RANSAC
                        H = ransac_homography(pointsAA, pointsBB, num_iterations=500, threshold=5.0)

                else:   # se non trova abbastanza match, prova a saltare un immagine e ad unire la successiva
                        print(f'Minimum number of matches not reached. It is not possible to calculate homography matrix.') 
                        gray_B= cv.cvtColor(immagini_input[contatore+2].copy(), cv.COLOR_BGR2GRAY) # nuova immmagine
                        kp_B, descriptor_B = sift.detectAndCompute(gray_B,None) # calcolo dei nuovi descrittori
                        matches = sorted((bf.match(descriptor_A, descriptor_B)), key = lambda x:x.distance)
                        keypoints_query_image = np.float32([keypoint.pt for keypoint in kp_B])
                        pointsBB = np.float32([keypoints_train_image[m.queryIdx] for m in matches])
                        pointsAA = np.float32([keypoints_query_image[m.trainIdx] for m in matches])
                        H = ransac_homography(pointsAA, pointsBB, num_iterations=500, threshold=5.0) # calcolo matrice di omografia
                        contatore += 1 # aggiornamento contatori
                        contatore_ind += 1

                # Dimensioni del panorama e calcolo della maschera di fusione
                height_img_A = immagini_input[contatore].shape[0] # altezza immagine
                width_img_A = immagini_input[contatore].shape[1] # larghezza prima immagine
                width_img_B = immagini_input[contatore+1].shape[1] # larghezza seconda immagine
                height_panorama = height_img_A # altezza panorama
                width_panorama = width_img_A + width_img_B # larghezza panorama
                lowest_width = min(width_img_A, width_img_B) # calcolo larghezza minima per definizione larghezza del filtro di transizione

                # Inizio filtro per transizione tra un'immagine e l'altra

                smoothing_window_percent = 0.10 # Indice compreso tra 0.00 e 1.00. che corriponde alla percentuale di area di immagine usata per il filtro di transizione
                smoothing_window_size = max(100, min(smoothing_window_percent * lowest_width, 1000)) # Sarà sicuramente un numero compreso tra 100 e 1000

                offset = int(smoothing_window_size / 2)

                barrier = immagini_input[contatore].shape[1] - int(smoothing_window_size / 2)

                mask_A = np.zeros((height_panorama, width_panorama)) # definizione matrice di base maschera A
                mask_A[:, barrier - offset : barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset).T, (height_panorama, 1)) # creazione di array descrescente da 1 a 0 nella parta dx dell'immagine per effetto sfumatura
                mask_A[:, : barrier - offset] = 1 # il resto della maschera sarà unitaria
                mask_A1 = cv.merge([mask_A, mask_A, mask_A]) # sovrapposizione dei tre maschere per dare profonodità pari a 3


                mask_B = np.zeros((height_panorama, width_panorama)) # definizione matrice di base maschera B
                mask_B[:, barrier - offset : barrier + offset] = np.tile(np.linspace(0, 1, 2 * offset).T, (height_panorama, 1)) # creazione di array descrescente da 1 a 0 nella parta sx dell'immagine per effetto sfumatura
                mask_B[:, barrier + offset :] = 1 # il resto della maschera sarà unitaria
                mask_B1 = cv.merge([mask_B, mask_B, mask_B]) # sovrapposizione dei tre maschere per dare profonodità pari a 3

                # Warping e fusione delle immagini
                img_rtd = cv.warpPerspective(immagini_input[contatore+1], H, (width_panorama, height_panorama)) # rotazione della seconda immagine rispetto alla prima
                panorama1 = np.zeros(((height_panorama), (width_panorama), 3)) # creazione di matrice di zeri grande come il panorama finale
                panorama1[0 : immagini_input[contatore].shape[0], 0 : immagini_input[contatore].shape[1], :] = immagini_input[contatore]
                panorama_final = panorama1*mask_A1 + img_rtd*mask_B1 # applicazione delle maschere alle due immagini e creazione del panorama
                
                # Ritaglio delle bande nere
                rows, cols = np.where(panorama_final[:, :, 0] != 0) # Trova le bande nere sulla dx

                min_row, max_row = min(rows), max(rows) + 1 # definizione massima e minima riga dell'immagine
                min_col, max_col = min(cols), max(cols) + 1 # definizione massima e minima colonna dell'immagine

                xA = max(cols[max(np.where(rows == 50+int(np.floor(max_row/2))))]) # definizione di punto A al limite della banda nera
                xB = max(cols[max(np.where(rows == -50+int(np.floor(max_row/2))))]) # definizione di punto B al limite della banda nera

                x0 = (xB-xA)*(-(50+int(np.floor(max_row/2))))/(-50+int(np.floor(max_row/2))-(50+int(np.floor(max_row/2))))+xA # eq. della retta passante per due punti per trovare punto più basso della banda nera
                xMAX = (xB-xA)*(max_row-(50+int(np.floor(max_row/2))))/(-50+int(np.floor(max_row/2))-(50+int(np.floor(max_row/2))))+xA # eq. della retta passante per due punti per trovare punto più alto della banda nera

                img_temp = ((panorama_final[min_row:max_row, min_col:int(np.floor((min(x0, xMAX)-1))), :]).astype('uint8')).copy() # eliminazione della banda nera sulla destra

                contatore += 1 # aggiornamento contatore

                print('Iteration number:', contatore_ind) # stampo il numero di iterazione

                H_prev = H.copy() # copia di H in nuova variabile
                H = None # pulizia delle variabili


        else:
                # Conversione in scala di grigi delle immagini

                gray_A= cv.cvtColor(immagini_input[contatore].copy(), cv.COLOR_BGR2GRAY)
                gray_B= cv.cvtColor(immagini_input[contatore+1].copy(), cv.COLOR_BGR2GRAY)

                # Estrazione dei keypoint e descrittori usando SIFT
                kp_A, descriptor_A = sift.detectAndCompute(gray_A,None)
                kp_B, descriptor_B = sift.detectAndCompute(gray_B,None)

                # Match dei descrittori
                matches = sorted((bf.match(descriptor_A, descriptor_B)), key = lambda x:x.distance)

                # Conversione dei keypoint in float32 per calcoli geometrici
                keypoints_train_image = np.float32([keypoint.pt for keypoint in kp_A])
                keypoints_query_image = np.float32([keypoint.pt for keypoint in kp_B])

                if len(matches) >= 4: # Verifica che ci siano sufficienti match
                        pointsBB = np.float32([keypoints_train_image[m.queryIdx] for m in matches])
                        pointsAA = np.float32([keypoints_query_image[m.trainIdx] for m in matches])
                        
                        # Calcolo dell'omografia usando RANSAC
                        H = ransac_homography(pointsAA, pointsBB, num_iterations=500, threshold=5.0)

                else:   # se non trova abbastanza match, prova a saltare un immagine e ad unire la successiva
                        print('Minimum number of matches not reached. It is not possible to calculate homography matrix!')
                        gray_B = None # pulizia variabili
                        kp_b = None
                        descriptor_B = None
                        matches = None
                        gray_B= cv.cvtColor(immagini_input[contatore+2].copy(), cv.COLOR_BGR2GRAY) # nuova immmagine
                        kp_B, descriptor_B = sift.detectAndCompute(gray_B,None) # calcolo dei nuovi descrittori
                        matches = sorted((bf.match(descriptor_A, descriptor_B)), key = lambda x:x.distance)
                        keypoints_query_image = np.float32([keypoint.pt for keypoint in kp_B])
                        pointsBB = np.float32([keypoints_train_image[m.queryIdx] for m in matches])
                        pointsAA = np.float32([keypoints_query_image[m.trainIdx] for m in matches])
                        H = ransac_homography(pointsAA, pointsBB, num_iterations=500, threshold=5.0) # calcolo matrice di omografia
                        contatore += 1 # aggiornamento contatori
                        contatore_ind += 1

                # Dimensioni del panorama e calcolo della maschera di fusione
                height_img_A = img_temp.shape[0] # altezza immagine
                width_img_A = img_temp.shape[1] # larghezza prima immagine
                width_img_B = immagini_input[contatore+1].shape[1] # larghezza seconda immagine
                height_panorama = height_img_A # altezza panorama
                width_panorama = width_img_A + width_img_B # larghezza panorama
                lowest_width = min(width_img_A, width_img_B) # calcolo larghezza minima per definizione larghezza del filtro di transizione

                # Inizio filtro per transizione tra un'immagine e l'altra

                smoothing_window_percent = 0.10 # Indice compreso tra 0.00 e 1.00. che corriponde alla percentuale di area di immagine usata per il filtro
                smoothing_window_size = max(100, min(smoothing_window_percent * lowest_width, 1000)) #sarà sicuramente un numero compreso tra 100 e 1000

                offset = int(smoothing_window_size / 2)

                barrier = img_temp.shape[1] - int(smoothing_window_size / 2)

                mask_A = np.zeros((height_panorama, width_panorama)) # definizione matrice di base maschera A
                mask_A[:, barrier - offset : barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset).T, (height_panorama, 1)) # creazione di array descrescente da 1 a 0 nella parta dx dell'immagine per effetto sfumatura
                mask_A[:, : barrier - offset] = 1 # il resto della maschera sarà unitaria
                mask_A1 = cv.merge([mask_A, mask_A, mask_A])  # sovrapposizione dei tre maschere per dare profonodità pari a 3


                mask_B = np.zeros((height_panorama, width_panorama)) # definizione matrice di base maschera B
                mask_B[:, barrier - offset : barrier + offset] = np.tile(np.linspace(0, 1, 2 * offset).T, (height_panorama, 1)) # creazione di array descrescente da 1 a 0 nella parta sx dell'immagine per effetto sfumatura
                mask_B[:, barrier + offset :] = 1 # il resto della maschera sarà unitaria
                mask_B1 = cv.merge([mask_B, mask_B, mask_B]) # sovrapposizione dei tre maschere per dare profonodità pari a 3

                Hn = H_prev @ H # calcolo matrice di omografia in modo ricorsivo, predendendo la precedente e moltiplicandola per quella nuova appena calcolata

                # Warping e fusione delle immagini
                img_rtd = cv.warpPerspective(immagini_input[contatore+1], Hn, (width_panorama, height_panorama)) #ruota la seconda immagine in prospettiva con la prima
                panorama1 = np.zeros(((height_panorama), (width_panorama), 3))  # creazione di matrice di zeri grande come il panorama finale
                panorama1[0 : img_temp.shape[0], 0 : img_temp.shape[1], :] = img_temp
                panorama_final = panorama1*mask_A1 + img_rtd*mask_B1  # applicazione delle maschere alle due immagini e creazione del panorama

                # Ritaglio delle bande nere
                rows, cols = np.where(panorama_final[:, :, 0] != 0) # Trova le bande nere sulla dx
                min_row, max_row = min(rows), max(rows) + 1 # definizione massima e minima riga dell'immagine
                min_col, max_col = min(cols), max(cols) + 1 # definizione massima e minima colonna dell'immagine

                xA = max(cols[max(np.where(rows == 50+int(np.floor(max_row/2))))]) # definizione di punto A al limite della banda nera
                xB = max(cols[max(np.where(rows == -50+int(np.floor(max_row/2))))]) # definizione di punto B al limite della banda nera

                x0 = (xB-xA)*(-(50+int(np.floor(max_row/2))))/(-50+int(np.floor(max_row/2))-(50+int(np.floor(max_row/2))))+xA # eq. della retta passante per due punti per trovare punto più basso della banda nera
                xMAX = (xB-xA)*(max_row-(50+int(np.floor(max_row/2))))/(-50+int(np.floor(max_row/2))-(50+int(np.floor(max_row/2))))+xA # eq. della retta passante per due punti per trovare punto più alto della banda nera

                img_temp = ((panorama_final[min_row:max_row, min_col:int(np.floor((min(x0, xMAX)))-1), :]).astype('uint8')).copy() # eliminazione della banda nera sulla destra

                contatore += 1 # aggiornamento contatore calcolo omografia
                contatore_ind += 1 # aggiornamento contatore stampa

                print('Iteration number:', contatore_ind)

                H_prev = Hn.copy() # copia di H in nuova variabile
                Hn = None # pulizia delle variabili
                H = None # pulizia delle variabili

img_dx = img_temp.copy() # creazione panormama destro
img_path = os.path.join(img_folder_output, 'output/panorama_dx.jpg') # definizione directory di output
cv.imwrite(img_path, img_dx) # salvataggio immagine nella directory di output

################################  ---  ALGORITMO DA DX A SX  ---  #################################################

print('Da DX a SX')

cont_s = int(np.floor(num_immagini_input/2)) # conteggio immagini di input
num=11 # definzione contatore num

while cont_s > 0:
        img_path = os.path.join(img_folder_output, 'buffer/img_'+str(num)+'.jpg') # salvataggio immagini riflesse di 180 gradi rispetto all'asse verticale
        cv.imwrite(img_path, cv.flip(immagini_input[cont_s].copy(), 1))
        num += 1 # aggiornamento contatori
        cont_s -= 1


# Impostazione della directory di input ed output
img_folder_output = (dirname(sys.argv[0]))
img_folder_input = os.path.join(img_folder_output, 'buffer')

# Lista delle immagini nella cartella di input
lst = os.listdir(img_folder_input) 
num_immagini_input = len(lst) # Numero totale delle nuove immagini di input
img_0 = 0 # Selezione della prima immagine

# Definizione contatori
contatore = 0
contatorepiu1 = 1
contatore_ind += 1

# pulizia delle variabili
H_prev = None
gray_A = None
gray_B = None
kp_A = None
kp_B = None
descriptor_A = None
descriptor_B = None
matches = None
keypoints_train_image = None
keypoints_query_image = None
immagini_input = []

# Caricamento delle immagini di input
for filename in os.listdir(img_folder_input):
        if filename.endswith('.jpg') or filename.endswith('.png'): # carico solo i file con estensione .jpg o .png
                img_path = os.path.join(img_folder_input, filename)
                imgn = cv.imread(img_path)
                immagini_input.append(imgn)


while contatore < (len(immagini_input)-1):

        if contatore < contatorepiu1:    # Prima iterazione 

                # Conversione in scala di grigi delle immagini
                gray_A= cv.cvtColor(immagini_input[contatore].copy(), cv.COLOR_BGR2GRAY)
                gray_B= cv.cvtColor(immagini_input[contatore+1].copy(), cv.COLOR_BGR2GRAY)
                
                # Estrazione dei keypoint e descrittori usando SIFT
                kp_A, descriptor_A = sift.detectAndCompute(gray_A,None)
                kp_B, descriptor_B = sift.detectAndCompute(gray_B,None)

                # Match dei descrittori
                matches = sorted((bf.match(descriptor_A, descriptor_B)), key = lambda x:x.distance)

                # Conversione dei keypoint in float32 per calcoli geometrici
                keypoints_train_image = np.float32([keypoint.pt for keypoint in kp_A])
                keypoints_query_image = np.float32([keypoint.pt for keypoint in kp_B])

                if len(matches) >= 4:
                        pointsBB = np.float32([keypoints_train_image[m.queryIdx] for m in matches])
                        pointsAA = np.float32([keypoints_query_image[m.trainIdx] for m in matches])

                        # Calcolo dell'omografia usando RANSAC
                        H = ransac_homography(pointsAA, pointsBB, num_iterations=500, threshold=5.0)

                else:   # se non trova abbastanza match, prova a saltare un immagine e ad unire la successiva
                        print(f'Minimum number of matches not reached. It is not possible to calculate homography matrix.')
                        gray_B= cv.cvtColor(immagini_input[contatore+2].copy(), cv.COLOR_BGR2GRAY) # nuova immagine
                        kp_B, descriptor_B = sift.detectAndCompute(gray_B,None) # calcolo dei nuovi descittori
                        matches = sorted((bf.match(descriptor_A, descriptor_B)), key = lambda x:x.distance)
                        keypoints_query_image = np.float32([keypoint.pt for keypoint in kp_B])
                        pointsBB = np.float32([keypoints_train_image[m.queryIdx] for m in matches])
                        pointsAA = np.float32([keypoints_query_image[m.trainIdx] for m in matches])
                        H = ransac_homography(pointsAA, pointsBB, num_iterations=500, threshold=5.0) # calcolo matrice di omografia
                        contatore += 1 # aggiornamento contatori
                        contatore_ind += 1

                # Dimensioni del panorama e calcolo della maschera di fusione
                height_img_A = immagini_input[contatore].shape[0] # altezza immagine
                width_img_A = immagini_input[contatore].shape[1] # larghezza prima immagine
                width_img_B = immagini_input[contatore+1].shape[1] # larghezza seconda immagine
                height_panorama = height_img_A # altezza panorama
                width_panorama = width_img_A + width_img_B # larghezza panorama
                lowest_width = min(width_img_A, width_img_B) # calcolo larghezza minima per definizione larghezza del filtro di transizione

                # Inizio filtro per transizione tra un'immagine e l'altra

                smoothing_window_percent = 0.10 # Indice compreso tra 0.00 e 1.00. che corriponde alla percentuale di area di immagine usata per il filtro di transizione
                smoothing_window_size = max(100, min(smoothing_window_percent * lowest_width, 1000)) # Sarà sicuramente un numero compreso tra 100 e 1000

                offset = int(smoothing_window_size / 2)

                barrier = immagini_input[contatore].shape[1] - int(smoothing_window_size / 2)

                mask_A = np.zeros((height_panorama, width_panorama)) # definizione matrice di base maschera A
                mask_A[:, barrier - offset : barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset).T, (height_panorama, 1)) # creazione di array descrescente da 1 a 0 nella parta dx dell'immagine per effetto sfumatura
                mask_A[:, : barrier - offset] = 1 # il resto della maschera sarà unitaria
                mask_A1 = cv.merge([mask_A, mask_A, mask_A]) # sovrapposizione dei tre maschere per dare profonodità pari a 3


                mask_B = np.zeros((height_panorama, width_panorama)) # definizione matrice di base maschera B
                mask_B[:, barrier - offset : barrier + offset] = np.tile(np.linspace(0, 1, 2 * offset).T, (height_panorama, 1)) # creazione di array descrescente da 1 a 0 nella parta sx dell'immagine per effetto sfumatura
                mask_B[:, barrier + offset :] = 1 # il resto della maschera sarà unitaria
                mask_B1 = cv.merge([mask_B, mask_B, mask_B]) # sovrapposizione dei tre maschere per dare profonodità pari a 3

                # Warping e fusione delle immagini
                img_rtd = cv.warpPerspective(immagini_input[contatore+1], H, (width_panorama, height_panorama))
                panorama1 = np.zeros(((height_panorama), (width_panorama), 3))
                panorama1[0 : immagini_input[contatore].shape[0], 0 : immagini_input[contatore].shape[1], :] = immagini_input[contatore]
                panorama_final = panorama1*mask_A1 + img_rtd*mask_B1

                # Ritaglio delle bande nere
                rows, cols = np.where(panorama_final[:, :, 0] != 0) #Trova le bande nere sulla dx

                min_row, max_row = min(rows), max(rows) + 1 # definizione massima e minima riga dell'immagine
                min_col, max_col = min(cols), max(cols) + 1 # definizione massima e minima colonna dell'immagine

                xA = max(cols[max(np.where(rows == 50+int(np.floor(max_row/2))))]) # definizione di punto A al limite della banda nera
                xB = max(cols[max(np.where(rows == -50+int(np.floor(max_row/2))))]) # definizione di punto B al limite della banda nera

                x0 = (xB-xA)*(-(50+int(np.floor(max_row/2))))/(-50+int(np.floor(max_row/2))-(50+int(np.floor(max_row/2))))+xA # eq. della retta passante per due punti per trovare punto più basso della banda nera
                xMAX = (xB-xA)*(max_row-(50+int(np.floor(max_row/2))))/(-50+int(np.floor(max_row/2))-(50+int(np.floor(max_row/2))))+xA # eq. della retta passante per due punti per trovare punto più alto della banda nera

                img_temp = ((panorama_final[min_row:max_row, min_col:int(np.floor((min(x0, xMAX)-1))), :]).astype('uint8')).copy() # eliminazione della banda nera sulla destra

                contatore += 1 # aggiornamento contatore

                print('Iteration number:', contatore_ind) # stampo il numero di iterazione

                H_prev = H.copy() # copia di H in nuova variabile
                H = None # pulizia delle variabili


        else:
                # Conversione in scala di grigi delle immagini

                gray_A= cv.cvtColor(immagini_input[contatore].copy(), cv.COLOR_BGR2GRAY)
                gray_B= cv.cvtColor(immagini_input[contatore+1].copy(), cv.COLOR_BGR2GRAY)

                # Estrazione dei keypoint e descrittori usando SIFT
                kp_A, descriptor_A = sift.detectAndCompute(gray_A,None)
                kp_B, descriptor_B = sift.detectAndCompute(gray_B,None)

                # Match dei descrittori
                matches = sorted((bf.match(descriptor_A, descriptor_B)), key = lambda x:x.distance)

                # Conversione dei keypoint in float32 per calcoli geometrici
                keypoints_train_image = np.float32([keypoint.pt for keypoint in kp_A])
                keypoints_query_image = np.float32([keypoint.pt for keypoint in kp_B])

                #print(len(matches))

                if len(matches) >= 4: # Verifica che ci siano sufficienti match
                        pointsBB = np.float32([keypoints_train_image[m.queryIdx] for m in matches])
                        pointsAA = np.float32([keypoints_query_image[m.trainIdx] for m in matches])
                        
                        # Calcolo dell'omografia usando RANSAC
                        H = ransac_homography(pointsAA, pointsBB, num_iterations=500, threshold=5.0)

                else:   # se non trova abbastanza match, prova a saltare un immagine e ad unire la successiva
                        print('Minimum number of matches not reached. It is not possible to calculate homography matrix!')
                        gray_B = None # pulizia variabili
                        kp_b = None
                        descriptor_B = None
                        matches = None
                        gray_B= cv.cvtColor(immagini_input[contatore+2].copy(), cv.COLOR_BGR2GRAY) # nuova immmagine
                        kp_B, descriptor_B = sift.detectAndCompute(gray_B,None) # calcolo dei nuovi descrittori
                        matches = sorted((bf.match(descriptor_A, descriptor_B)), key = lambda x:x.distance)
                        keypoints_query_image = np.float32([keypoint.pt for keypoint in kp_B])
                        pointsBB = np.float32([keypoints_train_image[m.queryIdx] for m in matches])
                        pointsAA = np.float32([keypoints_query_image[m.trainIdx] for m in matches])
                        H = ransac_homography(pointsAA, pointsBB, num_iterations=500, threshold=5.0) # calcolo matrice di omografia
                        contatore += 1 # aggiornamento contatori
                        contatore_ind += 1

                # Dimensioni del panorama e calcolo della maschera di fusione
                height_img_A = img_temp.shape[0] # altezza immagine
                width_img_A = img_temp.shape[1] # larghezza prima immagine
                width_img_B = immagini_input[contatore+1].shape[1] # larghezza seconda immagine
                height_panorama = height_img_A # altezza panorama
                width_panorama = width_img_A + width_img_B # larghezza panorama
                lowest_width = min(width_img_A, width_img_B) # calcolo larghezza minima per definizione larghezza del filtro di transizione

                # Inizio filtro per transizione tra un'immagine e l'altra

                smoothing_window_percent = 0.10 # Indice compreso tra 0.00 e 1.00. che corriponde alla percentuale di area di immagine usata per il filtro
                smoothing_window_size = max(100, min(smoothing_window_percent * lowest_width, 1000)) #sarà sicuramente un numero compreso tra 100 e 1000

                offset = int(smoothing_window_size / 2)

                barrier = img_temp.shape[1] - int(smoothing_window_size / 2)

                mask_A = np.zeros((height_panorama, width_panorama)) # definizione matrice di base maschera A
                mask_A[:, barrier - offset : barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset).T, (height_panorama, 1)) # creazione di array descrescente da 1 a 0 nella parta dx dell'immagine per effetto sfumatura
                mask_A[:, : barrier - offset] = 1 # il resto della maschera sarà unitaria
                mask_A1 = cv.merge([mask_A, mask_A, mask_A])


                mask_B = np.zeros((height_panorama, width_panorama)) # definizione matrice di base maschera B
                mask_B[:, barrier - offset : barrier + offset] = np.tile(np.linspace(0, 1, 2 * offset).T, (height_panorama, 1)) # creazione di array descrescente da 1 a 0 nella parta sx dell'immagine per effetto sfumatura
                mask_B[:, barrier + offset :] = 1 # il resto della maschera sarà unitaria
                mask_B1 = cv.merge([mask_B, mask_B, mask_B]) # sovrapposizione dei tre maschere per dare profonodità pari a 3

                Hn = H_prev @ H # calcolo matrice di omografia in modo ricorsivo, predendendo la precedente e moltiplicandola per quella nuova appena calcolata

                # Warping e fusione delle immagini
                img_rtd = cv.warpPerspective(immagini_input[contatore+1], Hn, (width_panorama, height_panorama)) #ruota la seconda immagine in prospettiva con la prima
                panorama1 = np.zeros(((height_panorama), (width_panorama), 3)) # creazione di matrice di zeri grande come il panorama finale
                panorama1[0 : img_temp.shape[0], 0 : img_temp.shape[1], :] = img_temp
                panorama_final = panorama1*mask_A1 + img_rtd*mask_B1 # applicazione delle maschere alle due immagini e creazione del panorama

                # Ritaglio delle bande nere
                rows, cols = np.where(panorama_final[:, :, 0] != 0) # Trova le bande nere sulla dx
                min_row, max_row = min(rows), max(rows) + 1 # definizione massima e minima riga dell'immagine
                min_col, max_col = min(cols), max(cols) + 1 # definizione massima e minima colonna dell'immagine

                xA = max(cols[max(np.where(rows == 50+int(np.floor(max_row/2))))]) # definizione di punto A al limite della banda nera
                xB = max(cols[max(np.where(rows == -50+int(np.floor(max_row/2))))]) # definizione di punto B al limite della banda nera

                x0 = (xB-xA)*(-(50+int(np.floor(max_row/2))))/(-50+int(np.floor(max_row/2))-(50+int(np.floor(max_row/2))))+xA # eq. della retta passante per due punti per trovare punto più basso della banda nera
                xMAX = (xB-xA)*(max_row-(50+int(np.floor(max_row/2))))/(-50+int(np.floor(max_row/2))-(50+int(np.floor(max_row/2))))+xA # eq. della retta passante per due punti per trovare punto più alto della banda nera

                img_temp = ((panorama_final[min_row:max_row, min_col:int(np.floor((min(x0, xMAX)))-1), :]).astype('uint8')).copy() # eliminazione della banda nera sulla destra

                contatore += 1 # aggiornamento contatore calcolo omografia
                contatore_ind += 1 # aggiornamento contatore stampa

                print('Iteration number:', contatore_ind)

                H_prev = Hn.copy() # copia di H in nuova variabile
                Hn = None # pulizia delle variabili
                H = None # pulizia delle variabili

img_sx = cv.flip(img_temp.copy(), 1) # creazione panormama sinistro
img_path = os.path.join(img_folder_output, 'output/panorama_sx.jpg') # definizione directory di output
cv.imwrite(img_path, img_sx) # salvataggio immagine nella directory di output

img_tot = np.zeros((height_img_A, (img_sx.shape[1] + img_dx.shape[1] - immagini_input[0].shape[1]), 3)) #creazione matrice di zeri grande come la somma dell due immagini
img_tot[:, 0:(img_sx.shape[1] - immagini_input[0].shape[1]//2), :] = img_sx.copy()[:, 0:(img_sx.shape[1] - immagini_input[0].shape[1]//2), :] # incollo immagine di sinistra
img_tot[:, ((img_sx.shape[1] - immagini_input[0].shape[1]//2)):(img_tot.shape[1]), :] = img_dx.copy()[:, (immagini_input[0].shape[1]//2):, :] # incollo immagine di destra


# procedura per tagliare le bande nere sopra e sotto

zz = np.zeros((int(img_tot.shape[0]),int(img_tot.shape[1])),dtype=np.uint8) #creazione matrice di zeri grande come il panorama


# creazione matrice booleana, imposta 0 dove trova pixel neri, altrimenti 1
for i in range(int(height_img_A)):
    for j in range(int(img_tot.shape[1])):
        if img_tot[i,j,0] == 0 and img_tot[i,j,1] == 0 and img_tot[i,j,2] == 0:
            zz[i,j] = 0
        else:
            zz[i,j] = 1

#divisione immagine parte alta/bassa
zz_up = zz[0:zz.shape[0]//2, 0:zz.shape[1]]
zz_down = zz[zz.shape[0]//2:zz.shape[0], 0:zz.shape[1]]

# definizione delle varibili e dei contatori
i = None
j = None
imin = zz_down.shape[0]
imax = 0

# indentificazxione pixel nero più alto
for i in range(zz_up.shape[0]):
       for j in range(zz_up.shape[1]):
              if zz_up[i,j] == 0 and i > imax:
                     imax = i

i = None
j = None

# indentificazxione pixel nero più basso
for i in range(zz_down.shape[0]):
       for j in range(zz_down.shape[1]):
              if zz_down[i,j] == 0 and i < imin:
                     imin = i

imin = imin + int(0.95 * zz_up.shape[0])

img_tot_f = img_tot.copy()[int(1.05 * imax):int(imin), :, :] # ritaglio dell'immagine

# Salva il panorama finale
img_path = os.path.join(img_folder_output, 'output/panorama_tot.jpg')
cv.imwrite(img_path, img_tot_f)

print(f'Panorama image created') # log finale

# Visualizza il panorama finale
image_temp_rgb = cv.cvtColor(img_tot_f.astype('float32'), cv.COLOR_BGR2RGB) #trasformazione da BGR a RGB
plt.imshow(image_temp_rgb.astype('uint8'))
plt.show()