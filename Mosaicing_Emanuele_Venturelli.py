# Progetto Computer Vision
# Titolo: Mosaicing
# Autore: Emanuele Venturelli - matricola VR509826
# Ultima modifica: 16/02/2025

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
counter = 0
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
status = []
scale_percent = 50  # Riduci la dimensione al 50%

# Caricamento delle immagini di input
for filename in os.listdir(img_folder_input):
        if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(img_folder_input, filename)
                imgn = cv.imread(img_path)
                width = int(imgn.shape[1] * scale_percent / 100)

                # Riduzione risoluzione dell'immagine per caloclo più veloce
                height = int(imgn.shape[0] * scale_percent / 100)
                dim = (width, height)
                immagini_input.append(cv.resize(imgn, dim, interpolation=cv.INTER_AREA))


################################  ---  ALGORITMO DA SX A DX  ---  #################################################

contatore = (img_0.copy()).astype('int')
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

                # Conversione dei keypoint in float32 per calcoli geometrici
                keypoints_train_image = np.float32([keypoint.pt for keypoint in kp_A])
                keypoints_query_image = np.float32([keypoint.pt for keypoint in kp_B])

                if len(matches) >= 4:
                        pointsBB = np.float32([keypoints_train_image[m.queryIdx] for m in matches])
                        pointsAA = np.float32([keypoints_query_image[m.trainIdx] for m in matches])

                        # Calcolo dell'omografia usando RANSAC
                        #errortresh = 5 #soglia di errore
                        H, status = ransac_homography(pointsAA, pointsBB, num_iterations=300, threshold=5.0)

                else:
                        print(f'Minimum number of matches not reached. It is not possible to calculate homography matrix')

                # Dimensioni del panorama e calcolo della maschera di fusione
                height_img_A = immagini_input[contatore].shape[0]
                width_img_A = immagini_input[contatore].shape[1]
                width_img_B = immagini_input[contatore+1].shape[1]
                height_panorama = height_img_A
                width_panorama = width_img_A + width_img_B
                lowest_width = min(width_img_A, width_img_B)

                # Inizio filtro per transizione tra un'immagine e l'altra

                smoothing_window_percent = 0.10 # Indice compreso tra 0.00 e 1.00. che corriponde alla percentuale di area di immagine usata per il filtro di transizione
                smoothing_window_size = max(100, min(smoothing_window_percent * lowest_width, 1000)) # Sarà sicuramente un numero compreso tra 100 e 1000

                offset = int(smoothing_window_size / 2)

                barrier = immagini_input[contatore].shape[1] - int(smoothing_window_size / 2)

                mask_A = np.zeros((height_panorama, width_panorama))
                mask_A[:, barrier - offset : barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset).T, (height_panorama, 1))
                mask_A[:, : barrier - offset] = 1
                mask_A1 = cv.merge([mask_A, mask_A, mask_A])


                mask_B = np.zeros((height_panorama, width_panorama))
                mask_B[:, barrier - offset : barrier + offset] = np.tile(np.linspace(0, 1, 2 * offset).T, (height_panorama, 1))
                mask_B[:, barrier + offset :] = 1
                mask_B1 = cv.merge([mask_B, mask_B, mask_B])

                # Warping e fusione delle immagini
                img_rtd = cv.warpPerspective(immagini_input[contatore+1], H, (width_panorama, height_panorama))
                panorama1 = np.zeros(((height_panorama), (width_panorama), 3))
                panorama1[0 : immagini_input[contatore].shape[0], 0 : immagini_input[contatore].shape[1], :] = immagini_input[contatore]
                panorama_final = panorama1*mask_A1 + img_rtd*mask_B1
                
                # Ritaglio delle bande nere
                rows, cols = np.where(panorama_final[:, :, 0] != 0) #Trova le bande nere sulla dx
                min_row, max_row = min(rows), max(rows) + 1
                min_col, max_col = min(cols), max(cols) + 1

                pixel = [0, 0, 0]
                maxcol_1 = max_col
                maxcol_2 = max_col

                while pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
                        maxcol_1 = maxcol_1 - 1
                        pixel = panorama_final[min_row, maxcol_1]


                pixel = [0, 0, 0]

                while pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
                        maxcol_2 -= 1
                        pixel = panorama_final[max_row-1, maxcol_2]

                if maxcol_1 > maxcol_2:
                        col_estrema = maxcol_2
                else:
                        col_estrema = maxcol_1

                img_temp = ((panorama_final[min_row:max_row, min_col:col_estrema, :]).astype('uint8')).copy()

                contatore += 1

                print('Iteration number:', contatore_ind)

        else:
                # Conversione in scala di grigi delle immagini

                gray_A= cv.cvtColor(img_temp.copy()[0:height, (img_temp.shape[1]-width):img_temp.shape[1]], cv.COLOR_BGR2GRAY)
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
                        #errortresh = 5.0  # Soglia di errore
                        H, status = ransac_homography(pointsAA, pointsBB, num_iterations=300, threshold=5.0)

                else:
                        print(f'Minimum number of matches not reached. It is not possible to calculate homography matrix')

                # Dimensioni del panorama e calcolo della maschera di fusione
                height_img_A = img_temp.shape[0]
                width_img_A = img_temp.shape[1]
                width_img_B = immagini_input[contatore+1].shape[1]
                height_panorama = height_img_A
                width_panorama = width_img_A + width_img_B
                lowest_width = min(width_img_A, width_img_B)

                # Inizio filtro per transizione tra un'immagine e l'altra

                smoothing_window_percent = 0.10 # Indice compreso tra 0.00 e 1.00. che corriponde alla percentuale di area di immagine usata per il filtro
                smoothing_window_size = max(100, min(smoothing_window_percent * lowest_width, 1000)) #sarà sicuramente un numero compreso tra 100 e 1000

                offset = int(smoothing_window_size / 2)

                barrier = img_temp.shape[1] - int(smoothing_window_size / 2)

                mask_A = np.zeros((height_panorama, width_panorama))
                mask_A[:, barrier - offset : barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset).T, (height_panorama, 1)) #crea un array che va 1 a 0 con tanti elementi come offset*2. alto come l'immagine
                mask_A[:, : barrier - offset] = 1
                mask_A1 = cv.merge([mask_A, mask_A, mask_A])


                mask_B = np.zeros((height_panorama, width_panorama))
                mask_B[:, barrier - offset : barrier + offset] = np.tile(np.linspace(0, 1, 2 * offset).T, (height_panorama, 1))
                mask_B[:, barrier + offset :] = 1
                mask_B1 = cv.merge([mask_B, mask_B, mask_B])

                # Warping e fusione delle immagini
                img_rtd = cv.warpPerspective(immagini_input[contatore+1], H, (width_panorama, height_panorama)) #ruota la seconda immagine in prospettiva con la prima

                larg = (img_rtd.shape[1] + img_temp.shape[1] - width)

                img_rtd_n = np.zeros(((img_rtd.shape[0]), (larg), 3))
                img_rtd_n[0 : img_rtd.shape[0], (img_temp.shape[1] - width):(img_rtd.shape[1] + img_temp.shape[1] - width), :] = img_rtd
                img_rtd_p = img_rtd_n.copy()[:, 0:img_rtd.shape[1], :]


                panorama1 = np.zeros(((height_panorama), (width_panorama), 3))
                panorama1[0 : img_temp.shape[0], 0 : img_temp.shape[1], :] = img_temp
                panorama_final = panorama1*mask_A1 + img_rtd_p*mask_B1

                # Ritaglio delle bande nere
                rows, cols = np.where(panorama_final[:, :, 0] != 0) # Trova le bande nere sulla dx
                min_row, max_row = min(rows), max(rows) + 1
                min_col, max_col = min(cols), max(cols) + 1

                pixel = [0, 0, 0]

                maxcol_1 = max_col
                maxcol_2 = max_col

                while pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
                        maxcol_1 = maxcol_1 - 1
                        pixel = panorama_final[min_row, maxcol_1]


                pixel = [0, 0, 0]

                while pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
                        maxcol_2 -= 1
                        pixel = panorama_final[max_row-1, maxcol_2]

                if maxcol_1 > maxcol_2:
                        col_estrema = maxcol_2
                else:
                        col_estrema = maxcol_1

                img_temp = ((panorama_final[min_row:max_row, min_col:col_estrema, :]).astype('uint8')).copy()

                contatore += 1
                contatore_ind += 1

                print('Iteration number:', contatore_ind)
                H = None


################################  ---  ALGORITMO DA DX A SX  ---  #################################################

# Definizione contatori
contatore = (img_0.copy()).astype('int')
contatore_f = contatore.copy()

while contatore > 0:

        # Flippa orizzontalmente le immagini temporanea e corrente
        img_temp_flipped = cv.flip(img_temp.copy(), 1)
        immagini_input_flipped = cv.flip(immagini_input[contatore-1].copy(), 1)
        
        # Converte le immagini in scala di grigi per l'elaborazione
        gray_A= cv.cvtColor(img_temp_flipped.copy()[0:height, (img_temp_flipped.shape[1]-width):img_temp_flipped.shape[1]], cv.COLOR_BGR2GRAY)
        gray_B= cv.cvtColor(immagini_input_flipped.copy(), cv.COLOR_BGR2GRAY)
        
        # Rileva keypoint e descrittori usando SIFT
        kp_A, descriptor_A = sift.detectAndCompute(gray_A,None)
        kp_B, descriptor_B = sift.detectAndCompute(gray_B,None)

        # Trova i match tra i descrittori e li ordina per distanza
        matches = sorted((bf.match(descriptor_A, descriptor_B)), key = lambda x:x.distance)

        # Estrae le coordinate dei keypoint corrispondenti
        keypoints_train_image = np.float32([keypoint.pt for keypoint in kp_A])
        keypoints_query_image = np.float32([keypoint.pt for keypoint in kp_B])

        if len(matches) >= 4: # Controlla che ci siano almeno 4 match per calcolare l'omografia
                pointsBB = np.float32([keypoints_train_image[m.queryIdx] for m in matches])
                pointsAA = np.float32([keypoints_query_image[m.trainIdx] for m in matches])

                # Calcola l'omografia usando il metodo RANSAC
                #errortresh = 5
                H, status = ransac_homography(pointsAA, pointsBB, num_iterations=300, threshold=5.0)

        else:
                print(f'Minimum number of matches not reached. It is not possible to calculate homography matrix')

        # Definisce le dimensioni del panorama risultante
        height_img_A = img_temp_flipped.shape[0]
        width_img_A = img_temp_flipped.shape[1]
        width_img_C = immagini_input_flipped.shape[1]
        height_panorama = height_img_A
        width_panorama = width_img_A + width_img_C
        lowest_width = min(width_img_A, width_img_C)

        # Inizio filtro per transizione tra un'immagine e l'altra

        smoothing_window_percent = 0.10 # Indice compreso tra 0.00 e 1.00. che corriponde alla percentuale di area di immagine usata per il filtro
        smoothing_window_size = max(100, min(smoothing_window_percent * lowest_width, 1000)) # Sarà sicuramente un numero compreso tra 100 e 1000

        offset = int(smoothing_window_size / 2)

        barrier = img_temp_flipped.shape[1] - int(smoothing_window_size / 2)

        # Crea la maschera per la prima immagine
        mask_A = np.zeros((height_panorama, width_panorama))
        mask_A[:, barrier - offset : barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset).T, (height_panorama, 1)) # Crea un array che va 1 a 0 con tanti elementi come offset*2 e alto come l'immagine
        mask_A[:, : barrier - offset] = 1
        mask_A1 = cv.merge([mask_A, mask_A, mask_A])

        # Crea la maschera per la seconda immagine
        mask_C = np.zeros((height_panorama, width_panorama))
        mask_C[:, barrier - offset : barrier + offset] = np.tile(np.linspace(0, 1, 2 * offset).T, (height_panorama, 1))
        mask_C[:, barrier + offset :] = 1
        mask_C1 = cv.merge([mask_C, mask_C, mask_C])

        # Applica la trasformazione prospettica alla seconda immagine
        img_rtd = cv.warpPerspective(immagini_input_flipped, H, (width_panorama, height_panorama)) # Ruota la seconda immagine in prospettiva con la prima

        larg = (img_rtd.shape[1] + img_temp_flipped.shape[1] - width)

        img_rtd_n = np.zeros(((img_rtd.shape[0]), (larg), 3))
        img_rtd_n[0 : img_rtd.shape[0], (img_temp_flipped.shape[1] - width):(img_rtd.shape[1] + img_temp_flipped.shape[1] - width), :] = img_rtd
        img_rtd_p = img_rtd_n.copy()[:, 0:img_rtd.shape[1], :]

        # Prepara il panorama combinando le immagini con le rispettive maschere
        panorama1 = np.zeros(((height_panorama), (width_panorama), 3))
        panorama1[0 : img_temp_flipped.shape[0], 0 : img_temp_flipped.shape[1], :] = img_temp_flipped
        panorama_final = panorama1*mask_A1 + img_rtd_p*mask_C1

        rows, cols = np.where(panorama_final[:, :, 0] != 0) # Trova le bande nere sulla dx
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1

        # Identifica il limite dell'immagine valida
        pixel = [0, 0, 0]
        maxcol_1 = max_col
        maxcol_2 = max_col

        while pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
                maxcol_1 = maxcol_1 - 1
                pixel = panorama_final[min_row, maxcol_1]


        pixel = [0, 0, 0]

        while pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
                maxcol_2 -= 1
                pixel = panorama_final[max_row-1, maxcol_2]

        if maxcol_1 > maxcol_2:
                col_estrema = maxcol_2
        else:
                col_estrema = maxcol_1

        # Flippa il panorama ottenuto e lo salva come immagine temporanea
        img_temp = cv.flip(((panorama_final[min_row:max_row, min_col:col_estrema, :]).astype('uint8')).copy(), 1)

        # Aggiorna i contatori
        contatore -= 1
        contatore_f += 1

        print('Iteration number:', contatore_f)
        H = None

# Salva il panorama finale
img_path = os.path.join(img_folder_output, 'output/panorama_tot.jpg')
cv.imwrite(img_path, img_temp)

print(f'Panorama image created')

# Visualizza il panorama finale
image_temp_rgb = cv.cvtColor(img_temp, cv.COLOR_BGR2RGB)
plt.imshow(image_temp_rgb)
plt.show()