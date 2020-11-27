from read_roi import read_roi_zip
from PIL import Image
import numpy as np
from matplotlib.path import Path
import cv2
import matplotlib.pyplot as plt

path = 'Rois/'

ROIS = read_roi_zip(path + '4402_01.zip')
im = Image.open(path + '4402_01.tif')

#Michał
#Utworzenie maski z pliku ROI
w, h = im.size
maskSum = 0
sumaKrawedzi = 0
iteracja = 0

def erozja(maska):
  #Maks
  #Pierwsza erozja maski przy użyciu elementu [[1,1,1],[1,1,1],[1,1,1]]
  kernel_one = np.ones((3,3),np.uint8)
  firstErosionImage = cv2.erode(maska,kernel_one)
  #Druga erozja maski przy użyciu elementu [[0,1,0],[1,1,1],[0,1,0]]
  kernel_two = np.array([[0,1,0], [1,1,1], [0,1,0]], np.uint8)
  secondErosionImage = cv2.erode(firstErosionImage,kernel_two)
  #Odjęcie od pierwotnej maski nowej zerodowanej maski w celu utworzenia pierwszej części krawędzi jądra oraz wnętrza jądra komórkowego
  nucleus_border = maska - secondErosionImage
  return nucleus_border

def dylatacja(maska):
  #Michał
  #Pojedyncza dylatacja maski pierwotnej
  kernel_one = np.ones((3,3),np.uint8)
  dilation = cv2.dilate(maska,kernel_one,iterations = 2)

  #Odjęcie nowej maski od maski pierwotnej w celu uzyskania drugiej części krawędzi jądra komórkowego
  dilation_border = maska - dilation

  dilation_border[dilation_border >= 250] = 255
  dilation_border[dilation_border < 250] = 0
  return dilation_border

def sumowanieKrawedzi(wynikErozji, wynikDylatacji):
  borders = wynikErozji + wynikDylatacji
  return borders

print(len(ROIS))
for k in ROIS.values():
  x = k.get('x')
  y = k.get('y')
  #liczba współrzędnych k-tego ROI
  n = len(x)
  #generowanie współrzędnych poligonu
  i = 0
  ListOfCorners = []
  for i in range(i,n):
    ListOfCorners.append((int((y[i])), int((x[i]))))
  #generowanie masek poligonów
  poly_path = Path(ListOfCorners)
  Nx, Ny = np.mgrid[:h, :w]
  coordinates = np.hstack((Nx.reshape(-1, 1), Ny.reshape(-1, 1)))
  mask = poly_path.contains_points(coordinates)
  mask = mask.reshape(h, w)
  mask = np.array(mask, dtype=bool)
  mask = 255 * mask
  mask = np.array(mask, dtype='uint8')

  wynikErozji = erozja(mask)
  wynikDylatacji = dylatacja(mask)

  sumaKrawedzi += sumowanieKrawedzi(wynikErozji, wynikDylatacji)

  maskSum += mask
  print("\rProgress: " + str(round(((iteracja+1.0) / len(ROIS)) * 100, 1)), end="%")
  iteracja += 1
print("\n")

sumaKrawedzi[sumaKrawedzi >= 250] = 128

print(np.unique(sumaKrawedzi))

imgSharpening = Image.fromarray(np.uint8(maskSum))
zmienna = imgSharpening - sumaKrawedzi

print(np.unique(zmienna))

zmienna[zmienna >= 253] = 255

print(np.unique(zmienna))
zmiennaImage = Image.fromarray(np.uint8(zmienna))
zmiennaImage.save(path + 'newImage.tif')

fig = plt.figure()
fig.set_size_inches(10, 10)
ax1 = fig.add_subplot(1,1,1)
ax1.imshow(zmiennaImage)
ax1.set_title("MASK")