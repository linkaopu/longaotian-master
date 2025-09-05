from PIL import Image
from pylab import *
import os

# im = array(Image.open('.\\images\\calibrated\\-962.png'))
im = array(Image.open(os.path.join('images','calibrated','2','20220621-100735-108.png')))
imshow(im)
print('Please click 2 points')
imshow(im)
x = ginput(2)
print( 'You clicked:', x)
plt.plot(x)
show()

