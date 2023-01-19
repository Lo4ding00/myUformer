# Import Image from wand.image module
from wand.image import Image
  
# Read image using Image() function
with Image(filename ="my_image.png") as img:
  
    # Generate noise image using spread() function
    img.noise("gaussian", attenuate = 1)
    img.save(filename ="sp_noise.png")