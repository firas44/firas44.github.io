# Install required libraries (if not already installed)
# !pip install animeganv2 opencv-python

import cv2
from animeganv2 import AnimeGANv2

# Load the pre-trained AnimeGANv2 model
gan = AnimeGANv2()

# Function to load and pre-process the image
def load_image(path):
  img = cv2.imread(path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for model compatibility
  img = cv2.resize(img, (256, 256))  # Resize to model input size
  return img

# Get the image path (replace 'path/to/your/image.jpg' with your actual path)
image_path = 'path/to/your/image.jpg'

# Load the image
img = load_image(image_path)

# Convert the image to cartoon
cartoon = gan.predict(img)

# Convert back to BGR for OpenCV display
cartoon = cv2.cvtColor(cartoon[0], cv2.COLOR_RGB2BGR)

# Display the original and cartoon images
cv2.imshow('Original Image', img)
cv2.imshow('Cartoonized Image', cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the cartoon image
# cv2.imwrite('cartoonized.jpg', cartoon)

print("Cartoonization complete!")