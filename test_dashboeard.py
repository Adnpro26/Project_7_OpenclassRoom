import os
from PIL import Image

# Obtenir le chemin absolu du dossier contenant le script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construire le chemin absolu pour le fichier image
image_path = os.path.join(script_dir, 'logo_image.jpg')


print(f"L'image est chargée depuis : {image_path}")
