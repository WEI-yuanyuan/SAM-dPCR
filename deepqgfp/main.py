import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from visualize import show
from plot import statsPlot

# Load the model
sys.path.append("..")
sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cpu" # Change this to "cuda" to use the GPU
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

# Adjust these paths as needed
inputDirectory = 'data/test/input'
outputDirectory = 'data/test/output'

# Adjust this factor to make the threshold more restrictive
# A higher factor will result in more negative classifications
threshold_factor = 1.2 

# Run the model and visualize the results
show(inputDirectory=inputDirectory, outputDirectory=outputDirectory, threshold_factor=threshold_factor, mask_generator=mask_generator)
statsPlot(outputDirectory=outputDirectory)

