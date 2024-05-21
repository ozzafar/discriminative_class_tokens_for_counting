from PIL import Image, ImageDraw
import random
import math

# Set image dimensions
width, height = 500, 500

# Create a new white image
image = Image.new('RGB', (width, height), 'white')
draw = ImageDraw.Draw(image)

# Number of dots
num_dots = 25

# Dot properties
dot_radius = 30
dot_color = 'blue'

# List to store the positions of the dots
dots = []

def is_valid_position(new_dot):
    for dot in dots:
        dist = math.sqrt((new_dot[0] - dot[0])**2 + (new_dot[1] - dot[1])**2)
        if dist < 2 * dot_radius:
            return False
    return True

while len(dots) < num_dots:
    x = random.randint(dot_radius, width - dot_radius)
    y = random.randint(dot_radius, height - dot_radius)
    if is_valid_position((x, y)):
        dots.append((x, y))
        draw.ellipse((x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius), fill=dot_color)

# Save the image
image.save(f'{num_dots}_dots.png')

# Display the image
image.show()
