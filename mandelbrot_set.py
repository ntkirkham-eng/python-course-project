import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors as mcolors
from PIL import Image
from dataclasses import dataclass
from PIL.ImageColor import getrgb
from math import log10

def run_matplotlib():
    def create_complex_matrix(x_min, x_max, y_min, y_max, pixel_density):
        # Creates a 2D complex matrix using the bounds given and the given pixel density
        real = np.linspace(x_min, x_max, int((x_max - x_min) * pixel_density))
        imaginary = np.linspace(y_min, y_max, int((y_max - y_min) * pixel_density))
         # Uses broadcasting to create a 2D grid of complex numbers
        return real[np.newaxis, :] + imaginary[:, np.newaxis] * 1j

    def is_stable(c, num_iterations):
        # Calculates if a complex value is stable over num_iterations
        # Initializes z with the same shape and type as the input grid
        z = np.zeros_like(c)
        # Tracks which points remain within the set throughout iterations
        mask = np.ones(c.shape, dtype=bool)

        for _ in range(num_iterations):
             # Vectorized update: only compute points that haven't escaped yet
            z[mask] = z[mask] ** 2 + c[mask]
            # Update mask: points are member candidates as long as magnitude <= 2
            mask &= abs(z) <= 2

        return mask
    
    while True:
        try:
            pixel_density = int(input("What pixel density do you want.\n"))
            if pixel_density <= 0:
                    print("Please enter a positive integer for pixel density.")
            else:
                break
        except ValueError:
            print("Please enter a positive integer for pixel density.")

    squares = False
    while True:
        squares_input = input("Enter '1' for squares, '2' for dots.\n").strip()
        if squares_input == '1':
            squares = True
            break
        elif squares_input == '2':
            squares = False
            break
        else:
            print("Invalid input. Please enter either '1' or '2'.")
            
    c = create_complex_matrix(-2, 0.5, -1.5, 1.5, pixel_density)
    stable_values = is_stable(c, 20)

    if squares:
        # Renders the entire boolean mask as a black/white image
        plt.imshow(stable_values, cmap = "binary")
    else:
        # Boolean indexing: Extracts only the coordinates inside the set
        members = c[stable_values]
        # Plots set members as individual points in complex space
        plt.scatter(members.real, members.imag, color = "black", marker = ",", s = 1)

    # Maintains visual proportions and cleans up display
    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

@dataclass
class Viewport:
    image_size: tuple # (width, height)
    center: complex
    width: float

    @property
    def scale(self):
        # Ratio of complex units per pixel
        return self.width / self.image_size[0]

    @property
    def height(self):
        # Calculates height based on width to maintain the image's aspect ratio
        return self.scale * self.image_size[1]
    
    def get_complex_grid(self):
        # Generates horizontal (x) coordinates from left to right (standard)
        x = np.linspace(self.center.real - self.width/2, 
                        self.center.real + self.width/2, 
                        self.image_size[0])
        
        # Generates vertical (y) coordinates from top to bottom (REVERSED)
        # This aligns the complex "top" with Pillow's y=0
        y = np.linspace(self.center.imag + self.height/2, 
                        self.center.imag - self.height/2, 
                        self.image_size[1])
        
         # Creates a 2D coordinate grid (X = real components, Y = imaginary components)
        X, Y = np.meshgrid(x, y)
        return X + 1j * Y

class MandelbrotNumPy:
    def __init__(self, max_iterations, escape_radius=2.0):
        self.max_iterations = max_iterations
        self.escape_radius = escape_radius

    def get_stability(self, c_array, smooth=True, clamp=True):
        # Initializes z with the same shape as the input grid
        z = np.zeros_like(c_array, dtype=np.complex128)
        # Pre-fills with max_iterations for points that never escape (the "set")
        counts = np.full(c_array.shape, self.max_iterations, dtype=np.float64)
        # Boolean mask to only perform math on points that haven't escaped yet
        mask = np.full(c_array.shape, True, dtype=bool)

        for i in range(self.max_iterations):
            z[mask] = z[mask]**2 + c_array[mask]
            
            # Identifies points that just crossed the escape threshold
            # Using squared magnitude avoids a costly square root operation
            escaped = (z.real**2 + z.imag**2 > self.escape_radius**2) & mask
            
            if smooth:
                # Renormalization formula to eliminate color banding
                z_abs = np.abs(z[escaped])
                counts[escaped] = i + 1 - np.log2(np.log(z_abs))
            else:
                counts[escaped] = i
                
            # Removes escaped points from the mask for the next iteration
            mask[escaped] = False
            # Break early if all points have escaped to save time
            if not np.any(mask): break

        # Normalizes counts to a 0.0 - 1.0 scale and applies a gamma of 0.7 to 
        # brighten the "mid-tones" and intricate boundary filaments.
        stability = np.power(counts / self.max_iterations, 0.7)
        return np.clip(stability, 0, 1) if clamp else stability

def run_pillow_black_and_white(image_width, complex_center, viewport_width, max_iterations):
    print("Calculating...")
    
    # Initializes the Mandelbrot calculator and Viewport
    mset = MandelbrotNumPy(max_iterations, escape_radius=2.0)
    view = Viewport((image_width, image_width), complex_center, viewport_width)
    
    # Generates the 2D grid of complex numbers
    c_grid = view.get_complex_grid()
    
    # Calculates stability (values will be 1.0 for points inside the set)
    # Smoothing turned off because it's not needed for pure B&W
    stability = mset.get_stability(c_grid, smooth=False, clamp=False)
    
    # Creates a binary mask: 0 (black) for inside the set, 255 (white) for outside
    # Logic: If stability is 1.0, the point never escaped -> Black (0)
    image_data = np.where(stability == 1.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(image_data)
    img.show()

def get_valid_mpl_palette(n_colors_per_segment=256):
    while True:
        colormap_name = input("Enter a matplotlib color palette name (e.g., 'magma', 'viridis', 'inferno'): ").strip()
        try:
            # Accesses the colormap registry
            cmap = mpl.colormaps[colormap_name]
            
            # Checks if it's a discrete (Listed) or continuous (LinearSegmented) palette
            if hasattr(cmap, "colors"):
                # For discrete palettes (like 'tab10' or 'Set1'), use the exact colors
                palette = np.array(cmap.colors)[:, :3] # Slice to RGB
            else:
                # For continuous palettes, if you want a higher color
                # density (256 * (num_colors-1))you can adjust n_colors
                samples = np.linspace(0, 1, n_colors_per_segment)
                palette = cmap(samples)[:, :3] # Slice to RGB
            
            # Denormalizes (0-1 to 0-255) and convert to NumPy uint8
            return (palette * 255).astype(np.uint8)
            
        except KeyError:
            print("Invalid matplotlib colormap. Please try again.")

def get_edge_palette():
    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[:128, :] = 255  # Exterior (White)
    palette[128:243, :] = np.linspace(255, 0, 115)[:, None] # Gray area
    palette[243:, :] = 255  # Interior
    return palette

def get_custom_numpy_palette():
    user_colors = []
    print("Enter CSS color names or hex codes (e.g., 'black', '#ff0055').")
    print("Type 'exit' when you are finished choosing colors.")

    while True:
        color_input = input("Enter color: ").strip().lower()
        if color_input == 'exit':
            if len(user_colors) < 2:
                print("You must enter at least two colors to create a gradient!")
                continue
            break
        try:
            # Gets RGB (0-255) from color name/hex
            rgb = np.array(getrgb(color_input))
            user_colors.append(rgb)
        except ValueError:
            print("Invalid color. Please try again.")

    # Converts list of color stops to a NumPy array (Stops, 3)
    user_colors = np.array(user_colors)
    num_stops = len(user_colors)

    # Calculate high-resolution palette size: 256 * (number of colors - 1)
    num_colors_in_palette = 256 * (num_stops - 1)

    # Defines the positions of the original color stops (0 to 1)
    # Example: 3 colors will have stops at [0.0, 0.5, 1.0]
    stops = np.linspace(0, 1, num_stops)

    # Defines the positions of every color in the new high-res palette
    full_range = np.linspace(0, 1, num_colors_in_palette)

    # Interpolates each RGB channel independently
    # This creates a smooth transition between all user-provided colors
    palette = np.zeros((num_colors_in_palette, 3), dtype=np.uint8)
    for i in range(3):  # 0=Red, 1=Green, 2=Blue
        palette[:, i] = np.interp(full_range, stops, user_colors[:, i])

    return palette

def run_pillow_color(image_width, complex_center, viewport_width, max_iterations):
    palette = None
    while True:
        print("Choose Color Mode: 1: Grayscale, 2: Matplotlib, 3: Edge, 4: Custom, 5: HSB")
        mode = input("Enter choice: ").strip()

        if mode == '1' or mode == '5':
            break  # No palette needed for these modes
        elif mode == '2':
            palette = get_valid_mpl_palette() 
            break
        elif mode == '3':
            palette = get_edge_palette()
            break
        elif mode == '4':
            palette = get_custom_numpy_palette()
            break
        else:
            print("Invalid input. Please enter a number between 1 and 5.")

    print("Calculating...")
    view = Viewport((image_width, image_width), complex_center, viewport_width)
    mset = MandelbrotNumPy(max_iterations, escape_radius=100)
    
    c_grid = view.get_complex_grid()
    stability = mset.get_stability(c_grid, smooth=True)

    if mode == '1':
        # Inverts stability so points in the set (1.0) become black (0) 
        # and points outside become shades of white (255)
        image_data = ((1 - stability) * 255).astype(np.uint8)
        # Creates image; Pillow auto-detects Grayscale (Mode 'L') from the uint8 2D array shape
        img = Image.fromarray(image_data)
    elif mode == '5':
        # Creates an HSB array by stacking 3 layers: [Hue, Saturation, Brightness]
        # Hue and Saturation are driven by stability; Brightness is set to 1.0 (full)
        hsv_image = np.stack([stability, stability, np.ones_like(stability)], axis=-1)
        # Identifies points inside the Mandelbrot set (stability == 1.0) and sets them to black
        hsv_image[stability == 1.0] = [0, 0, 0]
        # Converts the entire 3D HSV array to RGB and scales to 0-255 range
        rgb_data = (mcolors.hsv_to_rgb(hsv_image) * 255).astype(np.uint8)
        # Creates image; Pillow auto-detects RGB mode from the uint8 (H, W, 3) array shape
        img = Image.fromarray(rgb_data)
    else:
        # Maps stability values (0.0 - 1.0) to integer indices for the color palette
        indices = (stability * (len(palette) - 1)).astype(np.int32)
        # Advanced Indexing: Replaces every index with its corresponding RGB triplet from the palette
        # This creates a (Height, Width, 3) color array simultaneously
        rgb_data = palette[indices]
        # Creates image; Pillow auto-detects RGB mode from the uint8 data and 3D array shape
        img = Image.fromarray(rgb_data)

    img.show()

def run_pillow():
    while True:
        try:
            image_width = int(input("What image size do you want.\n"))
            if image_width <= 0:
                    print("Please enter a positive integer for image size.")
            else:
                break
        except ValueError:
            print("Please enter a positive integer for image size.")

    while True:
        try:
            real_center = float(input("Where do you want the real center to be.\n"))
            break
        except ValueError:
            print("Please enter an integer or float for the real center.")

    while True:
        try:
            imaginary_center = float(input("Where do you want the imaginary center to be.\n"))
            break
        except ValueError:
            print("Please enter an integer or float for the imaginary center.")

    complex_center = complex(real_center, imaginary_center)

    while True:
        try:
            viewport_width = float(input("What width of the area would you like to display.\n"))
            if viewport_width <= 0:
                    print("Please enter a positive number for the width of the area to display.")
            else:
                break
        except ValueError:
            print("Please enter a positive number for the width of the area to display.")

    # Calculates iterations dynamically: increases detail as the viewport shrinks (zoom increases)
    # base_iterations (150) is scaled by the log of the zoom level
    max_iterations = min(int(150 * (1 + log10(1.0 / viewport_width))), 50000) # Capped at 50k to prevent lag
    
    black_and_white = False
    while True:
        black_and_white_input = input("Enter '1' for black and white, '2' for color.\n").strip()
        if black_and_white_input == '1':
            black_and_white = True
            break
        elif black_and_white_input == '2':
            black_and_white = False
            break
        else:
            print("Invalid input. Please enter either '1' or '2'.")
            
    if black_and_white:
        run_pillow_black_and_white(image_width, complex_center, viewport_width, max_iterations)
    else:
        run_pillow_color(image_width, complex_center, viewport_width, max_iterations)
    
while True:
    print("Choose Mode: 1: Matplotlib, 2: Pillow, 3: Help, 4: Exit")
    mode = input("Enter mode: ").strip()

    if mode == '1':
        run_matplotlib()
        break
    elif mode == '2':
        run_pillow()
        break
    elif mode == '3':
        print("Matplotlib (key '1') creates an interactive graph")
        print("Pillow (key '2') creates and displays an image")
    elif mode == '4':
        break
    else:
        print("Invalid input. Please try again or type '3' for help.")