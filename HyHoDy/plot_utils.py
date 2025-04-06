import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def generate_contrast_colormap(n_colors=40, emphasize_first=5):
    colors = []
    
    # First approach: For the first few colors, use distinct colormaps
    distinct_cmaps = [plt.cm.viridis, plt.cm.plasma, plt.cm.cool, plt.cm.autumn, plt.cm.tab10, 
                      plt.cm.Set1, plt.cm.Dark2, plt.cm.Set3]
    
    # Get the first few highly distinct colors
    if emphasize_first > 0:
        # Use distinct color points from different colormaps
        for i in range(min(emphasize_first, len(distinct_cmaps))):
            cmap = distinct_cmaps[i]
            # Sample from different parts of each colormap
            point = 0.1 + (0.8 * (i % 3) / 3)  # Vary within the colormap too
            colors.append(cmap(point))
            
        # If we need more distinct colors than we have colormaps
        if emphasize_first > len(distinct_cmaps):
            # Use tab10/Set1 which have distinct categorical colors
            categorical_cmap = plt.cm.tab10
            for i in range(emphasize_first - len(distinct_cmaps)):
                idx = i % 10  # tab10 has 10 distinct colors
                colors.append(categorical_cmap(idx / 10))
    
    # For the remaining colors, use a perceptually uniform colormap with decreasing contrast
    remaining = n_colors - len(colors)
    if remaining > 0:
        # Create non-linear spacing that starts more spread out
        spacing = np.power(np.linspace(0, 1, remaining), 0.7)  # Power < 1 emphasizes early differences
        
        # Use the viridis colormap for the remaining colors
        colors.extend([plt.cm.viridis(point) for point in spacing])
    
    # Convert to hex codes
    hex_colors = [mcolors.to_hex(color) for color in colors]
    
    return hex_colors