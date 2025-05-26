import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors

# Set style for dark background
plt.style.use('dark_background')

# Create a figure with black background and no frame
# Use exact dimensions with no DPI scaling to avoid border issues
fig = plt.figure(figsize=(10, 8), facecolor='black', frameon=False)
# Create axes that completely fill the figure with no padding
ax = fig.add_axes([0, 0, 1, 1], facecolor='black')
# Ensure no spines are visible
for spine in ax.spines.values():
    spine.set_visible(False)

# Create a neural network structure using networkx
def create_network():
    G = nx.DiGraph()
    
    # Create layers
    layers = [4, 6, 6, 4]  # Define the number of nodes in each layer
    pos = {}
    nodes = []
    
    # Create nodes for each layer
    for i, layer_size in enumerate(layers):
        layer_nodes = []
        for j in range(layer_size):
            node_id = f'L{i}_{j}'
            G.add_node(node_id)
            # Position nodes in layers
            pos[node_id] = (i, j - layer_size/2)
            layer_nodes.append(node_id)
        nodes.append(layer_nodes)
    
    # Create edges between layers
    for i in range(len(layers)-1):
        for node1 in nodes[i]:
            for node2 in nodes[i+1]:
                G.add_edge(node1, node2)
    
    return G, pos

# Create the network
G, pos = create_network()

# Initialize edge colors for animation
edge_colors = np.zeros(len(G.edges()))
edge_collection = None

# Animation function
def update(frame):
    global edge_collection
    
    # Use modulo to ensure the animation loops smoothly
    # Using a large number for the modulo to avoid visible resets
    frame = frame % 1000
    
    # Update node positions for subtle movement
    # Slower movement to make it more subtle
    new_pos = {node: (x + 0.05*np.sin(frame/20 + y), y + 0.05*np.cos(frame/20 + x)) 
               for node, (x, y) in pos.items()}
    
    # Clear previous frame
    ax.clear()
    ax.set_facecolor('black')
    
    # Draw nodes with blue color
    nx.draw_networkx_nodes(G, new_pos, node_color="#00D9FF", 
                          node_size=300, alpha=0.8)
    
    # Create edge collection with color gradient
    edges = list(G.edges())
    edge_pos = np.asarray([(new_pos[e[0]], new_pos[e[1]]) for e in edges])
    
    # Create color gradient for edges with slower cycle
    edge_colors = np.zeros((len(edges), 4))
    for i, (u, v) in enumerate(edges):
        # Create flowing animation effect with longer cycle
        color = (frame + i) % 60 / 60  # Increased from 20 to 60 for slower cycle
        # Ensure alpha stays between 0 and 1
        alpha = 0.1 + 0.4 * (np.sin(color * 2 * np.pi) + 1) / 2
        edge_colors[i] = mcolors.to_rgba('cyan', alpha=alpha)
    
    # Draw edges
    edge_collection = LineCollection(edge_pos, colors=edge_colors)
    ax.add_collection(edge_collection)
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set fixed axis limits to prevent auto-scaling
    # Adjust limits to ensure content doesn't touch the edges
    ax.set_xlim(-1.2, 4.2)
    ax.set_ylim(-4.2, 4.2)
    
    return edge_collection,

# Create animation with more frames and slower interval for 120 seconds (2 minutes) total
# 120 seconds * 1000 ms/second / 50 ms per frame = 2400 frames
duration = 10 # in seconds
frame_interval = 50 # in milliseconds
nb_frames = duration * 1000 // frame_interval
anim = FuncAnimation(fig, update, frames=nb_frames, interval=frame_interval, blit=True)

# Save animation as MP4 for iMovie compatibility
# Using ffmpeg writer with high quality settings and transparent background
from matplotlib.animation import FFMpegWriter
writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=5000)

# Ensure no white borders by completely removing all margins and padding
ax.set_axis_off()  # Turn off the axis completely
plt.margins(0, 0)  # Set margins to zero
plt.gca().xaxis.set_major_locator(plt.NullLocator())  # Remove all ticks
plt.gca().yaxis.set_major_locator(plt.NullLocator())

# Save with transparent background and explicit zero padding
anim.save('llm_animation.mp4', writer=writer, 
          savefig_kwargs={
              'transparent': True, 
              'pad_inches': 0,
              'facecolor': 'black'
          })

# Also save as GIF if needed
# anim.save('llm_animation.gif', writer='pillow')

#plt.show()
