import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from IPython.display import HTML, display

def animate(data):
    frames=len(data)
    fig, ax = plt.subplots()

    def sample_frame(i):
        snapshot = data[i]
        mat.set_data(snapshot)
        return mat
    
    mat = ax.matshow(data[0])
    plt.colorbar(mat)
    animation = matplotlib.animation.FuncAnimation(fig, sample_frame, frames=frames, interval=500)
    plt.close()
    display(HTML(animation.to_html5_video()))