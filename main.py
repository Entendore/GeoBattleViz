import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import interp1d

# --- Simulation parameters ---
time_steps = 100
num_units = 5
np.random.seed(42)

# Base paths (without terrain effect)
army1_pos = np.cumsum(np.random.randn(time_steps, num_units, 2)*0.1 + 0.5, axis=0)
army2_pos = np.cumsum(np.random.randn(time_steps, num_units, 2)*0.1 - 0.5 + 5, axis=0)

# Random health and attack
army1_health = np.random.randint(50,101,num_units)
army2_health = np.random.randint(50,101,num_units)
army1_attack = np.random.randint(1,5,num_units)
army2_attack = np.random.randint(1,5,num_units)

# --- Terrain map with elevation ---
terrain_size = (50,50)
elevation = np.zeros(terrain_size)
# Mountains
elevation[30:40,30:40] = 1.0
# Hills
elevation[10:20,5:15] = 0.5
# Roads (faster)
roads = [(5, slice(None)), (25, slice(None))]

# Terrain speed modifiers based on elevation
terrain_speed = np.ones(terrain_size)
terrain_speed[elevation==0.5] = 0.6  # hills slow down
terrain_speed[elevation==1.0] = 0.3  # mountains very slow
for r in roads:
    terrain_speed[r] = 1.5  # roads speed up

# Apply terrain speed
def apply_terrain(pos):
    new_pos = pos.copy()
    for i in range(pos.shape[1]):  # units
        for t in range(1,pos.shape[0]):
            x_idx = int(np.clip((pos[t-1,i,1]+2)/14*terrain_size[1],0,terrain_size[1]-1))
            y_idx = int(np.clip((pos[t-1,i,0]+2)/14*terrain_size[0],0,terrain_size[0]-1))
            speed = terrain_speed[y_idx,x_idx]
            new_pos[t,i] = new_pos[t-1,i] + (pos[t,i]-pos[t-1,i])*speed
    return new_pos

army1_pos = apply_terrain(army1_pos)
army2_pos = apply_terrain(army2_pos)

# Smooth paths
def smooth_path(positions, factor=10):
    x = positions[:,:,1]
    y = positions[:,:,0]
    x_smooth = np.array([interp1d(np.arange(len(x)), x[:,i], kind='cubic')(np.linspace(0,len(x)-1,len(x)*factor)) for i in range(num_units)]).T
    y_smooth = np.array([interp1d(np.arange(len(y)), y[:,i], kind='cubic')(np.linspace(0,len(y)-1,len(y)*factor)) for i in range(num_units)]).T
    return np.stack((y_smooth, x_smooth), axis=2)

army1_smooth = smooth_path(army1_pos)
army2_smooth = smooth_path(army2_pos)
smooth_steps = army1_smooth.shape[0]

# --- Map setup ---
fig = plt.figure(figsize=(12,9))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.OCEAN, zorder=0)
ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=1)
ax.add_feature(cfeature.LAKES, alpha=0.5, zorder=1)
ax.add_feature(cfeature.RIVERS, zorder=1)

# --- Elevation shading ---
# Use a colormap to represent elevation
elev_cmap = plt.cm.terrain
ax.imshow(elevation, extent=[-2,12,-2,12], origin='lower', cmap=elev_cmap, alpha=0.6, zorder=0)

# Cities
cities = np.array([[2,2],[7,3],[5,8]])
ax.scatter(cities[:,1], cities[:,0], color='red', s=80, marker='o', label='City', zorder=3)

# Units
army1_dots, = ax.plot([],[], 'ro', markersize=8, label='Army 1')
army2_dots, = ax.plot([],[], 'bo', markersize=8, label='Army 2')
army1_labels = [ax.text(0,0,"", fontsize=8, color='red') for _ in range(num_units)]
army2_labels = [ax.text(0,0,"", fontsize=8, color='blue') for _ in range(num_units)]
attack_arrows = []

# Heatmap
heatmap_size = (50,50)
heatmap = np.zeros(heatmap_size)
heatmap_img = ax.imshow(
    heatmap, extent=[-2,12,-2,12], origin='lower', cmap='hot', alpha=0.3,
    zorder=0, transform=ccrs.PlateCarree()
)

ax.legend()

# --- Animation function ---
def animate(i):
    for idx in range(num_units):
        army1_dots.set_data(army1_smooth[i,:,1], army1_smooth[i,:,0])
        army2_dots.set_data(army2_smooth[i,:,1], army2_smooth[i,:,0])
        army1_labels[idx].set_position((army1_smooth[i,idx,1]+0.1, army1_smooth[i,idx,0]+0.1))
        army1_labels[idx].set_text(f"H:{army1_health[idx]}")
        army2_labels[idx].set_position((army2_smooth[i,idx,1]+0.1, army2_smooth[i,idx,0]+0.1))
        army2_labels[idx].set_text(f"H:{army2_health[idx]}")

    # Remove old arrows
    for arr in attack_arrows:
        arr.remove()
    attack_arrows.clear()

    # Draw attack arrows
    for idx,u1 in enumerate(army1_smooth[i]):
        distances = np.linalg.norm(army2_smooth[i]-u1, axis=1)
        target = army2_smooth[i][np.argmin(distances)]
        arr = ax.annotate(
            '', xy=(target[1],target[0]), xytext=(u1[1],u1[0]),
            arrowprops=dict(facecolor='yellow', shrink=0.05, width=2*army1_attack[idx], headwidth=8*army1_attack[idx]/2)
        )
        attack_arrows.append(arr)

    # Heatmap
    heatmap.fill(0)
    for unit in np.vstack((army1_smooth[i], army2_smooth[i])):
        x_idx = int((unit[1]+2)/14*heatmap_size[1])
        y_idx = int((unit[0]+2)/14*heatmap_size[0])
        if 0<=x_idx<heatmap_size[1] and 0<=y_idx<heatmap_size[0]:
            heatmap[y_idx,x_idx] += 1
    heatmap_img.set_data(heatmap)

    # Camera panning
    army_center = np.mean(np.vstack((army1_smooth[i], army2_smooth[i])), axis=0)
    center_x = np.mean(army_center[:,1])
    center_y = np.mean(army_center[:,0])
    zoom = 5
    ax.set_xlim(center_x-zoom/2, center_x+zoom/2)
    ax.set_ylim(center_y-zoom/2, center_y+zoom/2)

    return [army1_dots, army2_dots, heatmap_img] + army1_labels + army2_labels + attack_arrows

# --- Animate ---
ani = animation.FuncAnimation(fig, animate, frames=smooth_steps, interval=100, blit=True)

# Export MP4
ani.save("battle_geolayers_elevation.mp4", writer='ffmpeg', fps=10)
plt.show()
