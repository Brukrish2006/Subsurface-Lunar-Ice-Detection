import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import uniform_filter, maximum_filter

def main():
    print("Loading 3D data package...")
    try:
        data = np.load('Sverdrup_3D_Data.npz')
        lola = data['lola']
        tier_map = data['tier_map']
        path_y = data['path_y']
        path_x = data['path_x']
    except FileNotFoundError:
        print("Error: Sverdrup_3D_Data.npz not found. Run Lunar_ice_detection_v8_AI.py first.")
        return

    print("Downsampling for 3D performance...")
    DS = 4  # Downsample factor for terrain mesh
    lola_ds = lola[::DS, ::DS]
    
    # We must preserve small ice pixels when downsampling!
    # The user wants ONLY T3 (Yellow) and T4 (Red), and they want them
    # visibly bloated to match the massive scatter plots in the 2D map.
    
    # 1. Isolate the T3 and T4 masks from the high-res map
    t3_mask = (tier_map == 3)
    t4_mask = (tier_map == 4)
    
    # 2. Bloat them independently so Red doesn't swallow Yellow
    t3_bloat = maximum_filter(t3_mask, size=DS * 3)
    t4_bloat = maximum_filter(t4_mask, size=DS * 3)
    
    # 3. Create a clean tier map on the downsampled grid (defaulting to 0=Terrain)
    tier_ds = np.zeros_like(lola_ds, dtype=int)
    
    # Apply bloated T3, then bloated T4 (T4 draws on top where they overlap)
    tier_ds[t3_bloat[::DS, ::DS]] = 3
    tier_ds[t4_bloat[::DS, ::DS]] = 4
    
    # We do NOT smooth it, to preserve whatever LOLA craters exist
    z_data = lola_ds.copy()
    
    # --- DETREND THE SLOPE ---
    # The Sverdrup wall drops 2700 meters over 5km. This massive incline visually squashes 
    # all the 50m-deep craters into a flat "sheet of paper". We will subtract the best-fit 
    # inclined plane so the Z-axis purely represents the craters, ridges, and local depth!
    rows, cols = z_data.shape
    X_grid, Y_grid = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Fit plane: Z = aX + bY + c
    A = np.c_[X_grid.ravel(), Y_grid.ravel(), np.ones(rows*cols)]
    C, _, _, _ = np.linalg.lstsq(A, z_data.ravel(), rcond=None)
    plane = C[0]*X_grid + C[1]*Y_grid + C[2]
    
    z_data_detrended = z_data - plane
    # -------------------------
    
    # No need to build a float color_map. We will pass the integer tier_ds directly 
    # to Plotly and use a strict step-function colorscale to prevent rainbow interpolation!

    # Physical dimensions for Sverdrup crop (from logs: dx~5.24m, dy~2.56m)
    # Since we downsampled by DS, the new pixel spacing is:
    dx = 5.24 * DS
    dy = 2.56 * DS
    
    rows, cols = z_data.shape
    x_coords = np.arange(cols) * dx
    y_coords = np.arange(rows) * dy
    X, Y = np.meshgrid(x_coords, y_coords)

    print("Building 3D terrain surface...")
    surface = go.Surface(
        x=X, y=Y, z=z_data_detrended,
        surfacecolor=tier_ds,
        cmin=0, cmax=4,
        colorscale=[
            [0.0, 'rgb(180, 180, 180)'],   # 0 to 2.9: Terrain
            [0.725, 'rgb(180, 180, 180)'], 
            [0.725, 'rgb(255, 255, 0)'],   # 2.9 to 3.5: Yellow (T3)
            [0.875, 'rgb(255, 255, 0)'],
            [0.875, 'rgb(255, 0, 0)'],     # 3.5 to 4.0: Red (T4)
            [1.0, 'rgb(255, 0, 0)']
        ],
        showscale=False,
        lighting=dict(ambient=0.4, diffuse=1.0, roughness=0.8, specular=0.2, fresnel=0.2)
    )

    print("Overlaying A* rover path...")
    # Map the path onto the physical coordinate grid
    px_meters = (path_x / DS) * dx
    py_meters = (path_y / DS) * dy
    
    # Extract exact Z coordinates for the path so it floats perfectly on the mesh
    # We must also detrend the path Z coordinates using the exact same plane equation!
    # py and px are in the ORIGINAL resolution, so we divide by DS to get the plane coordinates.
    pz_raw = np.array([lola[py, px] for py, px in zip(path_y, path_x)])
    plane_z = C[0]*(path_x/DS) + C[1]*(path_y/DS) + C[2]
    pz_ds = pz_raw - plane_z + 5  # Add +5m offset so it floats above the terrain
    
    path_line = go.Scatter3d(
        x=px_meters, y=py_meters, z=pz_ds,
        mode='lines+markers',
        marker=dict(size=2, color='magenta'),
        line=dict(color='magenta', width=4),
        name='A* Rover Path'
    )
    
    target_marker = go.Scatter3d(
        x=[px_meters[-1]], y=[py_meters[-1]], z=[pz_ds[-1] + 15],
        mode='markers',
        marker=dict(symbol='diamond', size=10, color='red', line=dict(color='white', width=2)),
        name='T4 Target Destination'
    )
    
    landing_marker = go.Scatter3d(
        x=[px_meters[0]], y=[py_meters[0]], z=[pz_ds[0] + 15],
        mode='markers',
        marker=dict(symbol='square', size=8, color='#00ff44', line=dict(color='white', width=2)),
        name='Safe Landing Site'
    )

    # Dummy traces to create a legend for the ice colors
    legend_t3 = go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', marker=dict(size=10, color='rgb(255, 255, 0)'), name='Yellow: T3 High Confidence')
    legend_t4 = go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', marker=dict(size=10, color='rgb(255, 0, 0)'), name='Red: T4 Confirmed Ice')

    layout = go.Layout(
        title='Sverdrup Crater Topography (Slope Detrended to Reveal Craters)',
        scene=dict(
            xaxis=dict(title='Distance East (m)', visible=True, color='white'),
            yaxis=dict(title='Distance North (m)', visible=True, color='white', autorange='reversed'),
            zaxis=dict(title='Local Depth (m)', visible=True, color='white'),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.3), # Exaggerate the local craters heavily!
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.0)
            )
        ),
        paper_bgcolor='black',
        font=dict(color='white'),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.5)"
        )
    )

    fig = go.Figure(data=[surface, path_line, landing_marker, target_marker, legend_t3, legend_t4], layout=layout)
    
    print("Exporting interactive HTML...")
    fig.write_html('Sverdrup_3D_Interactive.html')
    print("Done! Open Sverdrup_3D_Interactive.html in any web browser.")

if __name__ == "__main__":
    main()
