import numpy as np
import plotly
import plotly.graph_objects as go

def frankot_chellappa(normals):
    """
    Reconstructs height map (Z) from surface normals using the Frankot-Chellappa algorithm.
    This typically produces a much smoother and more defined global shape than cumulative sums.
    """
    # 1. Get gradients from normals (nx, ny, nz)
    # p = -nx/nz, q = -ny/nz
    nx, ny, nz = normals[:,:,0], normals[:,:,1], normals[:,:,2]
    
    # Avoid division by zero
    nz_safe = np.where(nz == 0, 1e-5, nz)
    p = -nx / nz_safe
    q = -ny / nz_safe
    
    # Mask out background (where normals are zero)
    mask = (nz != 0).astype(float)
    p *= mask
    q *= mask

    # 2. Fourier Transform of gradients
    rows, cols = p.shape
    P = np.fft.fft2(p)
    Q = np.fft.fft2(q)
    
    # 3. Frequency grid
    # (wx, wy) range from -pi to pi
    wx, wy = np.meshgrid(
        np.fft.fftfreq(cols) * 2 * np.pi,
        np.fft.fftfreq(rows) * 2 * np.pi
    )
    
    # 4. Integration in Frequency Domain
    # Equation: Z_hat = (-j * wx * P - j * wy * Q) / (wx^2 + wy^2)
    # The term (wx^2 + wy^2) acts as an inverse Laplacian filter
    numerator = -1j * wx * P - 1j * wy * Q
    denominator = wx**2 + wy**2
    
    # Avoid division by zero at DC component (0,0)
    denominator[0, 0] = 1.0 
    
    Z_hat = numerator / denominator
    Z_hat[0, 0] = 0  # Force mean height to 0
    
    # 5. Inverse Fourier Transform to get Z
    Z = np.real(np.fft.ifft2(Z_hat))
    
    return Z * mask


def image_renderer(S_r, albedo_estimate):
    plotly.io.renderers.default = "browser"

    height_map = frankot_chellappa(S_r)

    # --- 1. Create Background Mask ---
    # We define background as pixels where the albedo (brightness) is very dark.
    THRESHOLD = 0.1  # Adjust this: Higher = removes more edges, Lower = keeps more background

    # Calculate intensity (handle RGB or Grayscale input)
    if albedo_estimate.ndim == 3:
        intensity = np.mean(albedo_estimate, axis=2)
    else:
        intensity = albedo_estimate

    # Create a boolean mask (True = Cat, False = Background)
    cat_mask = intensity > THRESHOLD


    # --- 2. Prepare Data ---
    # Get height map from previous step
    z_data = height_map.copy()

    # Apply the mask: Set background pixels to NaN so Plotly ignores them
    z_data[~cat_mask] = np.nan 


    # --- 3. Prepare Color ---
    # (Same logic as before, just ensuring dimensions match)
    if albedo_estimate.ndim == 3:
        surface_color = albedo_estimate.copy()
        # Normalize if needed
        if surface_color.max() > 1.0: surface_color /= 255.0
    else:
        surface_color = np.stack([albedo_estimate]*3, axis=-1)

    # Optional: You can also NaN out the color, though masking Z is usually enough
    surface_color[~cat_mask] = np.nan


    # --- 4. Render ---
    fig = go.Figure(data=[go.Surface(
        z=z_data, 
        surfacecolor=surface_color,
        colorscale='Viridis',
        
        # Use 'connectgaps=False' to ensure NaNs create holes, not stretched polygons
        connectgaps=False, 
        
        lighting=dict(
            ambient=0.4, 
            diffuse=0.5, 
            roughness=0.9, 
            specular=0.1, 
            fresnel=0.2
        )
    )])

    fig.update_layout(
        title='3D Reconstruction (Frankot-Chellappa)',
        autosize=True,
        width=800, height=800,
        scene=dict(
            # 'manual' mode allows us to force specific ratios
            aspectmode='manual', 
            # HERE IS THE FIX: Increase 'z' to e.g., 0.5 or 1.0 to emphasize height
            aspectratio=dict(x=1, y=1, z=0.4), 
            camera=dict(
                eye=dict(x=0, y=-0.1, z=2.5), # Top-down view
                up=dict(x=0, y=0, z=0)
            ),
            zaxis=dict(range=[height_map.min(), height_map.max()]) # Tights bounds
        )
    )

    fig.show()

