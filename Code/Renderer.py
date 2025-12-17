import numpy as np
import plotly
import plotly.graph_objects as go

def frankot_chellappa(surface_normals):
    """
    Reconstruct depth map (Z) from surface normals using the Frankot-Chellappa algorithm.

    Parameters:
        surface_normals (numpy.ndarray): unnormalized surface normals of shape (H, W, 3).

    Returns:
        depth_map (numpy.ndarray): reconstructed depth map of shape (H, W).
    """

    # 1. get gradients from surface normals (dx, dy, dz)
    dx, dy, dz = surface_normals[:, :, 0], surface_normals[:, :, 1], surface_normals[:, :, 2]

    # 2. compute p = -dx/dz and q = -dy/dz
    p = np.divide(dx, dz, where=(dz != 0)) # avoid division by zero
    q = np.divide(dy, dz, where=(dz != 0)) # avoid division by zero

    # 3. mask out background (where normals are zero)
    mask = (dz != 0).astype(float)
    p *= mask
    q *= mask

    # 4. Fourier transform of gradients p and q
    p_fft = np.fft.fft2(p)
    q_fft = np.fft.fft2(q)

    # 5. create frequency grid
    num_rows, num_cols = p.shape
    wx, wy = np.meshgrid(
        np.fft.fftfreq(num_cols) * 2 * np.pi,
        np.fft.fftfreq(num_rows) * 2 * np.pi
    )

    # 6. integrate in the frequency domain
    # equation: Z_hat = (-j * wx * P - j * wy * Q) / (wx^2 + wy^2)
    # the term (wx^2 + wy^2) acts as an inverse Laplacian filter
    numerator = -1j * wx * p_fft - 1j * wy * q_fft
    denominator = wx**2 + wy**2
    denominator[0, 0] = 1.0 # avoid division by zero at denominator component (0, 0)
    Z_hat = numerator / denominator
    Z_hat[0, 0] = 0 # force that the mean height is 0

    # 7. inverse Fourier transform to get Z
    Z = np.real(np.fft.ifft2(Z_hat))

    # 8. apply mask to Z
    depth_map = Z * mask

    return depth_map

def render_depth_map(depth_map, albedo_estimate, albedo_threshold):
    plotly.io.renderers.default = "browser"

    # 1. create background image using the albedo estimate
    intensity = np.mean(albedo_estimate, axis=2) # compute intencity value for each pixel
    background_mask = intensity < albedo_threshold # True = Background, False = Foreground

    # 2. prepare depth map
    depth_map = depth_map.copy()
    depth_map[background_mask] = np.nan # set background pixels to nan such that Plotly ignores them

    # 3. prepare color
    surface_color = albedo_estimate.copy()
    if surface_color.max() > 1.0: surface_color /= 255.0 # normalize if needed

    # 4. render depth map
    fig = go.Figure(data=[go.Surface(
        z=depth_map, 
        surfacecolor=surface_color,
        colorscale='Viridis',
        lighting=dict(
            ambient=0.4, 
            diffuse=0.5, 
            roughness=0.9, 
            specular=0.1, 
            fresnel=0.2
        )
    )])

    # update layout
    fig.update_layout(
        title='Depth Map Reconstruction (Frankot-Chellappa)',
        autosize=True,
        width=800, height=800,
        scene=dict(
            aspectmode='manual', 
            aspectratio=dict(x=1, y=1, z=0.4), 
            camera=dict(
                eye=dict(x=0, y=-0.1, z=2.5), # Top-down view
                up=dict(x=0, y=0, z=0)
            ),
            zaxis=dict(range=[depth_map.min(), depth_map.max()]) # Tight bounds
        )
    )

    fig.show()
