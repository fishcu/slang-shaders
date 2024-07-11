import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

def hash13(p3):
    p3 = np.fmod(p3 * 0.1031, 1.0)
    p3 += np.dot(p3, p3.T[::-1] + 31.32)
    return np.fmod((p3[0] + p3[1]) * p3[2], 1.0)

def glsl_mod(x, y):
    return x - y * np.floor(x / y)

def anisotropic_filtering_debug(uv, dx, dy, filtering_level, rng_seed, rotation_angle):
    ellipse_approx = 2.0 * np.array([dx, dy]).T
    num_samples = int(2 ** filtering_level)

    max_axis_sq = max(np.dot(dx, dx), np.dot(dy, dy))
    max_axis = np.sqrt(max_axis_sq)
    mip_level = max(0.5 * (np.log2(max_axis_sq) - filtering_level), 0.0)

    random_angle = rotation_angle
    rot = np.array([[np.cos(random_angle), -np.sin(random_angle)],
                    [np.sin(random_angle), np.cos(random_angle)]])

    alpha = np.array([0.7548776662, 0.56984029])

    samples = []
    weights = []
    for i in range(num_samples):
        # offset = glsl_mod(rot.dot(i * alpha - 0.5) + 0.5, 1) - 0.5
        # offset = glsl_mod(rot.dot(glsl_mod(i * alpha, 1) - 0.5) + 0.5, 1) - 0.5
        # offset = glsl_mod(i * alpha, 1)  - 0.5
        offset = rot.dot(glsl_mod(i * alpha, 1) - 0.5)
        t = np.linalg.norm(offset) * 1.41421
        weight = max(0, 1.0 - t * t * (3.0 - 2.0 * t))
        elliptical_offset = ellipse_approx.dot(offset)
        samples.append(elliptical_offset)
        weights.append(weight)

    # i = -1
    # while len(samples) < num_samples:
    #     i += 1
    #     # important to use fmod with 1 here!
    #     offset = glsl_mod(i * alpha, 1)  - 0.5
    #     offset = rot.dot(offset)

    #     # smaller square has sqrt(2) / 4 side length
    #     if abs(offset[0]) > 0.35355339059  or abs(offset[1]) > 0.35355339059:
    #         continue
        
    #     # scale back to [-0.5, 0.5]^2
    #     offset = offset * 1.414
        
    #     t = np.linalg.norm(offset) * 1.41421
    #     weight = max(0, 1.0 - t * t * (3.0 - 2.0 * t))
    #     elliptical_offset = ellipse_approx.dot(offset)
    #     samples.append(elliptical_offset)
    #     weights.append(weight)
        

    return np.array(samples), np.array(weights)

def main():
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.3)

    uv = np.array([0.5, 0.5])
    dx = np.array([1.0, 0.0])
    dy = np.array([0.0, 1.0])
    filtering_level = 4
    rng_seed = np.array([0.1, 0.2, 0.3])
    rotation_angle = 0

    samples, weights = anisotropic_filtering_debug(uv, dx, dy, filtering_level, rng_seed, rotation_angle)

    scatter = ax.scatter(samples[:, 0], samples[:, 1], c=weights, cmap='viridis', s=50)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.grid(True)

    # Create sliders
    ax_filter = plt.axes([0.1, 0.15, 0.8, 0.03])
    ax_rotation = plt.axes([0.1, 0.1, 0.8, 0.03])
    ax_dx = plt.axes([0.1, 0.05, 0.8, 0.03])
    ax_dy = plt.axes([0.1, 0.0, 0.8, 0.03])

    slider_filter = Slider(ax_filter, 'Filtering Level', 0, 8, valinit=filtering_level)
    slider_rotation = Slider(ax_rotation, 'Rotation', 0, 2*np.pi, valinit=rotation_angle)
    slider_dx = Slider(ax_dx, 'dx', -0.5, 0.5, valinit=dx[0])
    slider_dy = Slider(ax_dy, 'dy', -0.5, 0.5, valinit=dy[1])

    def update(val):
        nonlocal dx, dy
        filtering_level = slider_filter.val
        rotation_angle = slider_rotation.val
        dx = np.array([slider_dx.val, dx[1]])
        dy = np.array([dy[0], slider_dy.val])

        samples, weights = anisotropic_filtering_debug(uv, dx, dy, filtering_level, rng_seed, rotation_angle)
        scatter.set_offsets(samples)
        scatter.set_array(weights)
        fig.canvas.draw_idle()

    slider_filter.on_changed(update)
    slider_rotation.on_changed(update)
    slider_dx.on_changed(update)
    slider_dy.on_changed(update)

    plt.show()

if __name__ == "__main__":
    main()
