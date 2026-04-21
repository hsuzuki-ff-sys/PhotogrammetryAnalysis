import numpy as np 
import matplotlib.pyplot as plt 
from scipy.interpolate import griddata


def vis(x,y,z, date, ind=False, asp=False, Verdose=False): 
    '''This fucntion is to plot 3D vector in 3D plot and XY, XZ, YZ slices.
    Input:
        x,y,z (array) 
        ind (Boolean) : True, show index of points in XY slice
        asp (Boolean) : True, set aspect ration to be 'equal' in  
    Output: '''
    fig = plt.figure(figsize=(12, 10))

 
    # 3D plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(x, y, z, c='blue')
    # ax1.set_title('3D Plot')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')

    # XY plane
    ax2 = fig.add_subplot(222)
    ax2.scatter(x, y, c='red')
    if ind:
        for i, (xi, yi) in enumerate(zip(x, y)):
            ax2.text(xi, yi, str(i), fontsize=9, color='black')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_aspect("equal")
    ax2.grid()

    # XZ plane
    ax3 = fig.add_subplot(223)
    ax3.scatter(x, z)
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Z (mm)')
    ax3.grid()

    # YZ plane
    ax4 = fig.add_subplot(224)
    ax4.scatter(y, z, c='purple')
    ax4.set_xlabel('Y (mm)')
    ax4.set_ylabel('Z (mm)')
    ax4.grid()

    if asp: 
        ax3.set_ylim(min(z)*0.5, max(z)*1.3)
        ax4.set_ylim(min(z)*0.5, max(z)*1.3)

    if Verdose: 
        ax1.set_title(f"Maximum difference: {max(z)-min(z):.4f} (mm)")
        fig.savefig(f"figs/vis-{date}.png")


def svd_rotate(ref, pts):
    """
    Singular Value Decompision on reference points xr, yr, zr and rotate coordinate of all points
    """

    # references 
    mean_ref = ref.mean(axis=0)
    ref_centered = ref - mean_ref

    U, S, Vt = np.linalg.svd(ref_centered, full_matrices=False)

    #rottation matrix bases on 
    R = Vt.T   

    rotated_ref = ref_centered @ R

    # other points
    pts_centered = pts - mean_ref

    rotated_pts = pts_centered @ R

    Xr, Yr, Zr = rotated_ref.T
    X, Y, Z = rotated_pts.T

    return Xr, Yr, Zr, X, Y, Z, R, mean_ref


def surfacefit(x, y, z, method='cubic', gridsize=200):
    
    xi = np.linspace(min(x), max(x), gridsize)
    yi = np.linspace(min(y), max(y), gridsize)
    Xg, Yg = np.meshgrid(xi, yi)

    Zg = griddata(
        points=np.column_stack((x, y)),
        values=z,
        xi=(Xg, Yg),
        method=method
    )

    return Xg, Yg, Zg
