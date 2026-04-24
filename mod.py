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
    if asp: 
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
    xi = np.linspace(-400, 400, gridsize)
    yi = np.linspace(-400, 400, gridsize)
    
    # xi = np.linspace(min(x), max(x), gridsize)
    # yi = np.linspace(min(y), max(y), gridsize)
    Xg, Yg = np.meshgrid(xi, yi)

    Zg = griddata(
        points=np.column_stack((x, y)),
        values=z,
        xi=(Xg, Yg),
        method=method
    )

    return Xg, Yg, Zg


def write_txt_from_arrays(filename, x, y, z, date):
    data = np.column_stack((x, y, z))
    np.savetxt(f"Data/HighPressureTesting-{date}/"+filename, data, fmt="%.6f", delimiter="\t")


def linear_transformation(refs, arr, indr, inda):
    x_ref, y_ref = refs[indr,0], refs[indr,1]
    x_arr, y_arr = arr[inda,0], arr[inda,1]

    dx = x_ref - x_arr
    dy = y_ref - y_arr

    return arr[:,0] + dx, arr[:,1] + dy

def align_points_2d(refs, arr):
    c_ref = refs.mean(axis=0)
    c_arr = arr.mean(axis=0)

    X = refs - c_ref
    Y = arr - c_arr

    H = Y.T @ X
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T

    rotated = (R @ arr.T).T 

    return rotated, R

import numpy as np

def circleRadius(b, c, d):
  '''Ref :https://stackoverflow.com/questions/52990094/calculate-circle-given-3-points-code-explanation '''
  temp = c[0]**2 + c[1]**2
  bc = (b[0]**2 + b[1]**2 - temp) / 2
  cd = (temp - d[0]**2 - d[1]**2) / 2
  det = (b[0] - c[0]) * (c[1] - d[1]) - (c[0] - d[0]) * (b[1] - c[1])

  if abs(det) < 1.0e-10:
    return None

  # Center of circle
  cx = (bc*(c[1] - d[1]) - cd*(b[1] - c[1])) / det
  cy = ((b[0] - c[0]) * cd - (c[0] - d[0]) * bc) / det

  radius = ((cx - b[0])**2 + (cy - b[1])**2)**.5

  return cx, cy, radius