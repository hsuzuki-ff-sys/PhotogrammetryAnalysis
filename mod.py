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
    Singular Value Decompision on reference points xr, yr, zr and rotate coordinate of all points.
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
    '''
    This function take 3D coordinate information and generate surface fit using method provided. Interpolation is done in scipy.griddata package.
    Input: 
        x, y, z (array of float)
            3D coordinate of points 
        method, default 'cubic' (string)
            method of interpolation for scipy.griddata
        gridsize, default 200 (int)
            gridsize of interpolation surface

    Output: 
        Xg, Yg, Zg (array of float)
            Fitted surface
    '''

    # set up grid
    xi = np.linspace(-400, 400, gridsize)
    yi = np.linspace(-400, 400, gridsize)
    
    Xg, Yg = np.meshgrid(xi, yi)

    # perform interpolation
    Zg = griddata(
        points=np.column_stack((x, y)),
        values=z,
        xi=(Xg, Yg),
        method=method
    )

    return Xg, Yg, Zg


from numpy.polynomial.polynomial import polyvander2d 
def even_polyfit(x, y, z, order=4, gridsize=200): 
    '''
    This function take 3D coordinate information and generate surface fit to even polynomials with least square method.
    Input: 
        x, y, z (array of float)
            3D coordinate of points 
        order, default 4 (even int)
            order of polynomial
        gridsize, default 200 (int)
            gridsize of interpolation surface

    Output: 
        Xg, Yg, Zg (array of float)
            Fitted surface
    '''

    # make sure order given is even 
    if order% 2 !=0 : 
        raise ValueError("Polynomial Order have to be even.")
    
    # get Vandermonde matrix 
    # https://numpy.org/doc/2.2/reference/generated/numpy.polynomial.polynomial.polyvander2d.html 
    vander = polyvander2d(x, y, [order, order])

    # get even terms only
    px, py = np.meshgrid(np.arange(order + 1), np.arange(order + 1), indexing='ij')
    even_mask = (px % 2 == 0) & (py % 2 == 0)
    mask = even_mask.ravel()
    powers = list(zip(px[even_mask].ravel(), py[even_mask].ravel()))
    V = vander[:, mask]

    # fit least squares
    coeffs, _, _, _  = np.linalg.lstsq(V, z, rcond=None)

    # generate meshgrids
    xi = np.linspace(-400, 400, gridsize)
    yi = np.linspace(-400, 400, gridsize)
    Xg, Yg = np.meshgrid(xi, yi)

    # generate surface with fitted coefficients
    Zg = np.zeros_like(Xg)
    for c, (px, py) in zip(coeffs, powers):
        Zg += c * (Xg**px) * (Yg**py)

    return Xg, Yg, Zg


def write_txt_from_arrays(filename, x, y, z, date):
    '''
    This function generate a txt file contains 3D coordinate information of targets. 

    Input:  
        filename (string)
            name of file to be saved 
        x, y, z (array)
            arrays of coordinates
        date (string)
            date of the measurements, used for filename. 
    '''
    data = np.column_stack((x, y, z))
    np.savetxt(f"Data/HighPressureTesting-{date}/"+filename, data, fmt="%.6f", delimiter="\t")


def linear_transformation(refs, arr, indr, inda):
    '''
    This function linearly translate array points on xy plane such that selected points overlap.

    Input: 
        refs (2D array)
            all 3D coordinates of one dataset
        arr (2D array)
            all 3D coordinates of another dataset, this will be linearly tanslated to match points on refs points. 
        indr (array)
            array of indeces from refs to be referenced
        inda (array)
            array of indeces from arr to be used for translation        
    
    Output: 
        x, y (array) 
            new x, y coordinates of arr
    '''
    
    
    x_ref, y_ref = refs[indr,0], refs[indr,1]
    x_arr, y_arr = arr[inda,0], arr[inda,1]

    dx = x_ref - x_arr
    dy = y_ref - y_arr

    return arr[:,0] + dx, arr[:,1] + dy

def align_points_2d(refs, arr):
    '''
    This function rotate array points on xy plane such that selected points overlap.

    Input: 
        refs (2D array)
            3D coordinates of reference points from one dataset
        arr (2D array)
            3D coordinates of reference points from another dataset
    
    Output:
        rotated (2D arrayd) 
            new x, y coordinates of arr
        R (matrix)
            2D rotation matrix used. 
    '''
    # Find center of reference points
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
  '''From three points on a circle, determine center and radius of a circle. Ref :https://stackoverflow.com/questions/52990094/calculate-circle-given-3-points-code-explanation 
  
  Input: 
    b, c, d (size 3 array)
        x, y, z coordinates of each points 

  Output: 
    cx, cy (float)
        coordinate of center of the circle
    raidus (float)
        radius of the circle  
  
  '''
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

def min_gradient(x, y):
    '''This function find a x axis position where gradient of the curve is 0. 

    Input: 
        x, y (narray) 

    Output:
        x0 (float)
            x value where gradient is 0. 
    '''
    
    dydx = np.gradient(y, x)
    i0 = np.nanargmin(np.abs(dydx))
    x0 = x[i0]
    return x0

def extract_cuts(Xg, Yg, Zg):
    '''This function returns pair of xz and yz cuts of the surface. 

    Input: 
        Xg, Yg, Zg (array)
            surface 
        
    Output: 
        (x_xz, z_xz) (tuple)
            array of x,z values on xz cut
        (y_yz, z_yz) (tuple)
            array of y,z values on yz cut
    '''
    ny, nx = Zg.shape
    ix = nx // 2   # middle column
    iy = ny // 2   # middle row

    x_xz = Xg[iy, :]
    z_xz = Zg[iy, :]

    y_yz = Yg[:, ix]
    z_yz = Zg[:, ix]

    return (x_xz, z_xz), (y_yz, z_yz)

def spherical_fit(x,y,z): 
    '''
    This function fit a sphere to given 3D points using least square method.

    Input: 
        x, y, z (array)
            3D coordinates of target points

    Output: 
        R (float)
            Radius of fitted sphere
    '''

    A = np.column_stack([x, y, z, np.ones_like(x)])
    b = x**2 + y**2 + z**2 

    c1, c2, c3, a = np.linalg.lstsq(A, b, rcond=None)[0] / 2
    
    R = np.sqrt(c1**2 + c2**2 + c3**2 + a)
    return R