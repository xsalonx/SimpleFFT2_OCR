from numpy.fft import *
from PIL import Image
import matplotlib.pyplot as plt
import scipy.sparse as ssparse
import scipy.sparse.linalg as ssparselin

from reduce import *
from OCR.chars import *
from parsing import *

from deskew import determine_skew
from skimage import io
from skimage.color import rgb2gray, rgba2rgb
from skimage.transform import rotate



def load_binary(path, b=40, k_ratio=0.25, low_rank=True, d=0.42, alpha=200, make_deskew=False):
    # handing rotated images
    if make_deskew:
        image = io.imread(path)
        grayscale = rgb2gray(rgba2rgb(image))
        angle = determine_skew(grayscale)
        print(f"angle = {angle}")
        rotated = rotate(image, angle, resize=True, cval=1) * 255
        deskewed_name = path[:-4] + "deskew.png"
        io.imsave(deskewed_name, rotated.astype(np.uint8))
        arr = np.array(Image.open(deskewed_name).convert('L'), dtype=float)
    else:
        arr = np.array(Image.open(path).convert('L'), dtype=float)

    arr = 255 - arr
    arr[arr <= b] = 0

    if low_rank:
        A = ssparse.csc_matrix(arr, dtype=float)
        U, s, Vt = ssparselin.svds(A, k=int(k_ratio * min(A.shape)))
        arr = U @ np.diag(s) @ Vt

    arr[arr <= b] = 0
    arr[arr > b] = 1
    return (arr - d) * alpha


def analyze_scan(Image_path, char_paths, Tau=None):
    if Tau is None:
        Tau = [0.95] * len(char_paths)

    Im_arr = load_binary(Image_path)
    print("main image loaded")
    chars_arrays = [load_binary(c_p, low_rank=True) for c_p in char_paths]

    Im_fft = fft2(Im_arr)
    print("Main image fft calculated")
    rot_chars_ffts = [fft2(np.rot90(el_arr, 2), s=Im_arr.shape) for el_arr in chars_arrays]
    print("Characters image ffts calculated")

    C = [ifft2(Im_fft * ref) for ref in rot_chars_ffts]
    RC = [np.real(c) for c in C]
    print("ffts correlations calculated")

    global_max = [np.max(rc) for rc in RC]
    positions = [(rc >= tau * glob_m) for (rc, glob_m), tau in zip(zip(RC, global_max), Tau)]

    P = [np.array(np.nonzero(pos == 1)).T for pos in positions]
    P = [list(set([(y, x) for y, x in pos])) for pos in P]

    for k in range(len(P)):
        P[k] = reduce(positions[k], RC[k], P[k])


    [print(len(p), end=' ') for p in P]
    return P, Im_arr, [a.shape for a in chars_arrays]


def main(Im_path, chars_pre_path, save_name_txt="ocr_res.txt"):
    chars = [c for c, _ in chars_d]
    Tau = [w - 0.00 for _, w in chars_d]

    print(chars)
    chars_paths = [f"{chars_pre_path}{MAP_CHAR_TO_FILE_NAME[c]}.png" for c in chars]

    P, Im_arr, chars_shapes = analyze_scan(Im_path, chars_paths, Tau=Tau)
    X = [[x for _, x in p] for p in P]
    Y = [[y for y, _ in p] for p in P]

    plt.imshow(np.array(Im_arr, dtype=float), cmap="binary")
    for (x, y) in zip(X, Y):
        plt.scatter(x, y, s=30)

    Yl, Xl = get_cluster_to_lines_to_plot(P)

    for (x, y) in zip(Xl, Yl):
        plt.plot(x, y)

    print("\ntext read form image:\n\n")
    chars_widths = [s[1] for s in chars_shapes]
    text = parse(P, chars_d, Dx=chars_widths)
    with open(save_name_txt, "w") as f:
        f.write(text)
    print(text)

    plt.show()

if __name__ == '__main__':

    Im_path = "../../data/zad2_silmarillion/para1_reduced.png"
    # Im_path = "data/zad2_silmarillion/para1.png"
    # Im_path = "data/zad2_silmarillion/para1_rot_15.png"
    # Im_path = "data/zad2_silmarillion/para1_noisy.png"
    # Im_path = "data/zad2_silmarillion/page1_14_time_new_roma.png"

    chars_pre_path = "../../data/zad2_silmarillion/chars/"

    main(Im_path, chars_pre_path)
