
from numpy.fft import *
from PIL import Image
import matplotlib.pyplot as plt
import scipy.sparse as ssparse
import scipy.sparse.linalg as ssparselin

from src.OCR.reduce import *
from src.OCR.chars import *
from src.OCR.parsing import *

from deskew import determine_skew
from skimage import io
from skimage.color import rgb2gray, rgba2rgb
from skimage.transform import rotate

from multiprocessing import Process, shared_memory


def load_binary(path, b=45, k_ratio=0.25, low_rank=True, d=0.45, alpha=300, make_deskew=False):
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

    arr = (arr - d) * alpha
    arr = np.array(arr, dtype=complex)
    sh = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    arr_sh = np.ndarray(arr.shape, dtype=arr.dtype, buffer=sh.buf)
    arr_sh[:] = arr[:]

    return sh, arr.shape, arr.nbytes


def calc_ffts_and_rot(sh_out, sh_in, shape):
    out_arr = [np.ndarray(shape=shape, dtype=complex, buffer=sh.buf) for sh in sh_out]
    in_arr = [np.ndarray(shape=sh[1], dtype=complex, buffer=sh[0].buf) for sh in sh_in]

    for i in range(len(out_arr)):
        out_arr[i][:] = fft2(np.rot90(in_arr[i], 2), s=shape)[:]


def calc_corr_ifft(sh_out, sh_Im, sh_in, SHAPE):
    # print(sh_in)
    out_arr = [np.ndarray(shape=SHAPE, dtype=complex, buffer=sh.buf) for sh in sh_out]
    Im_arr = np.ndarray(shape=SHAPE, dtype=complex, buffer=sh_Im.buf)
    in_arr = [np.ndarray(shape=SHAPE, dtype=complex, buffer=sh.buf) for sh in sh_in]

    for i in range(len(out_arr)):
        out_arr[i][:] = ifft2(Im_arr * in_arr[i])[:]
        # plt.imshow(np.real(out_arr[i]))
        # plt.show()


def analyze_scan(Image_path, char_paths, Tau=None):
    if Tau is None:
        Tau = [0.95] * len(char_paths)

    Im_data = load_binary(Image_path, make_deskew=True)
    SHAPE = Im_data[1]
    print("Main image loaded")
    chars_data = [load_binary(c_p, low_rank=True) for c_p in char_paths]
    print("Chars loaded")

    Im_arr = np.ndarray(shape=SHAPE, dtype=complex, buffer=Im_data[0].buf)
    sh_fft_im = shared_memory.SharedMemory(create=True, size=Im_arr.nbytes)
    Im_fft_sh_arr = np.ndarray(shape=SHAPE, dtype=complex, buffer=sh_fft_im.buf)
    Im_fft_sh_arr[:] = fft2(Im_arr)[:]
    print("Main image fft calculated")

    processes_numb = 7
    N = len(chars_data)
    h = N // processes_numb

    """
        Calculating patterns rot ffts
    """
    sh_rot_chars_ffts = [shared_memory.SharedMemory(create=True, size=Im_data[2]) for _ in range(N)]
    proc_shs_out = [sh_rot_chars_ffts[i * h: (i + 1) * h if i < processes_numb - 1 else N] for i in
                    range(processes_numb)]
    proc_shs_in = [chars_data[i * h: (i + 1) * h if i < processes_numb - 1 else N] for i in
                   range(processes_numb)]
    Processes = []
    for k in range(processes_numb):
        Processes.append(Process(target=calc_ffts_and_rot, args=(proc_shs_out[k], proc_shs_in[k], SHAPE,)))
    for k in range(processes_numb):
        Processes[k].start()
    for k in range(processes_numb):
        Processes[k].join()

    print("Characters image ffts calculated")

    """
        calculating fft correlations
    """

    sh_C = [shared_memory.SharedMemory(create=True, size=Im_data[2]) for _ in range(N)]
    C = [np.ndarray(shape=SHAPE, dtype=complex, buffer=sh.buf) for sh in sh_C]

    proc_shs_out = [sh_C[i * h: (i + 1) * h if i < processes_numb - 1 else N] for i in
                    range(processes_numb)]
    proc_shs_in = [sh_rot_chars_ffts[i * h: (i + 1) * h if i < processes_numb - 1 else N] for i in
                   range(processes_numb)]
    Processes = []
    for k in range(processes_numb):
        Processes.append(Process(target=calc_corr_ifft, args=(proc_shs_out[k], sh_fft_im, proc_shs_in[k], SHAPE, )))
    for k in range(processes_numb):
        Processes[k].start()
    for k in range(processes_numb):
        Processes[k].join()




    RC = [np.real(c) for c in C]
    print("ffts correlations calculated")

    global_max = [np.max(rc) for rc in RC]
    positions = [(rc >= tau * glob_m) for (rc, glob_m), tau in zip(zip(RC, global_max), Tau)]

    P = [np.array(np.nonzero(pos == 1)).T for pos in positions]
    P = [list(set([(y, x) for y, x in pos])) for pos in P]

    for k in range(len(P)):
        P[k] = reduce(positions[k], RC[k], P[k])

    [print(len(p), end=' ') for p in P]
    chars_shapes = [s[1] for s in chars_data]
    return P, np.real(Im_arr).copy(), chars_shapes


def main_ocr(Im_path, chars_pre_path, save_name_txt="out/ocr_res.txt"):
    chars = [c for c, _ in chars_d]
    Tau = [w - 0.00 for _, w in chars_d]

    print(chars)
    chars_paths = [f"{chars_pre_path}{MAP_CHAR_TO_FILE_NAME[c]}.png" for c in chars]

    P, Im_arr, chars_shapes = analyze_scan(Im_path, chars_paths, Tau=Tau)
    X = [[x for _, x in p] for p in P]
    Y = [[y for y, _ in p] for p in P]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(np.array(Im_arr, dtype=float), cmap="binary")

    for (x, y) in zip(X, Y):
        ax.scatter(x, y, s=30)

    Yl, Xl = get_cluster_to_lines_to_plot(P)

    for (x, y) in zip(Xl, Yl):
        ax.plot(x, y)

    print("\ntext read form image:\n\n")
    chars_widths = [s[1] for s in chars_shapes]
    text = parse(P, chars_d, Dx=chars_widths)
    with open(save_name_txt, "w") as f:
        f.write(text)
    print(text)
    return text



