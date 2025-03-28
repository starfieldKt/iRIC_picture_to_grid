import cv2
import matplotlib.pyplot as plt
import numpy as np


# imread : 画像ファイルを読み込んで、多次元配列(numpy.ndarray)にする。
# 第一引数 : 画像のファイルパス

# 第二引数(任意) : 画像の形式を指定。
# cv2.IMREAD_COLOR or 1を指定する場合 : (画像をカラー(RGB)で読み込む。 第二引数を指定しない場合にも選ばれる。デフォルト。)
# cv2.IMREAD_UNCHANGED or -1を指定する場合 : (画像をカラー(RGB)に透過度を加えた形式(RGBA)で読み込む。)
# cv2.IMREAD_GRAYSCALE or 0を指定する場合 : (画像をグレースケールで読み込む。)

# 戻り値
# cv2.IMREAD_COLOR or 1を指定した場合 : 行(高さ) x 列(幅) x 色の三次元配列(numpy.ndarray)が返される。
# cv2.IMREAD_UNCHANGED or -1を指定した場合 : 行(高さ) x 列(幅) x 色の三次元配列(numpy.ndarray)が返される。
# cv2.IMREAD_GRAYSCALE or 0を指定した場合 : 行(高さ) x 列(幅)の二次元配列(numpy.ndarray)が返される。
# ファイルが存在しない場合や例外が発生した場合等 : None




# 減色処理
# 画像のRGB値を指定したビット数で減色する
# フルカラー24bitはR:8 G:8 B:8　8bitカラーはR:3 G:3 B:2
# フルカラーを8bitカラーに変換する場合は、i=5, j=5, k=6とする
def color_reduction(image, i, j, k):
    """
    減色を行う。\n
    Args:
        image : 画像のnumpy配列
        i : Rの減色bit数
        j : Gの減色bit数
        k : Bの減色bit数
    Returns:
        image_reduction : 減色した画像のnumpy配列
    """
    image_reduction = np.uint8(image / [2**i, 2**j, 2**k]) * [
        2**i,
        2**j,
        2**k,
    ]

    return image_reduction

def down_scale(image, alpha):
    """
    画像を縮小する。\n
    Args:
        image : 画像のnumpy配列
        alpha : 縮小率
    Returns:
        image_reduction : 縮小した画像のnumpy配列
    """
    h, w = image.shape
    return cv2.resize(image, (int(w / alpha), int(h / alpha)))


# モザイク加工
# 画像を縮小して、拡大することでモザイク加工を行う
def mosaic(image, alpha):
    """
    モザイク処理を行う。\n
    Args:
        image : 画像のnumpy配列
        alpha : 縮小率
    Returns:
        image_mosaic : モザイク処理した画像のnumpy配列
    """

    h, w = image.shape

    # 一度縮小する
    res = cv2.resize(image, (int(w / alpha), int(h / alpha)))
    res = cv2.resize(res, (w, h), interpolation=cv2.INTER_NEAREST)
    return res

# ドット絵加工
# 256色のパレットを用いて、画像をドット絵に変換する
def pixel_proc(image, alpha=1, i=5, j=5, k=6):
    """
    モザイク処理を行う。\n
    Args:
        image : 画像のnumpy配列
        alpha : 縮小率
        i : Rの減色bit数
        j : Gの減色bit数
        k : Bの減色bit数
    Returns:
        image_mosaic : モザイク処理した画像のnumpy配列
    """
    return color_reduction(mosaic(image, alpha), i, j, k)

# テスト用
if __name__ == "__main__":
    iric = "./g3984.png"
    image = "python_pillow_quantize_01.png"

    # 画像の読み込み
    image_original_BGR = cv2.imread(image, cv2.IMREAD_COLOR)
    # BGR⇒RGBへ変換
    image_original_RGB = cv2.cvtColor(image_original_BGR, cv2.COLOR_BGR2RGB)

    # 減色処理
    image_8bit = pixel_proc(image_original_RGB, 20, 5, 5, 6)

    plt.imshow(image_8bit)
    plt.show()
