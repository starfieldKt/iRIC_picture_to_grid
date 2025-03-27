import make_pixel
import sys
import cv2
import matplotlib.pyplot as plt
from iric import *

def make_grid_cord(base_point_x, base_point_y, pix_to_meter, height, width):
    """
    グリッドの座標を作成する。\n
    Args:
        base_point_x : 基準点のx座標
        base_point_y : 基準点のy座標
        pix_to_meter : メートル当たりのピクセル数
        height : 画像の高さ
        width : 画像の幅
    Returns:
        grid_cord : グリッドの座標
    """
    # 座標配列x,yはnumpy配列
    # 格子のインデックスi,jはiが画像の幅方向、jが画像の高さ方向
    # ピクセル＝セルの中心と考えるので、格子点の数は(width+1)*(height+1)となる
    # それぞれ(width+1, height+1)のnumpy配列を作成
    grid_node_x_array = np.arange(0, width+1, 1)
    grid_node_y_array = np.arange(0, height+1, 1)
    grid_node_y, grid_node_x = np.meshgrid(grid_node_y_array, grid_node_x_array)

    # ピクセルのサイズをかけてメートルに変換、基準点にシフトさせる
    grid_node_x = grid_node_x * pix_to_meter + base_point_x
    grid_node_y = grid_node_y * pix_to_meter + base_point_y

    return grid_node_x, grid_node_y

# メイン
if __name__ == "__main__":
    ier = 0

    # CGNSファイルの名前取得
    if len(sys.argv) < 2:
        print("Error: CGNS file name not specified.")
        exit()

    write_cgns_name = sys.argv[1]

    # write_cgns_name = ""

    # CGNSファイルを開く
    fid = cg_iRIC_Open(write_cgns_name, IRIC_MODE_MODIFY)

    # CGNSファイルから条件の読み込み
    #==========================================================================
    image_original_name = cg_iRIC_Read_String(fid, "read_image_name")
    image_scale_down_rate = cg_iRIC_Read_Integer(fid, "image_scale_down_rate")
    elevation_for_color_0 = cg_iRIC_Read_Real(fid, "elevation_for_color_0")
    elevation_for_color_255 = cg_iRIC_Read_Real(fid, "elevation_for_color_255")
    base_point_x = cg_iRIC_Read_Real(fid, "base_point_x")
    base_point_y = cg_iRIC_Read_Real(fid, "base_point_y")
    pix_to_meter = cg_iRIC_Read_Real(fid, "pix_to_meter")


    # 画像の読み込み及びnumpy配列化
    #==========================================================================
    # 画像を8bitグレースケールで読み込む
    image_8bit_gray = cv2.imread(image_original_name, cv2.IMREAD_GRAYSCALE)
    
    # モザイク化する場合はモザイク処理を行う
    if image_scale_down_rate > 1:
        image_8bit_gray = make_pixel.mosaic(image_8bit_gray, image_scale_down_rate)

    # 画像の高さと幅を取得
    height, width = image_8bit_gray.shape

    # グリッドの座標を作成
    # ピクセルの中心をセルの中心と考える
    # そのため格子点の数は(width+1)*(height+1)となる
    # i,jはセルのインデックス
    # i=1, j=1が左下のセルの中心
    # iが画像の幅方向、jが画像の高さ方向
    #   j
    #   ↑
    # 5 |
    # 4 |
    # 3 |
    # 2 |
    # 1 | 
    #    --------------→ i
    #     1 2 3 4 5 6 7
    #==========================================================================
    grid_node_x, grid_node_y = make_grid_cord(base_point_x, base_point_y, pix_to_meter, height, width)


    # 各ピクセルの8bitグレースケール値を標高値に変換
    # 0 -> elevation_for_color_0, 255 -> elevation_for_color_255
    #==========================================================================
    elevation_array_cell_origin = image_8bit_gray * (elevation_for_color_255 - elevation_for_color_0) / 255 + elevation_for_color_0

    # 配列のインデックスを格子セルのインデックスの並び方に変換
    # 画像から読み込んだ値は左上が原点であるため、左下が原点の格子セルの並びに変換する
    # そのため、配列を上下反転し、(high, width)の配列を(width, high)の配列に変換する
    # これにより、格子のセルのインデックスと配列のインデックスが一致する
    #==========================================================================
    elevation_array_cell_transpose = np.fliplr(elevation_array_cell_origin.T)

    # セル中心の値を格子点の値に変換
    # 格子点の値は、周囲のセルの値の平均値とする
    # 端の場合は周囲2つのセルの平均値とする
    # 角の場合は周囲1つのセルの値とする
    #==========================================================================
    elevation_array_node = np.zeros((width+1, height+1))
    elevation_array_node[1:-1, 1:-1] = (elevation_array_cell_transpose[:-1, :-1] + elevation_array_cell_transpose[1:, :-1] + elevation_array_cell_transpose[:-1, 1:] + elevation_array_cell_transpose[1:, 1:]) / 4
    elevation_array_node[0, 1:-1] = (elevation_array_cell_transpose[0, :-1] + elevation_array_cell_transpose[0, 1:]) / 2
    elevation_array_node[-1, 1:-1] = (elevation_array_cell_transpose[-1, :-1] + elevation_array_cell_transpose[-1, 1:]) / 2
    elevation_array_node[1:-1, 0] = (elevation_array_cell_transpose[:-1, 0] + elevation_array_cell_transpose[1:, 0]) / 2
    elevation_array_node[1:-1, -1] = (elevation_array_cell_transpose[:-1, -1] + elevation_array_cell_transpose[1:, -1]) / 2
    elevation_array_node[0, 0] = elevation_array_cell_transpose[0, 0]
    elevation_array_node[-1, 0] = elevation_array_cell_transpose[-1, 0]
    elevation_array_node[0, -1] = elevation_array_cell_transpose[0, -1]
    elevation_array_node[-1, -1] = elevation_array_cell_transpose[-1, -1]

    # 格子の座標と標高をCGNSファイルに書き込む
    # 引数はFortranのように列優先の1次元配列であるため、配列を転置する
    #==========================================================================
    cg_iRIC_Write_Grid2d_Coords(fid, width+1, height+1, grid_node_x.flatten(order="F"), grid_node_y.flatten(order="F"))
    cg_iRIC_Write_Grid_Real_Cell(fid, "Elevation", elevation_array_cell_transpose.flatten(order="F"))
    cg_iRIC_Write_Grid_Real_Node(fid, "Elevation", elevation_array_node.flatten(order="F"))

    plt.imshow(image_8bit_gray, cmap="gray")
    plt.show()

    cg_iRIC_Close(fid)

# デバッグ用

if __name__ == "__fugamain__":
    ier = 0

    # CGNSファイルから条件の読み込み
    #==========================================================================
    image_original_name = "w_to_B.png"
    image_scale_down_rate = 1
    elevation_for_color_0 = 0
    elevation_for_color_255 = 255
    base_point_x = 0.0
    base_point_y = 0.0
    pix_to_meter = 1.0


    # 画像の読み込み及びnumpy配列化
    #==========================================================================
    # 画像を8bitグレースケールで読み込む
    image_8bit_gray = cv2.imread(image_original_name, cv2.IMREAD_GRAYSCALE)
    
    # モザイク化する場合はモザイク処理を行う
    if image_scale_down_rate > 1:
        image_8bit_gray = make_pixel.mosaic(image_8bit_gray, image_scale_down_rate)

    # 画像の高さと幅を取得
    height, width = image_8bit_gray.shape

    # グリッドの座標を作成
    # ピクセルの中心をセルの中心と考える
    # そのため格子点の数は(width+1)*(height+1)となる
    # i,jはセルのインデックス
    # i=1, j=1が左下のセルの中心
    # iが画像の幅方向、jが画像の高さ方向
    #   j
    #   ↑
    # 5 |
    # 4 |
    # 3 |
    # 2 |
    # 1 | 
    #    --------------→ i
    #     1 2 3 4 5 6 7
    #==========================================================================
    grid_node_x, grid_node_y = make_grid_cord(base_point_x, base_point_y, pix_to_meter, height, width)


    # 各ピクセルの8bitグレースケール値を標高値に変換
    # 0 -> elevation_for_color_0, 255 -> elevation_for_color_255
    #==========================================================================
    elevation_array_cell_origin = image_8bit_gray * (elevation_for_color_255 - elevation_for_color_0) / 255 + elevation_for_color_0

    # 配列のインデックスを格子セルのインデックスの並び方に変換
    # 画像から読み込んだ値は左上が原点であるため、左下が原点の格子セルの並びに変換する
    # そのため、配列を上下反転し、(high, width)の配列を(width, high)の配列に変換する
    # これにより、格子のセルのインデックスと配列のインデックスが一致する
    #==========================================================================
    elevation_array_cell_transpose = np.flipud(elevation_array_cell_origin.T) 

    # セル中心の値を格子点の値に変換
    # 格子点の値は、周囲のセルの値の平均値とする
    # 端の場合は周囲2つのセルの平均値とする
    # 角の場合は周囲1つのセルの値とする
    #==========================================================================
    elevation_array_node = np.zeros((width+1, height+1))
    elevation_array_node[1:-1, 1:-1] = (elevation_array_cell_transpose[:-1, :-1] + elevation_array_cell_transpose[1:, :-1] + elevation_array_cell_transpose[:-1, 1:] + elevation_array_cell_transpose[1:, 1:]) / 4
    elevation_array_node[0, 1:-1] = (elevation_array_cell_transpose[0, :-1] + elevation_array_cell_transpose[0, 1:]) / 2
    elevation_array_node[-1, 1:-1] = (elevation_array_cell_transpose[-1, :-1] + elevation_array_cell_transpose[-1, 1:]) / 2
    elevation_array_node[1:-1, 0] = (elevation_array_cell_transpose[:-1, 0] + elevation_array_cell_transpose[1:, 0]) / 2
    elevation_array_node[1:-1, -1] = (elevation_array_cell_transpose[:-1, -1] + elevation_array_cell_transpose[1:, -1]) / 2
    elevation_array_node[0, 0] = elevation_array_cell_transpose[0, 0]
    elevation_array_node[-1, 0] = elevation_array_cell_transpose[-1, 0]
    elevation_array_node[0, -1] = elevation_array_cell_transpose[0, -1]
    elevation_array_node[-1, -1] = elevation_array_cell_transpose[-1, -1]

    # 格子の座標と標高をCGNSファイルに書き込む
    # 引数はFortranのように列優先の1次元配列であるため、配列を転置する
    #==========================================================================
    # cg_iRIC_Write_Grid2d_Coords(fid, width+1, height+1, grid_node_x.flatten(order="F"), grid_node_y.flatten(order="F"))
    # cg_iRIC_Write_Grid_Real_Cell(fid, "Elevation", elevation_array_cell_transpose.flatten(order="F"))
    # cg_iRIC_Write_Grid_Real_Node(fid, "Elevation", elevation_array_node.flatten(order="F"))

    plt.imshow(image_8bit_gray, cmap="gray")
    plt.show()
