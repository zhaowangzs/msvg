import PIL
from PIL import Image, ImageDraw

def draw_detection_box(input_path, output_path, coordinates):
    # 打开图片
    image = Image.open(input_path)

    # 创建绘图对象
    draw = ImageDraw.Draw(image)

    # 绘制检测框
    draw.rectangle(coordinates, outline="orange", width=5)

    # 保存图片
    image.save(output_path)

if __name__ == "__main__":
    # 指定输入图片路径
    input_path = "/media/xd/disk3/zxh/Mymodel1/images/00010.jpg"

    # 指定输出图片路径
    output_path = "/media/xd/disk3/zxh/Mymodel1/images/out/00010.jpg"

    # 指定坐标（左上角和右下角的坐标）
    coordinates = [(84, 286), (635, 526)]

    # 绘制检测框并保存
    draw_detection_box(input_path, output_path, coordinates)
