import subprocess

def subset_font(input_path, output_path, text_file):
    """
    使用 pyftsubset 工具对字体进行子集化和压缩。

    参数:
    input_path (str): 输入字体文件路径
    output_path (str): 输出字体文件路径
    text_file (str): 包含所需字符的文本文件路径
    """
    command = [
        'pyftsubset',
        input_path,
        f'--output-file={output_path}',
        f'--text-file={text_file}',
        '--flavor=woff2',
        '--layout-features=*' # 保留字体特性
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"成功创建子集化字体: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"子集化失败: {e.stderr}")
    except FileNotFoundError:
        print("错误: 'pyftsubset' 命令未找到。请确保 'fonttools' 已安装并已添加到系统路径中。")

# 使用示例
subset_font('./SourceHanSansCN-Medium.otf', './SourceHanSansCN-Medium-subset.ttf', './chars.txt')