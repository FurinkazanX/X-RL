"""临时运行脚本"""

import sys
import os

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.abspath('.'))

# 导入并运行 main 函数
from xrl.main import main

if __name__ == "__main__":
    main()
