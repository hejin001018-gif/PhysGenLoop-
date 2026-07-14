"""支持 ``python -m critic`` 的模块入口。"""

from .cli import main

# 将 CLI 返回码交还给操作系统，便于 shell/CI 正确识别执行结果。
raise SystemExit(main())
