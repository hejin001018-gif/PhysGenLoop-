"""安装布局和公开 API 的最小契约。"""


def test_public_package_can_be_imported():
    """正式包名不应依赖包含连字符的仓库目录名。"""

    import pavg_critic

    assert pavg_critic.__version__
    assert pavg_critic.PhysicsCritic is not None


def test_cli_program_name_matches_installed_entry_point():
    from pavg_critic.cli import build_parser

    assert build_parser().prog == "pavg-critic"
