from pathlib import Path


def test_user_counter_is_no_database_and_has_all_display_modes():
    source = Path(__file__).parents[1].joinpath("user_counter.py").read_text(encoding="utf-8")
    assert "st.session_state" in source
    assert "render_count" in source
    assert '"sidebar"' in source
    assert '"footer"' in source
    assert "sqlite" not in source.lower()
    assert "sqlalchemy" not in source.lower()
    assert "redis" not in source.lower()


def test_app_uses_counter_on_main_sidebar_and_footer():
    source = Path(__file__).parents[1].joinpath("app.py").read_text(encoding="utf-8")
    assert 'render_count("main")' in source
    assert 'render_count("sidebar")' in source
    assert 'render_count("footer")' in source
