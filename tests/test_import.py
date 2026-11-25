def test_packages_import():
    import tellolab
    import tellosim

    assert tellosim.__version__
    assert tellolab.__version__
