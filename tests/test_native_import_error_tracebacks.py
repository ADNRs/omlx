def _assert_detached(exc):
    if exc is None:
        return
    assert exc.__traceback__ is None
    assert exc.__cause__ is None
    assert exc.__context__ is None


def test_optional_native_kernel_import_errors_do_not_retain_tracebacks():
    from omlx.custom_kernels.qwen35_prefill import fast as qwen35_fast

    _assert_detached(qwen35_fast.import_error())
