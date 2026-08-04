import faulthandler
import platform
import sys
import traceback

faulthandler.enable()


def log(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def main() -> None:
    log("=" * 72)
    log("ArborEnum wheel smoke test")
    log("=" * 72)
    log("Python executable:", sys.executable)
    log("Python version:", sys.version)
    log("Platform:", platform.platform())
    log("Machine:", platform.machine())

    try:
        log("\nImporting numpy...")
        import numpy
        log("numpy:", numpy.__version__)

        log("\nImporting pandas...")
        import pandas
        log("pandas:", pandas.__version__)

        log("\nImporting matplotlib...")
        import matplotlib
        log("matplotlib:", matplotlib.__version__)

        log("\nImporting sklearn...")
        import sklearn
        log("scikit-learn:", sklearn.__version__)

        log("\nImporting compiled extension directly...")
        from arborenum import _core
        log("_core location:", _core.__file__)

        log("\nImporting arborenum...")
        import arborenum
        log("arborenum location:", arborenum.__file__)
        log("arborenum exports:", sorted(dir(arborenum)))

        log("\nImporting public classes...")
        from arborenum import ArborEnum, ThresholdGuessBinarizer

        log("ArborEnum:", ArborEnum)
        log("ThresholdGuessBinarizer:", ThresholdGuessBinarizer)

        log("\nInstantiating ArborEnum...")
        model = ArborEnum()
        log("ArborEnum instance:", type(model))

        log("\nInstantiating ThresholdGuessBinarizer...")
        binarizer = ThresholdGuessBinarizer()
        log("ThresholdGuessBinarizer instance:", type(binarizer))

        log("\nArborEnum wheel smoke test passed.")

    except BaseException as exc:
        log(
            "\nArborEnum wheel smoke test failed:",
            type(exc).__name__,
            repr(exc),
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise


if __name__ == "__main__":
    main()