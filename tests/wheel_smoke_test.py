import platform
import sys
import traceback


def main() -> None:
    print("=" * 72)
    print("ArborEnum wheel smoke test")
    print("=" * 72)
    print("Python executable:", sys.executable)
    print("Python version:", sys.version)
    print("Platform:", platform.platform())
    print("Machine:", platform.machine())

    try:
        print("\nImporting arborenum...")
        import arborenum

        print("arborenum location:", arborenum.__file__)
        print("arborenum exports:", sorted(dir(arborenum)))

        print("\nImporting public classes...")
        from arborenum import ArborEnum, ThresholdGuessBinarizer

        print("ArborEnum:", ArborEnum)
        print("ThresholdGuessBinarizer:", ThresholdGuessBinarizer)

        print("\nInstantiating public classes...")
        model = ArborEnum()
        binarizer = ThresholdGuessBinarizer()

        print("ArborEnum instance:", type(model))
        print("ThresholdGuessBinarizer instance:", type(binarizer))
        print("\nArborEnum wheel smoke test passed.")

    except BaseException:
        print("\nArborEnum wheel smoke test failed.", file=sys.stderr)
        print("Full exception:", file=sys.stderr)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()