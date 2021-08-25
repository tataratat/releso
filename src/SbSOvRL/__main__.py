import sys

def main(*args) -> None:
    print("Hallo", args)

def entrypoint() -> None:
    main(sys.argv[1:])

if __name__ == '__main__':  # pragma: no cover
    main(sys.argv[1:])