import sys
import faulthandler

if "--cad-worker" in sys.argv:
    from cq_editor.ai.cad_runtime import worker_main

    worker_main()
    raise SystemExit(0)

# PyInstaller with console=False sets sys.stdout/stderr to None on Windows
if sys.stdout is None:
    sys.stdout = open("nul", "w")
if sys.stderr is None:
    sys.stderr = open("nul", "w")

faulthandler.enable()

from cq_editor.cqe_run import main

if __name__ == "__main__":
    main()
