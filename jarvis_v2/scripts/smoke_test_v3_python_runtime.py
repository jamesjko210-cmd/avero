from __future__ import annotations

import os
import shlex
import stat
import subprocess
import sys
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jarvis_v2.scripts.setup_status_auth import setup_status_auth
from jarvis_v2.scripts import v3_python_runtime
from jarvis_v2.scripts.test_v3_launcher_env import empty_v3_launcher_env
from jarvis_v2.scripts.v3_python_runtime import (
    V3PythonProbe,
    V3PythonUnavailable,
    V3InheritedEnvironmentConflict,
    assert_v3_environment_custody,
    inspect_v3_python,
    preferred_v3_python,
)
from jarvis_v2.ui.status_config import status_auth_token_is_valid


ROOT = Path(__file__).resolve().parents[2]
QUICKSTART = ROOT / "QUICKSTART.md"
LAUNCHERS = (
    "launch_jarvis_v3.py",
    "launch_jarvis_v3_calendar_auth.py",
    "launch_jarvis_v3_chat.py",
    "launch_jarvis_v3_voice.py",
    "launch_jarvis_v3_dashboard.py",
)
HELP_BEFORE_ENV_LAUNCHERS = frozenset(
    {"launch_jarvis_v3_calendar_auth.py", "launch_jarvis_v3_dashboard.py"}
)


def test_v3_launchers_reexec_before_entrypoint_import() -> None:
    launchers = {
        "launch_jarvis_v3.py": "from jarvis_v2.scripts.ask import main",
        "launch_jarvis_v3_calendar_auth.py": (
            "authority=_DIRECT_EXECUTION_AUTHORITY"
        ),
        "launch_jarvis_v3_chat.py": "from jarvis_v2.scripts.chat import main",
        "launch_jarvis_v3_dashboard.py": (
            "from jarvis_v2.scripts.run_status_server import manual_foreground_main"
        ),
        "launch_jarvis_v3_voice.py": "from jarvis_v2.scripts.talk import main",
    }
    for filename, entrypoint_import in launchers.items():
        source = (ROOT / filename).read_text(encoding="utf-8")
        reexec_index = source.find("reexec_with_v3_python(Path(__file__))")
        import_index = source.find(entrypoint_import)
        if reexec_index < 0 or import_index < 0 or reexec_index >= import_index:
            raise SystemExit(f"{filename} must select the V3 interpreter before importing its entrypoint")

    voice_launcher = ROOT / "launch_jarvis_v3_voice.py"
    voice_source = voice_launcher.read_text(encoding="utf-8")
    if 'sys.argv.append("--speak")' not in voice_source:
        raise SystemExit("V3 voice launcher must enable spoken replies by default")
    with TemporaryDirectory(prefix="jarvis-v3-voice-help-env-") as temp:
        help_env = empty_v3_launcher_env(Path(temp) / "runtime.env")
        help_run = subprocess.run(
            [sys.executable, str(voice_launcher), "--help"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=15,
            env=help_env,
        )
    if help_run.returncode != 0 or "--no-speak" not in help_run.stdout:
        raise SystemExit("V3 voice launcher help must expose the text-only override")

    quickstart = " ".join(QUICKSTART.read_text(encoding="utf-8").split())
    for expected in (
        "Python 3.11 or newer",
        "python3 -m venv .venv",
        "JARVIS_V3_PYTHON",
        "selected owner-only `runtime.env` is authoritative",
        "shell value is used only when no V3 environment file is selected",
        "they must match exactly",
        "may be a normal Homebrew or project-venv symlink",
        "resolved target must stay outside V2 custody",
        "same target policy used by automatic launcher discovery",
        "not protection from another same-UID process racing a pathname",
        ".venv/bin/python3 -m jarvis_v2.scripts.setup_status_auth",
        ".venv/bin/python3 -m jarvis_v2.scripts.copy_status_auth",
        "chmod 700",
        "./launch_jarvis_v3.py",
        "./launch_jarvis_v3_calendar_auth.py",
        "./launch_jarvis_v3_chat.py",
        "./launch_jarvis_v3_voice.py",
        "./launch_jarvis_v3_dashboard.py",
        "http://127.0.0.1:8766",
        "username `jarvis`",
        "pbcopy",
        "pbcopy </dev/null",
        "approval readiness ID",
        "approval packet ID",
        "approve approval ID",
        "must not be retried automatically",
        "Unrelated shell settings and exact matches are preserved",
    ):
        if expected not in quickstart:
            raise SystemExit(f"V3 quick start missed required daily-use guidance: {expected}")
    if "sed -n 's/^JARVIS_STATUS_AUTH_TOKEN=//p'" in quickstart:
        raise SystemExit("V3 quick start still exposes the legacy shell password extraction path")
    if '"$JARVIS_V3_PYTHON" -m' in quickstart or "export JARVIS_V3_PYTHON=" in quickstart:
        raise SystemExit("V3 quick start helper commands still depend on prior shell state")


def test_fresh_user_bootstrap_is_isolated_and_owner_only() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-fresh-launch-") as temp:
        root = Path(temp)
        home = root / "home"
        home.mkdir()
        env_dir = home / ".jarvis_v3"
        env_dir.mkdir(mode=0o700)
        env_file = env_dir / "runtime.env"

        result = setup_status_auth(env_file)
        if not result.token_generated or result.state != "generated":
            raise SystemExit(f"fresh V3 bootstrap did not generate dashboard auth: {result}")
        mode = stat.S_IMODE(env_file.stat().st_mode)
        if mode != 0o600:
            raise SystemExit(f"fresh V3 environment custody is {mode:o}, expected 600")
        directory_mode = stat.S_IMODE(env_dir.stat().st_mode)
        if directory_mode != 0o700:
            raise SystemExit(
                f"fresh V3 environment directory custody is {directory_mode:o}, expected 700"
            )
        lines = env_file.read_text(encoding="utf-8").splitlines()
        token_lines = [line for line in lines if line.startswith("JARVIS_STATUS_AUTH_TOKEN=")]
        if len(token_lines) != 1:
            raise SystemExit("fresh V3 environment did not contain exactly one dashboard auth entry")
        token = token_lines[0].partition("=")[2]
        if not status_auth_token_is_valid(token):
            raise SystemExit("fresh V3 environment stored an invalid dashboard auth token")

        child_env = {
            "HOME": str(home),
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(ROOT),
            "JARVIS_V3_ENV": str(env_file),
            "JARVIS_STATUS_AUTH_TOKEN": token,
            "JARVIS_CHAT_TIMEOUT_SECONDS": "17",
        }
        code = """
import os
from pathlib import Path
from jarvis_v2.config import load_config
from jarvis_v2.ui.status_config import status_auth_config_from_env, status_port_from_env

home = Path(os.environ['HOME'])
config = load_config()
auth = status_auth_config_from_env()
assert config.data_dir == home / '.jarvis_v3'
assert config.db_path == home / '.jarvis_v3' / 'jarvis.sqlite'
assert config.obsidian_vault == home / '.jarvis_v3' / 'Vault'
assert status_port_from_env() == 8766
assert auth.configured and auth.valid and auth.source == 'env'
assert not any('jarvis_v2' in str(value).casefold() for value in (config.data_dir, config.db_path, config.obsidian_vault))
print('fresh V3 launcher environment is isolated')
"""
        config_check = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(ROOT),
            env=child_env,
            capture_output=True,
            text=True,
            timeout=20,
        )
        if config_check.returncode != 0:
            raise SystemExit(
                "fresh V3 launcher environment check failed: "
                f"stdout={config_check.stdout!r}; stderr={config_check.stderr!r}"
            )

        for filename in LAUNCHERS:
            help_run = subprocess.run(
                [sys.executable, str(ROOT / filename), "--help"],
                cwd=str(ROOT),
                env=child_env,
                capture_output=True,
                text=True,
                timeout=20,
            )
            if help_run.returncode != 0 or not help_run.stdout.strip():
                raise SystemExit(
                    f"{filename} did not start its isolated V3 help flow: "
                    f"stdout={help_run.stdout!r}; stderr={help_run.stderr!r}"
                )


def test_inherited_custody_conflicts_fail_before_runtime_import() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-inherited-custody-") as temp:
        root = Path(temp)
        env_file = root / "runtime.env"
        file_token = "file-token-is-owner-only-and-long-enough-0001"
        env_file.write_text(
            "JARVIS_DATA_DIR=/private/tmp/jarvis-v3-selected-data\n"
            f"JARVIS_STATUS_AUTH_TOKEN={file_token}\n",
            encoding="utf-8",
        )
        env_file.chmod(0o600)

        probe_dir = root / "probe"
        probe_dir.mkdir()
        marker = root / "runtime-import-attempted"
        (probe_dir / "sitecustomize.py").write_text(
            "import importlib.abc\n"
            "import os\n"
            "from pathlib import Path\n"
            "class Probe(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, fullname, path=None, target=None):\n"
            "        if fullname in {'jarvis_v2.agent.runtime', 'jarvis_v2.memory.store'}:\n"
            "            Path(os.environ['JARVIS_IMPORT_PROBE']).touch()\n"
            "        return None\n"
            "import sys\n"
            "sys.meta_path.insert(0, Probe())\n",
            encoding="utf-8",
        )
        poisoned = {
            "HOME": str(root / "home"),
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.pathsep.join((str(probe_dir), str(ROOT))),
            "PYTHONDONTWRITEBYTECODE": "1",
            "JARVIS_V3_ENV": str(env_file),
            "JARVIS_DATA_DIR": "/private/tmp/synthetic-prior-generation-data",
            "JARVIS_STATUS_AUTH_TOKEN": "inherited-token-must-never-be-used-0002",
            "JARVIS_IMPORT_PROBE": str(marker),
            "JARVIS_CHAT_TIMEOUT_SECONDS": "17",
        }
        (root / "home").mkdir()
        for filename in LAUNCHERS:
            if marker.exists():
                marker.unlink()
            result = subprocess.run(
                [sys.executable, str(ROOT / filename), "--help"],
                cwd=str(ROOT),
                env=poisoned,
                capture_output=True,
                text=True,
                timeout=20,
            )
            if filename in HELP_BEFORE_ENV_LAUNCHERS:
                if result.returncode != 0 or not result.stdout.strip() or marker.exists():
                    raise SystemExit(f"{filename} help read selected custody state")
                continue
            if result.returncode != 78:
                raise SystemExit(
                    f"{filename} did not fail closed on inherited custody conflict: "
                    f"returncode={result.returncode}; stdout={result.stdout!r}; stderr={result.stderr!r}"
                )
            for expected in ("JARVIS_DATA_DIR", "JARVIS_STATUS_AUTH_TOKEN"):
                if expected not in result.stderr:
                    raise SystemExit(f"{filename} omitted conflicting key name {expected}")
            for forbidden in (
                file_token,
                poisoned["JARVIS_STATUS_AUTH_TOKEN"],
                poisoned["JARVIS_DATA_DIR"],
            ):
                if forbidden in result.stdout or forbidden in result.stderr:
                    raise SystemExit(f"{filename} leaked a conflicting custody value")
            if marker.exists():
                raise SystemExit(f"{filename} imported runtime/storage before custody refusal")


def test_unreadable_selected_environment_fails_bounded_before_runtime_import() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-env-refusal-") as temp:
        root = Path(temp)
        probe_dir = root / "probe"
        probe_dir.mkdir()
        marker = root / "runtime-import-attempted"
        (probe_dir / "sitecustomize.py").write_text(
            "import importlib.abc\n"
            "import os\n"
            "from pathlib import Path\n"
            "class Probe(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, fullname, path=None, target=None):\n"
            "        if fullname in {'jarvis_v2.agent.runtime', 'jarvis_v2.memory.store'}:\n"
            "            Path(os.environ['JARVIS_IMPORT_PROBE']).touch()\n"
            "        return None\n"
            "import sys\n"
            "sys.meta_path.insert(0, Probe())\n",
            encoding="utf-8",
        )

        missing = root / "missing-private-name.env"
        directory = root / "directory-private-name.env"
        directory.mkdir()
        symlink = root / "symlink-private-name.env"
        symlink.symlink_to(missing)
        malformed = root / "malformed-private-name.env"
        malformed.write_bytes(b"JARVIS_DATA_DIR=\xff\xfe\n")
        malformed.chmod(0o600)
        insecure_empty = root / "insecure-empty-private-name.env"
        insecure_empty.write_text("", encoding="utf-8")
        insecure_empty.chmod(0o644)
        secret_marker = "synthetic-selected-env-secret-must-not-leak"
        insecure_secret = root / "insecure-secret-private-name.env"
        insecure_secret.write_text(
            f"OPENAI_API_KEY={secret_marker}\n",
            encoding="utf-8",
        )
        insecure_secret.chmod(0o644)

        cases = {
            "missing": missing,
            "directory": directory,
            "symlink": symlink,
            "malformed": malformed,
            "insecure-empty": insecure_empty,
            "insecure-secret": insecure_secret,
        }
        for case_name, selected_env in cases.items():
            child_env = {
                "HOME": str(root / "home"),
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": os.pathsep.join((str(probe_dir), str(ROOT))),
                "PYTHONDONTWRITEBYTECODE": "1",
                "JARVIS_V3_ENV": str(selected_env),
                "JARVIS_IMPORT_PROBE": str(marker),
            }
            (root / "home").mkdir(exist_ok=True)
            for filename in LAUNCHERS:
                if marker.exists():
                    marker.unlink()
                result = subprocess.run(
                    [sys.executable, str(ROOT / filename), "--help"],
                    cwd=str(ROOT),
                    env=child_env,
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                if filename in HELP_BEFORE_ENV_LAUNCHERS:
                    if result.returncode != 0 or not result.stdout.strip() or marker.exists():
                        raise SystemExit(
                            f"{filename} help read {case_name} selected environment"
                        )
                    continue
                if result.returncode != 78:
                    raise SystemExit(
                        f"{filename} did not fail closed for {case_name} selected env: "
                        f"returncode={result.returncode}; stdout={result.stdout!r}; "
                        f"stderr={result.stderr!r}"
                    )
                rendered = result.stdout + result.stderr
                for expected in (
                    "could not read the selected environment safely",
                    "using QUICKSTART.md",
                ):
                    if expected not in rendered:
                        raise SystemExit(
                            f"{filename} omitted bounded {case_name} recovery: {rendered!r}"
                        )
                for forbidden in (
                    "Traceback",
                    str(root),
                    selected_env.name,
                    "missing-private-name",
                    "directory-private-name",
                    "symlink-private-name",
                    "malformed-private-name",
                    "insecure-empty-private-name",
                    "insecure-secret-private-name",
                    secret_marker,
                ):
                    if forbidden in rendered:
                        raise SystemExit(
                            f"{filename} leaked {case_name} environment detail {forbidden!r}"
                        )
                if len(rendered) > 500:
                    raise SystemExit(
                        f"{filename} returned unbounded {case_name} recovery: {len(rendered)}"
                    )
                if marker.exists():
                    raise SystemExit(
                        f"{filename} imported runtime/storage before {case_name} env refusal"
                    )


def test_symlink_loop_path_custody_fails_bounded_before_runtime_import() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-path-custody-refusal-") as temp:
        root = Path(temp)
        home = root / "home"
        home.mkdir()
        probe_dir = root / "probe"
        probe_dir.mkdir()
        marker = root / "runtime-import-attempted"
        (probe_dir / "sitecustomize.py").write_text(
            "import importlib.abc\n"
            "import os\n"
            "from pathlib import Path\n"
            "class Probe(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, fullname, path=None, target=None):\n"
            "        if fullname in {'jarvis_v2.agent.runtime', 'jarvis_v2.memory.store'}:\n"
            "            Path(os.environ['JARVIS_IMPORT_PROBE']).touch()\n"
            "        return None\n"
            "import sys\n"
            "sys.meta_path.insert(0, Probe())\n",
            encoding="utf-8",
        )
        loop_a = root / "loop-a-private-name"
        loop_b = root / "loop-b-private-name"
        loop_a.symlink_to(loop_b)
        loop_b.symlink_to(loop_a)
        child_env = {
            "HOME": str(home),
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.pathsep.join((str(probe_dir), str(ROOT))),
            "PYTHONDONTWRITEBYTECODE": "1",
            "JARVIS_DATA_DIR": str(loop_a / "state"),
            "JARVIS_IMPORT_PROBE": str(marker),
        }
        for filename in LAUNCHERS:
            if marker.exists():
                marker.unlink()
            result = subprocess.run(
                [sys.executable, str(ROOT / filename), "--help"],
                cwd=str(ROOT),
                env=child_env,
                capture_output=True,
                text=True,
                timeout=20,
            )
            rendered = result.stdout + result.stderr
            if filename in HELP_BEFORE_ENV_LAUNCHERS:
                if result.returncode != 0 or not result.stdout.strip() or marker.exists():
                    raise SystemExit(f"{filename} help evaluated symlink-loop custody")
                continue
            if result.returncode != 78:
                raise SystemExit(
                    f"{filename} did not fail closed for symlink-loop custody: "
                    f"returncode={result.returncode}; stdout={result.stdout!r}; "
                    f"stderr={result.stderr!r}"
                )
            for expected in (
                "could not validate setting JARVIS_DATA_DIR safely",
                "link or alias",
                "Review only JARVIS_DATA_DIR",
                "retry `setup check`",
                "Launcher path recovery",
            ):
                if expected not in rendered:
                    raise SystemExit(
                        f"{filename} omitted bounded path-custody recovery: {rendered!r}"
                    )
            for forbidden in (
                "Traceback",
                str(root),
                loop_a.name,
                loop_b.name,
                "Symlink loop",
                "JARVIS_DB_PATH",
                "JARVIS_GOOGLE_TOKEN",
            ):
                if forbidden in rendered:
                    raise SystemExit(
                        f"{filename} leaked path-custody detail {forbidden!r}: {rendered!r}"
                    )
            if len(rendered) > 500:
                raise SystemExit(
                    f"{filename} returned unbounded path-custody recovery: {len(rendered)}"
                )
            if marker.exists():
                raise SystemExit(
                    f"{filename} imported runtime/storage before path-custody refusal"
                )


def test_exact_custody_matches_and_non_state_env_are_preserved() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-matching-custody-") as temp:
        root = Path(temp)
        env_file = root / "runtime.env"
        selected_data = str(root / "selected-data")
        selected_token = "matching-token-is-owner-only-and-long-enough-0003"
        env_file.write_text(
            f"JARVIS_DATA_DIR={selected_data}\n"
            f"JARVIS_STATUS_AUTH_TOKEN={selected_token}\n",
            encoding="utf-8",
        )
        env_file.chmod(0o600)
        compatible_env = {
            "JARVIS_V3_ENV": str(env_file),
            "JARVIS_DATA_DIR": selected_data,
            "JARVIS_STATUS_AUTH_TOKEN": selected_token,
            "JARVIS_CHAT_TIMEOUT_SECONDS": "17",
        }
        with patch.dict(os.environ, compatible_env, clear=True):
            assert_v3_environment_custody()
            if os.environ.get("JARVIS_CHAT_TIMEOUT_SECONDS") != "17":
                raise SystemExit("non-state inherited environment was not preserved")

        with patch.dict(
            os.environ,
            {**compatible_env, "JARVIS_DATA_DIR": str(root / "conflicting-data")},
            clear=True,
        ):
            try:
                assert_v3_environment_custody()
            except V3InheritedEnvironmentConflict as exc:
                if exc.keys != ("JARVIS_DATA_DIR",):
                    raise SystemExit(f"custody conflict returned unexpected keys: {exc.keys}")
            else:
                raise SystemExit("conflicting inherited custody was accepted")

        with patch.dict(
            os.environ,
            {**compatible_env, "JARVIS_STATUS_AUTH_TOKEN": ""},
            clear=True,
        ):
            try:
                assert_v3_environment_custody()
            except V3InheritedEnvironmentConflict as exc:
                if exc.keys != ("JARVIS_STATUS_AUTH_TOKEN",):
                    raise SystemExit(f"blank custody conflict returned unexpected keys: {exc.keys}")
            else:
                raise SystemExit("blank inherited custody masked the selected V3 value")


def test_selected_environment_python_is_authoritative_and_bounded() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-selected-python-") as temp:
        root = Path(temp)
        selected = root / "selected-python-private-name"
        inherited = root / "inherited-python-private-name"
        current = root / "current-python-private-name"
        unsafe = root / "unsafe-python-private-name"
        for candidate in (selected, inherited, current, unsafe):
            candidate.write_text("", encoding="utf-8")
            candidate.chmod(0o700)
        unsafe.chmod(0o720)

        env_file = root / "runtime.env"
        env_file.write_text(
            f"JARVIS_V3_PYTHON={selected}\n",
            encoding="utf-8",
        )
        env_file.chmod(0o600)

        inspected: list[Path] = []

        def inspector(
            path: Path,
            _required: tuple[str, ...],
            _preferred: tuple[str, ...],
        ) -> V3PythonProbe:
            inspected.append(path)
            return V3PythonProbe(path, (3, 14), (), ())

        with patch.dict(
            os.environ,
            {"JARVIS_V3_ENV": str(env_file)},
            clear=True,
        ):
            custody = assert_v3_environment_custody()
            configured = v3_python_runtime._configured_python(custody)
            if configured != selected:
                raise SystemExit("selected owner-only environment Python was ignored")
            chosen = preferred_v3_python(
                current_executable=current,
                project_root=root / "no-local-environment",
                configured=configured,
                common_candidates=(inherited,),
                preferred_modules=(),
                inspector=inspector,
            )
            if chosen != selected or inspected != [selected]:
                raise SystemExit(
                    "selected owner-only environment Python was not authoritative"
                )

        inspected.clear()
        with patch.dict(
            os.environ,
            {
                "JARVIS_V3_ENV": str(env_file),
                "JARVIS_V3_PYTHON": str(selected),
            },
            clear=True,
        ):
            custody = assert_v3_environment_custody()
            if v3_python_runtime._configured_python(custody) != selected:
                raise SystemExit("exact inherited Python match changed selection")

        for inherited_value in (str(inherited), ""):
            inspected.clear()
            with patch.dict(
                os.environ,
                {
                    "JARVIS_V3_ENV": str(env_file),
                    "JARVIS_V3_PYTHON": inherited_value,
                },
                clear=True,
            ):
                try:
                    assert_v3_environment_custody()
                except V3InheritedEnvironmentConflict as exc:
                    if exc.keys != ("JARVIS_V3_PYTHON",):
                        raise SystemExit(
                            f"Python custody conflict returned unexpected keys: {exc.keys}"
                        )
                    rendered = str(exc)
                    if str(selected) in rendered or str(inherited) in rendered:
                        raise SystemExit("Python custody conflict leaked a configured path")
                else:
                    raise SystemExit("mismatched inherited Python escaped custody refusal")
            if inspected:
                raise SystemExit("conflicting inherited Python reached the import probe")

        empty_env = root / "runtime-without-python.env"
        empty_env.write_text("JARVIS_CHAT_TIMEOUT_SECONDS=17\n", encoding="utf-8")
        empty_env.chmod(0o600)
        with patch.dict(
            os.environ,
            {
                "JARVIS_V3_ENV": str(empty_env),
                "JARVIS_V3_PYTHON": str(inherited),
            },
            clear=True,
        ):
            try:
                assert_v3_environment_custody()
            except V3InheritedEnvironmentConflict as exc:
                if exc.keys != ("JARVIS_V3_PYTHON",):
                    raise SystemExit("missing selected Python returned the wrong conflict")
            else:
                raise SystemExit("inherited Python filled an omitted selected setting")

        with patch.dict(
            os.environ,
            {"JARVIS_V3_PYTHON": str(inherited)},
            clear=True,
        ):
            if v3_python_runtime._configured_python() != inherited:
                raise SystemExit("inherited Python was not preserved without a selected env")

        invalid_values = (
            "relative/python3",
            "~/private-python3",
            "/private/tmp/../unsafe-python3",
            "/private/tmp/unsafe\npython3",
            "/" + ("x" * (v3_python_runtime.MAX_CONFIGURED_PYTHON_PATH_BYTES + 1)),
        )
        for invalid in invalid_values:
            inspected.clear()
            try:
                v3_python_runtime._configured_python(
                    v3_python_runtime._V3EnvironmentCustody(True, invalid)
                )
            except V3PythonUnavailable as exc:
                if invalid in str(exc) or len(str(exc)) > 80:
                    raise SystemExit("invalid configured Python leaked its value")
            else:
                raise SystemExit("invalid configured Python spelling was accepted")
            if inspected:
                raise SystemExit("invalid configured Python reached the import probe")

        for unavailable in (root / "missing-python-private-name", unsafe):
            inspected.clear()
            try:
                preferred_v3_python(
                    current_executable=current,
                    project_root=root / "no-local-environment",
                    configured=unavailable,
                    common_candidates=(),
                    preferred_modules=(),
                    inspector=inspector,
                )
            except V3PythonUnavailable as exc:
                if str(unavailable) in str(exc):
                    raise SystemExit("unavailable configured Python leaked its path")
            else:
                raise SystemExit("unavailable configured Python was accepted")
            if inspected:
                raise SystemExit("unavailable configured Python reached the import probe")

        inspected.clear()
        no_loop = preferred_v3_python(
            current_executable=selected,
            project_root=root / "no-local-environment",
            configured=selected,
            common_candidates=(),
            preferred_modules=(),
            inspector=inspector,
        )
        if no_loop is not None or inspected != [selected]:
            raise SystemExit("selected environment Python caused a re-exec loop")

        hardlink_alias = root / "selected-python-hardlink-private-name"
        os.link(selected, hardlink_alias)
        if not v3_python_runtime._same_executable(selected, hardlink_alias):
            raise SystemExit("hardlink interpreter identity was not detected")

def test_selected_environment_python_reexec_uses_exact_candidate() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-selected-python-reexec-") as temp:
        root = Path(temp)
        selected = root / "selected-python-private-name"
        selected.write_text("", encoding="utf-8")
        selected.chmod(0o700)
        script = root / "launcher.py"
        script.write_text("", encoding="utf-8")
        custody = v3_python_runtime._V3EnvironmentCustody(True, str(selected))
        original_argv = list(sys.argv)
        sys.argv[:] = [str(script), "--help"]
        try:
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(
                    v3_python_runtime,
                    "assert_v3_environment_custody",
                    return_value=custody,
                ),
                patch.object(
                    v3_python_runtime,
                    "preferred_v3_python",
                    return_value=selected,
                ) as chooser,
                patch.object(v3_python_runtime.os, "execve") as execve,
            ):
                v3_python_runtime.reexec_with_v3_python(script)
        finally:
            sys.argv[:] = original_argv
        if chooser.call_count != 1:
            raise SystemExit("selected environment Python was resolved more than once")
        kwargs = chooser.call_args.kwargs
        if (
            kwargs.get("configured") != selected
            or kwargs.get("_configured_is_authoritative") is not True
        ):
            raise SystemExit("re-exec chooser did not receive the selected environment Python")
        if execve.call_count != 1:
            raise SystemExit("selected environment Python did not use one guarded re-exec")
        executable, argv, child_environment = execve.call_args.args
        if executable != str(selected) or argv != [
            str(selected),
            str(script.resolve()),
            "--help",
        ]:
            raise SystemExit("guarded re-exec changed the selected candidate or arguments")
        marker = child_environment.get(v3_python_runtime._V3_PYTHON_REEXEC_GUARD)
        if (
            type(marker) is not str
            or v3_python_runtime._SHA256_RE.fullmatch(marker) is None
            or str(selected) in marker
            or str(script) in marker
            or v3_python_runtime._V3_PYTHON_REEXEC_GUARD in os.environ
        ):
            raise SystemExit("guarded re-exec marker was missing, unbounded, or leaked authority")

        drift_candidate = root / "drift-python-private-name"
        drift_candidate.write_text("", encoding="utf-8")
        drift_candidate.chmod(0o700)
        markers = (
            marker,
            v3_python_runtime._reexec_guard(drift_candidate, script.resolve()),
            "not-a-valid-marker",
        )
        for inherited_marker in markers:
            output = StringIO()
            with (
                patch.dict(
                    os.environ,
                    {
                        v3_python_runtime._V3_PYTHON_REEXEC_GUARD: inherited_marker,
                    },
                    clear=True,
                ),
                patch.object(
                    v3_python_runtime,
                    "assert_v3_environment_custody",
                    return_value=custody,
                ),
                patch.object(
                    v3_python_runtime,
                    "preferred_v3_python",
                    return_value=selected,
                ),
                patch.object(v3_python_runtime.os, "execve") as blocked_execve,
                redirect_stderr(output),
            ):
                try:
                    v3_python_runtime.reexec_with_v3_python(script)
                except SystemExit as exc:
                    if exc.code != 78:
                        raise SystemExit("re-exec marker refusal used an unexpected exit")
                else:
                    raise SystemExit("inherited re-exec marker authorized another hop")
            rendered = output.getvalue()
            if (
                blocked_execve.called
                or "JARVIS_V3_PYTHON" not in rendered
                or "Traceback" in rendered
                or str(root) in rendered
                or selected.name in rendered
                or drift_candidate.name in rendered
                or inherited_marker in rendered
                or len(rendered) > 400
            ):
                raise SystemExit("re-exec marker refusal was unsafe or leaked local detail")

        child_cases = (
            ("valid", selected, marker, True),
            ("selection-drift", drift_candidate, marker, False),
            ("malformed", selected, "not-a-valid-marker", False),
        )
        for case_name, current, inherited_marker, should_pass in child_cases:
            output = StringIO()
            with (
                patch.dict(
                    os.environ,
                    {
                        v3_python_runtime._V3_PYTHON_REEXEC_GUARD: inherited_marker,
                    },
                    clear=True,
                ),
                patch.object(v3_python_runtime.sys, "executable", str(current)),
                patch.object(
                    v3_python_runtime,
                    "assert_v3_environment_custody",
                    return_value=custody,
                ),
                patch.object(
                    v3_python_runtime,
                    "preferred_v3_python",
                    return_value=None,
                ),
                patch.object(v3_python_runtime.os, "execve") as child_execve,
                redirect_stderr(output),
            ):
                try:
                    v3_python_runtime.reexec_with_v3_python(script)
                except SystemExit as exc:
                    if should_pass or exc.code != 78:
                        raise SystemExit(
                            f"{case_name} child marker used an unexpected refusal"
                        )
                else:
                    if not should_pass:
                        raise SystemExit(f"{case_name} child marker was consumed")
                    if v3_python_runtime._V3_PYTHON_REEXEC_GUARD in os.environ:
                        raise SystemExit("verified child marker was not consumed")
            rendered = output.getvalue()
            if (
                child_execve.called
                or (should_pass and rendered)
                or (not should_pass and "invalid interpreter restart marker" not in rendered)
                or "Traceback" in rendered
                or str(root) in rendered
                or inherited_marker in rendered
                or len(rendered) > 400
            ):
                raise SystemExit(f"{case_name} child marker handling was unsafe")


def test_selected_environment_python_target_policy_is_coherent() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-python-path-custody-") as temp:
        root = Path(temp)

        v2_parent = root / "jarvis-v2"
        v2_parent.mkdir()
        v2_candidate = v2_parent / "python3-private-name"
        v2_candidate.write_text("", encoding="utf-8")
        v2_candidate.chmod(0o700)

        safe_parent = root / "safe-parent-private-name"
        safe_parent.mkdir(mode=0o700)
        safe_candidate = safe_parent / "python3-private-name"
        safe_candidate.write_text("", encoding="utf-8")
        safe_candidate.chmod(0o700)
        symlink_parent = root / "symlink-parent-private-name"
        symlink_parent.symlink_to(safe_parent)
        symlink_candidate = root / "symlink-python-private-name"
        symlink_candidate.symlink_to(safe_candidate)
        resolved_v2_candidate = root / "resolved-v2-python-private-name"
        resolved_v2_candidate.symlink_to(v2_candidate)

        writable_parent = root / "writable-parent-private-name"
        writable_parent.mkdir(mode=0o700)
        writable_candidate = writable_parent / "python3-private-name"
        writable_candidate.write_text("", encoding="utf-8")
        writable_candidate.chmod(0o700)
        writable_parent.chmod(0o770)

        writable_file = root / "writable-python-private-name"
        writable_file.write_text("", encoding="utf-8")
        writable_file.chmod(0o720)

        hardlink_source = root / "hardlink-source-private-name"
        hardlink_source.write_text("", encoding="utf-8")
        hardlink_source.chmod(0o700)
        hardlink_candidate = root / "hardlink-python-private-name"
        os.link(hardlink_source, hardlink_candidate)

        env_file = root / "runtime.env"
        for case_name, candidate in (
            ("v2-spelling", v2_candidate),
            ("v2-resolved-target", resolved_v2_candidate),
        ):
            env_file.write_text(
                f"JARVIS_V3_PYTHON={candidate}\n",
                encoding="utf-8",
            )
            env_file.chmod(0o600)
            with patch.dict(
                os.environ,
                {"JARVIS_V3_ENV": str(env_file)},
                clear=True,
            ):
                try:
                    assert_v3_environment_custody()
                except v3_python_runtime.V3PathCustodyError as exc:
                    if (
                        exc.key != "JARVIS_V3_PYTHON"
                        or exc.reason_code != "overlaps_v2"
                    ):
                        raise SystemExit(
                            f"{case_name} Python custody returned the wrong bounded reason"
                        )
                    rendered = str(exc)
                    if (
                        str(root) in rendered
                        or candidate.name in rendered
                        or case_name in rendered
                        or len(rendered) > 120
                    ):
                        raise SystemExit(f"{case_name} Python custody leaked path detail")
                else:
                    raise SystemExit(f"{case_name} configured Python passed custody")

        current = root / "current-python-private-name"
        current.write_text("", encoding="utf-8")
        current.chmod(0o700)
        inspected: list[Path] = []

        def inspector(
            path: Path,
            _required: tuple[str, ...],
            _preferred: tuple[str, ...],
        ) -> V3PythonProbe:
            inspected.append(path)
            return V3PythonProbe(path, (3, 14), (), ())

        accepted = (
            symlink_parent / safe_candidate.name,
            symlink_candidate,
            writable_candidate,
            hardlink_candidate,
        )
        for candidate in accepted:
            inspected.clear()
            env_file.write_text(
                f"JARVIS_V3_PYTHON={candidate}\n",
                encoding="utf-8",
            )
            env_file.chmod(0o600)
            with patch.dict(
                os.environ,
                {"JARVIS_V3_ENV": str(env_file)},
                clear=True,
            ):
                custody = assert_v3_environment_custody()
                configured = v3_python_runtime._configured_python(custody)
                chosen = preferred_v3_python(
                    current_executable=current,
                    project_root=root / "no-local-environment",
                    configured=configured,
                    common_candidates=(),
                    preferred_modules=(),
                    inspector=inspector,
                )
            if chosen != candidate or inspected != [candidate]:
                raise SystemExit("reasonable explicit Python alias used a different target policy")

        inspected.clear()
        env_file.write_text(
            f"JARVIS_V3_PYTHON={writable_file}\n",
            encoding="utf-8",
        )
        env_file.chmod(0o600)
        with patch.dict(
            os.environ,
            {"JARVIS_V3_ENV": str(env_file)},
            clear=True,
        ):
            custody = assert_v3_environment_custody()
            configured = v3_python_runtime._configured_python(custody)
            try:
                preferred_v3_python(
                    current_executable=current,
                    project_root=root / "no-local-environment",
                    configured=configured,
                    common_candidates=(),
                    preferred_modules=(),
                    inspector=inspector,
                )
            except V3PythonUnavailable as exc:
                if str(writable_file) in str(exc):
                    raise SystemExit("unsafe explicit Python leaked its target")
            else:
                raise SystemExit("unsafe explicit Python target was accepted")
        if inspected:
            raise SystemExit("unsafe explicit Python target reached the import probe")


def test_interpreter_wrapper_reexec_stops_after_one_hop() -> None:
    live: Path | None = None
    for candidate in (
        Path(sys.executable),
        *v3_python_runtime.COMMON_V3_PYTHONS,
    ):
        if not v3_python_runtime._reasonable_python_executable(candidate):
            continue
        probe = inspect_v3_python(candidate, ("sqlite3", "ssl"), ())
        if probe is not None and probe.launch_ready:
            live = candidate
            break
    if live is None:
        raise SystemExit("no reasonable live Python was available for wrapper-loop coverage")

    with TemporaryDirectory(prefix="jarvis-v3-python-wrapper-loop-") as temp:
        root = Path(temp)
        wrapper = root / "python-wrapper-private-name"
        wrapper.write_text(
            "#!/bin/sh\n"
            f"exec {shlex.quote(str(live))} \"$@\"\n",
            encoding="utf-8",
        )
        wrapper.chmod(0o700)
        env_file = root / "runtime.env"
        env_file.write_text(
            f"JARVIS_V3_PYTHON={wrapper}\n",
            encoding="utf-8",
        )
        env_file.chmod(0o600)
        result = subprocess.run(
            [str(live), str(ROOT / "launch_jarvis_v3.py"), "--help"],
            cwd=str(ROOT),
            env={
                "HOME": str(root),
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": str(ROOT),
                "PYTHONDONTWRITEBYTECODE": "1",
                "JARVIS_V3_ENV": str(env_file),
            },
            capture_output=True,
            text=True,
            timeout=15,
        )
        rendered = result.stdout + result.stderr
        if (
            result.returncode != 78
            or "interpreter restart did not converge" not in rendered
            or "JARVIS_V3_PYTHON" not in rendered
            or "Traceback" in rendered
            or str(root) in rendered
            or wrapper.name in rendered
            or len(rendered) > 500
        ):
            raise SystemExit("interpreter wrapper loop did not stop with bounded recovery")


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-probe-custody-") as temp:
        probe_root = Path(temp)
        marker = probe_root / "probe-environment-result"
        wrapper = probe_root / "probe-python"
        sentinel_keys = (
            "JARVIS_STATUS_AUTH_TOKEN",
            "OPENAI_API_KEY",
            "JARVIS_PROBE_SENTINEL_SECRET",
        )
        condition = " || ".join(f'[ -n "${{{key}+x}}" ]' for key in sentinel_keys)
        wrapper.write_text(
            "#!/bin/sh\n"
            f"if {condition}; then\n"
            f"  printf present > {shlex.quote(str(marker))}\n"
            "else\n"
            f"  printf absent > {shlex.quote(str(marker))}\n"
            "fi\n"
            f"exec {shlex.quote(sys.executable)} \"$@\"\n",
            encoding="utf-8",
        )
        wrapper.chmod(0o700)
        with patch.dict(
            os.environ,
            {
                "JARVIS_STATUS_AUTH_TOKEN": "probe-must-not-see-dashboard-secret",
                "OPENAI_API_KEY": "probe-must-not-see-api-secret",
                "JARVIS_PROBE_SENTINEL_SECRET": "probe-must-not-see-sentinel",
            },
            clear=False,
        ):
            custody_probe = inspect_v3_python(wrapper, ("sqlite3", "ssl"), ())
        if custody_probe is None or not custody_probe.launch_ready:
            raise SystemExit("scrubbed interpreter probe could not inspect a compatible wrapper")
        if marker.read_text(encoding="utf-8") != "absent":
            raise SystemExit("interpreter candidate received inherited Jarvis credentials")

        with (
            patch.dict(
                os.environ,
                {"JARVIS_PROBE_SENTINEL_SECRET": "timeout-probe-secret"},
                clear=False,
            ),
            patch.object(
                v3_python_runtime.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired([str(wrapper)], 5),
            ) as timed_run,
        ):
            if inspect_v3_python(wrapper, ("sqlite3", "ssl"), ("optional_adapter",)) is not None:
                raise SystemExit("timed-out interpreter probe was accepted")
        probe_environment = timed_run.call_args.kwargs.get("env") or {}
        if any(key in probe_environment for key in sentinel_keys):
            raise SystemExit("timed-out interpreter probe inherited a secret env key")
        if set(probe_environment) != {
            "LANG",
            "LC_ALL",
            "PATH",
            "PYTHONDONTWRITEBYTECODE",
            "PYTHONNOUSERSITE",
        }:
            raise SystemExit(f"interpreter probe environment was not minimal: {probe_environment}")

    with TemporaryDirectory(prefix="jarvis-v3-python-runtime-") as temp:
        root = Path(temp)
        current = root / "current-python"
        arm = root / "opt-homebrew-python"
        intel = root / "usr-local-python"
        local = root / ".venv" / "bin" / "python3"
        configured = root / "configured-python"
        after_preferred = root / "after-preferred-python"
        group_writable = root / "group-writable-python"
        world_writable = root / "world-writable-python"
        local.parent.mkdir(parents=True)
        for candidate in (
            current,
            arm,
            intel,
            local,
            configured,
            after_preferred,
            group_writable,
            world_writable,
        ):
            candidate.write_text("", encoding="utf-8")
            candidate.chmod(0o700)
        group_writable.chmod(0o720)
        world_writable.chmod(0o702)

        def probe(
            path: Path,
            version: tuple[int, int],
            *,
            missing_required: tuple[str, ...] = (),
            missing_preferred: tuple[str, ...] = (),
        ) -> V3PythonProbe:
            return V3PythonProbe(path, version, missing_required, missing_preferred)

        probes = {
            current: probe(current, (3, 9)),
            local: probe(local, (3, 12), missing_preferred=("optional_adapter",)),
            arm: probe(arm, (3, 14), missing_required=("ssl",)),
            intel: probe(intel, (3, 12)),
            configured: probe(configured, (3, 12), missing_preferred=("optional_adapter",)),
            after_preferred: probe(after_preferred, (3, 14)),
            group_writable: probe(group_writable, (3, 14)),
            world_writable: probe(world_writable, (3, 14)),
        }
        inspected: list[Path] = []
        inspected_preferred: list[tuple[str, ...]] = []

        def inspector(
            path: Path,
            _required: tuple[str, ...],
            _preferred: tuple[str, ...],
        ) -> V3PythonProbe | None:
            inspected.append(path)
            inspected_preferred.append(_preferred)
            for candidate, candidate_probe in probes.items():
                if candidate.resolve() == path.resolve():
                    return V3PythonProbe(
                        path,
                        candidate_probe.version,
                        tuple(
                            name
                            for name in candidate_probe.missing_required
                            if name in _required
                        ),
                        tuple(
                            name
                            for name in candidate_probe.missing_preferred
                            if name in _preferred
                        ),
                    )
            return None

        with patch.dict(
            os.environ,
            {"JARVIS_MODEL_PROVIDER": "ollama"},
            clear=True,
        ):
            selected = preferred_v3_python(
                current_executable=current,
                project_root=root,
                common_candidates=(arm, intel),
                inspector=inspector,
            )
        if selected is None or selected.resolve() != local.resolve():
            raise SystemExit(
                f"stdlib Ollama runtime did not select the first core-compatible interpreter: {selected}"
            )
        if inspected_preferred != [()]:
            raise SystemExit(
                f"stdlib Ollama runtime still requested a preferred SDK module: {inspected_preferred}"
            )
        inspected.clear()
        inspected_preferred.clear()

        selected = preferred_v3_python(
            current_executable=current,
            project_root=root,
            common_candidates=(
                arm,
                group_writable,
                world_writable,
                intel,
                after_preferred,
            ),
            preferred_modules=("optional_adapter",),
            inspector=inspector,
        )
        if selected is None or selected.resolve() != intel.resolve():
            raise SystemExit(
                "Python 3.9 or a broken ARM candidate masked the compatible optional-adapter interpreter: "
                f"{selected}"
            )
        if any(
            path.resolve() in {group_writable.resolve(), world_writable.resolve()}
            for path in inspected
        ):
            raise SystemExit("group/world-writable auto-discovered interpreter was probed")
        if any(path.resolve() == after_preferred.resolve() for path in inspected):
            raise SystemExit("selection did not stop at the first preferred-ready interpreter")

        probes[local] = probe(local, (3, 12))
        inspected.clear()
        selected = preferred_v3_python(
            current_executable=current,
            project_root=root,
            common_candidates=(arm, intel),
            preferred_modules=("optional_adapter",),
            inspector=inspector,
        )
        if selected is None or selected.resolve() != local.resolve():
            raise SystemExit(f"ready project-local V3 interpreter was not preferred: {selected}")
        if len(inspected) != 1 or inspected[0].resolve() != local.resolve():
            raise SystemExit(f"project-local preferred interpreter did not short-circuit: {inspected}")

        probes[local] = probe(local, (3, 12), missing_preferred=("optional_adapter",))
        probes[current] = probe(current, (3, 12), missing_preferred=("optional_adapter",))
        probes[arm] = probe(arm, (3, 12), missing_preferred=("optional_adapter",))
        probes[intel] = probe(intel, (3, 12), missing_preferred=("optional_adapter",))
        probes[after_preferred] = probe(
            after_preferred,
            (3, 12),
            missing_preferred=("optional_adapter",),
        )
        inspected.clear()
        selected = preferred_v3_python(
            current_executable=current,
            project_root=root,
            common_candidates=(arm, intel, after_preferred),
            preferred_modules=("optional_adapter",),
            inspector=inspector,
        )
        if selected is None or selected.resolve() != local.resolve():
            raise SystemExit(f"earliest core-compatible fallback was not preserved: {selected}")
        if len(inspected) != 5:
            raise SystemExit(f"fallback selection did not inspect all candidates once: {inspected}")

        probes[current] = probe(current, (3, 12), missing_preferred=("optional_adapter",))
        selected = preferred_v3_python(
            current_executable=current,
            project_root=root / "no-local-environment",
            common_candidates=(arm, intel),
            preferred_modules=(),
            inspector=inspector,
        )
        if selected is not None:
            raise SystemExit(f"compatible current interpreter could recurse: {selected}")

        selected = preferred_v3_python(
            current_executable=current,
            configured=configured,
            preferred_modules=("optional_adapter",),
            inspector=inspector,
        )
        if selected is None or selected.resolve() != configured.resolve():
            raise SystemExit(f"compatible configured interpreter was not authoritative: {selected}")

        inspected.clear()
        try:
            preferred_v3_python(
                current_executable=current,
                configured=group_writable,
                preferred_modules=("optional_adapter",),
                inspector=inspector,
            )
        except V3PythonUnavailable:
            pass
        else:
            raise SystemExit("group-writable configured interpreter was accepted")
        if inspected:
            raise SystemExit("group-writable configured interpreter reached the import probe")

        probes[current] = probe(current, (3, 9))
        probes[arm] = probe(arm, (3, 10))
        probes[intel] = probe(intel, (3, 12), missing_required=("sqlite3",))
        try:
            preferred_v3_python(
                current_executable=current,
                project_root=root / "no-local-environment",
                common_candidates=(arm, intel),
                preferred_modules=("optional_adapter",),
                inspector=inspector,
            )
        except V3PythonUnavailable:
            pass
        else:
            raise SystemExit("incompatible interpreters did not fail closed")

    live_probe = inspect_v3_python(
        Path(sys.executable),
        ("sqlite3", "ssl"),
        (),
    )
    if live_probe is None or not live_probe.launch_ready:
        raise SystemExit("the active smoke interpreter failed its real compatibility probe")

    error_output = StringIO()
    with (
        patch.object(v3_python_runtime, "assert_v3_environment_custody", lambda: None),
        patch.object(
            v3_python_runtime,
            "preferred_v3_python",
            side_effect=V3PythonUnavailable("synthetic unavailable interpreter"),
        ),
        redirect_stderr(error_output),
    ):
        try:
            v3_python_runtime.reexec_with_v3_python(ROOT / "launch_jarvis_v3.py")
        except SystemExit as exc:
            if exc.code != 78:
                raise SystemExit(f"missing interpreter used unexpected exit code: {exc.code}")
        else:
            raise SystemExit("missing interpreter did not stop before runtime import")
    rendered_error = error_output.getvalue()
    if (
        "Python 3.11 or newer" not in rendered_error
        or "JARVIS_V3_PYTHON" not in rendered_error
        or "Traceback" in rendered_error
        or "synthetic unavailable" in rendered_error
        or len(rendered_error) > 700
    ):
        raise SystemExit(f"missing-interpreter recovery was not bounded: {rendered_error!r}")

    test_v3_launchers_reexec_before_entrypoint_import()
    test_fresh_user_bootstrap_is_isolated_and_owner_only()
    test_inherited_custody_conflicts_fail_before_runtime_import()
    test_unreadable_selected_environment_fails_bounded_before_runtime_import()
    test_symlink_loop_path_custody_fails_bounded_before_runtime_import()
    test_exact_custody_matches_and_non_state_env_are_preserved()
    test_selected_environment_python_is_authoritative_and_bounded()
    test_selected_environment_python_reexec_uses_exact_candidate()
    test_selected_environment_python_target_policy_is_coherent()
    test_interpreter_wrapper_reexec_stops_after_one_hop()
    print("V3 Python runtime selection smoke passed")


if __name__ == "__main__":
    main()
