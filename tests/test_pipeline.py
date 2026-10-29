"""
Testes de ponta a ponta sem baixar modelos: os dois motores sao substituidos
por dublês (um whisper-cli falso e um modulo faster_whisper falso), e o resto
do pipeline roda de verdade (ffmpeg, PyMuPDF, manifesto, indice).

    pip install pytest pymupdf
    pytest -q
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import transcrever_pos as tp  # noqa: E402

precisa_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg ausente")


def gerar_video(caminho: Path, segundos: int = 3) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={segundos}",
         "-f", "lavfi", "-i", f"color=c=black:s=64x64:d={segundos}", "-shortest", str(caminho)],
        check=True,
    )


def gerar_pdf(caminho: Path, paginas: list[str]) -> None:
    import pymupdf
    caminho.parent.mkdir(parents=True, exist_ok=True)
    d = pymupdf.open()
    for txt in paginas:
        p = d.new_page()
        if txt:
            p.insert_text((60, 90), txt, fontsize=12)
    d.save(caminho)
    d.close()


@pytest.fixture
def whisper_cli_falso(tmp_path: Path) -> Path:
    """Imita o whisper-cli: grava <of>.json no formato do whisper.cpp e loga Vulkan."""
    script = tmp_path / "whisper_cli_falso.py"
    script.write_text(textwrap.dedent("""
        import json, sys
        a = sys.argv
        of = a[a.index("-of") + 1]
        if "-np" not in a:  # como o real: -np silencia o log de inicializacao
            sys.stderr.write("ggml_vulkan: Found 1 Vulkan devices:\\n")
        dados = {"transcription": [
            {"offsets": {"from": 0, "to": 1500}, "text": " Kafka garante ordem por partição."},
            {"offsets": {"from": 1500, "to": 3000}, "text": " Idempotência evita duplicidade."},
        ]}
        open(of + ".json", "w", encoding="utf-8").write(json.dumps(dados, ensure_ascii=False))
    """), encoding="utf-8")
    if sys.platform == "win32":
        lancador = tmp_path / "whisper-cli.bat"
        lancador.write_text(f'@"{sys.executable}" "{script}" %*\n', encoding="utf-8")
    else:
        lancador = tmp_path / "whisper-cli"
        lancador.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
        lancador.chmod(0o755)
    return lancador


@precisa_ffmpeg
def test_motor_whispercpp_ponta_a_ponta(tmp_path: Path, whisper_cli_falso: Path, capsys):
    origem, destino = tmp_path / "XP", tmp_path / "XP" / "TRANSCRICAO"
    gerar_video(origem / "Modulo 1 - Streaming" / "aula 01.mp4")
    gerar_pdf(origem / "Modulo 1 - Streaming" / "apostila.pdf", ["Kafka e Spark", ""])
    modelo = tmp_path / "ggml-fake.bin"
    modelo.write_bytes(b"0")

    args = ["--origem", str(origem), "--destino", str(destino), "--motor", "whispercpp",
            "--whisper-cli", str(whisper_cli_falso), "--modelo-ggml", str(modelo)]
    tp.main(args)

    md = (destino / "Modulo 1 - Streaming" / "aula 01.md").read_text(encoding="utf-8")
    assert "Kafka garante ordem por partição." in md
    assert "Idempotência evita duplicidade." in md
    assert "duracao 00:00:03" in md
    assert "GPU via Vulkan detectada" in capsys.readouterr().out

    pdf_md = (destino / "Modulo 1 - Streaming" / "apostila.md").read_text(encoding="utf-8")
    assert "## Pagina 1" in pdf_md and "Kafka e Spark" in pdf_md

    indice = (destino / "INDICE.md").read_text(encoding="utf-8")
    assert "Modulo%201%20-%20Streaming/aula%2001.md" in indice

    manifesto = json.loads((destino / tp.MANIFESTO).read_text(encoding="utf-8"))
    assert len(manifesto) == 2
    assert any(v.get("motor") == "whispercpp" for v in manifesto.values())

    # Segunda execucao: nada a refazer (checkpoint idempotente)
    tp.main(args)
    assert "0 processados, 2 ja existentes, 0 falhas" in capsys.readouterr().out


@precisa_ffmpeg
def test_motor_faster_whisper_com_modulo_falso(tmp_path: Path, monkeypatch):
    class Seg:
        def __init__(self, a, b, t):
            self.start, self.end, self.text = a, b, t

    class WhisperModelFalso:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, *a, **k):
            return iter([Seg(0.0, 2.0, " Olá, turma."), Seg(2.0, 3.0, " ")]), None

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=WhisperModelFalso))
    origem, destino = tmp_path / "in", tmp_path / "out"
    gerar_video(origem / "a.mp4", 2)
    tp.main(["--origem", str(origem), "--destino", str(destino), "--dispositivo", "cpu"])
    md = (destino / "a.md").read_text(encoding="utf-8")
    assert "Olá, turma." in md


def test_whispercpp_exige_modelo(tmp_path: Path):
    with pytest.raises(SystemExit):
        tp.MotorWhisperCpp("whisper-cli-inexistente", str(tmp_path / "nao.bin"), "pt", 2)


def test_paragrafos_por_tempo():
    segs = [(i * 10.0, i * 10.0 + 10.0, f"frase {i}") for i in range(10)]
    texto, total = tp.segmentos_para_markdown(Path("x.mp4"), 100.0, segs)
    assert total == 10
    assert texto.count("**[") == 2  # ~45 s por paragrafo => 2 blocos em 100 s
