#!/usr/bin/env python3
"""
transcrever_pos.py
------------------
Pipeline local que transforma uma biblioteca de videos e PDFs em uma base de
conhecimento em Markdown, pronta para busca, estudo e reaproveitamento.

Projetado para rodar por horas ou dias sem supervisao:
  - espelha a arvore de pastas de origem no destino
  - mantem um manifesto (checkpoint): o que ja foi processado nao reprocessa
  - pode ser interrompido com Ctrl+C e retomado depois sem perder trabalho
  - gera um INDICE.md navegavel ao final

Dois motores de transcricao:
  faster-whisper  CPU (qualquer maquina) ou GPU NVIDIA (CUDA)
  whispercpp      GPU AMD/Intel/NVIDIA via Vulkan, usando o whisper-cli do whisper.cpp

INSTALACAO (comum aos dois motores)
    pip install pymupdf
    # ffmpeg no PATH:  winget install Gyan.FFmpeg   (Windows)

MOTOR faster-whisper
    pip install faster-whisper

MOTOR whispercpp (GPU AMD no Windows)
    1. Baixe um build do whisper.cpp compilado com Vulkan para Windows x64 e
       extraia numa pasta, por exemplo C:\\whisper. O projeto oficial
       (github.com/ggml-org/whisper.cpp) ainda nao publica esse build; ha builds
       de terceiros, ou compile o oficial com Visual Studio Build Tools +
       Vulkan SDK + cmake -DGGML_VULKAN=1.
    2. Baixe o modelo (recomendado: rapido e bom em portugues):
       https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin
    3. Teste: C:\\whisper\\whisper-cli.exe --help
       Na primeira transcricao, o log deve mostrar "ggml_vulkan: Found 1 Vulkan devices".

USO TIPICO
    # 1) Inventario: quanto conteudo existe, sem processar nada
    python transcrever_pos.py --origem "G:/XP" --destino "G:/XP/TRANSCRICAO" --inventario

    # 2) PDFs (minutos)
    python transcrever_pos.py --origem "G:/XP" --destino "G:/XP/TRANSCRICAO" --apenas pdf

    # 3) Piloto de 3 videos na GPU AMD
    python transcrever_pos.py --origem "G:/XP" --destino "G:/XP/TRANSCRICAO" --limite 3 ^
        --motor whispercpp --whisper-cli "C:/whisper/whisper-cli.exe" ^
        --modelo-ggml "C:/whisper/ggml-large-v3-turbo-q5_0.bin"

    # 4) Rodada completa: o mesmo comando sem --limite (pode fechar e retomar)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator
from urllib.parse import quote

EXTS_VIDEO = {".mp4", ".mkv", ".mov", ".avi", ".wmv", ".webm", ".m4v", ".flv"}
EXTS_AUDIO = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".wma", ".flac"}
EXTS_MIDIA = EXTS_VIDEO | EXTS_AUDIO
EXTS_PDF = {".pdf"}
MANIFESTO = "_manifesto.json"
PARAGRAFO_S = 45  # fecha um paragrafo a cada ~45 s de fala

# Um segmento transcrito: (inicio_s, fim_s, texto)
Segmento = tuple[float, float, str]


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def humanizar_bytes(n: float) -> str:
    for unidade in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unidade == "TB":
            return f"{n:.1f} {unidade}"
        n /= 1024
    return f"{n:.1f} TB"


def hhmmss(segundos: float) -> str:
    segundos = int(segundos)
    return f"{segundos // 3600:02d}:{(segundos % 3600) // 60:02d}:{segundos % 60:02d}"


def chave_arquivo(caminho: Path) -> str:
    """Identidade barata e estavel: caminho + tamanho + data de modificacao."""
    st = caminho.stat()
    return f"{caminho.as_posix()}|{st.st_size}|{int(st.st_mtime)}"


def exige_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        sys.exit(
            "ERRO: ffmpeg nao encontrado no PATH.\n"
            "  Windows: winget install Gyan.FFmpeg\n"
            "  macOS:   brew install ffmpeg\n"
            "  Linux:   sudo apt install ffmpeg"
        )


def duracao_wav(caminho: Path) -> float:
    with wave.open(str(caminho), "rb") as w:
        return w.getnframes() / float(w.getframerate())


# --------------------------------------------------------------------------
# Manifesto (checkpoint)
# --------------------------------------------------------------------------

@dataclass
class Manifesto:
    caminho: Path
    feitos: dict = field(default_factory=dict)

    @classmethod
    def carregar(cls, destino: Path) -> "Manifesto":
        p = destino / MANIFESTO
        if p.exists():
            try:
                return cls(p, json.loads(p.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                log("AVISO: manifesto corrompido, recomecando o controle do zero.")
        return cls(p, {})

    def ja_feito(self, chave: str) -> bool:
        return chave in self.feitos

    def marcar(self, chave: str, saida: Path, meta: dict) -> None:
        self.feitos[chave] = {"saida": saida.as_posix(), **meta}
        self.salvar()

    def salvar(self) -> None:
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.caminho.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.feitos, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.caminho)  # escrita atomica: um Ctrl+C nao corrompe o manifesto


# --------------------------------------------------------------------------
# Extracao de PDF
# --------------------------------------------------------------------------

def pdf_para_markdown(origem: Path, destino: Path) -> dict:
    try:
        import pymupdf as fitz
    except ImportError:  # versoes antigas da biblioteca
        import fitz

    doc = fitz.open(origem)
    total_paginas = doc.page_count
    partes = [f"# {origem.stem}\n", f"> Fonte: `{origem.name}` — {total_paginas} paginas\n"]
    vazias = 0

    for i, pagina in enumerate(doc, start=1):
        texto = pagina.get_text("text").strip()
        if not texto:
            vazias += 1
            continue
        texto = re.sub(r"\n{3,}", "\n\n", texto)
        partes.append(f"\n## Pagina {i}\n\n{texto}\n")

    doc.close()
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text("\n".join(partes), encoding="utf-8")

    if vazias:
        log(f"    nota: {vazias} de {total_paginas} pagina(s) sem texto extraivel "
            f"(provavel PDF escaneado — precisara de OCR)")

    return {"tipo": "pdf", "paginas": total_paginas, "paginas_vazias": vazias}


# --------------------------------------------------------------------------
# Motores de transcricao
# --------------------------------------------------------------------------

def extrair_audio(origem: Path, destino_wav: Path) -> None:
    """WAV 16 kHz mono PCM: o formato que os dois motores esperam."""
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(origem),
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(destino_wav)],
        check=True,
    )


class MotorFasterWhisper:
    nome = "faster-whisper"

    def __init__(self, modelo: str, dispositivo: str, idioma: str):
        from faster_whisper import WhisperModel

        if dispositivo == "auto":
            try:
                import ctranslate2
                dispositivo = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                dispositivo = "cpu"
        tipo_calculo = "float16" if dispositivo == "cuda" else "int8"
        log(f"Carregando Whisper '{modelo}' em {dispositivo} ({tipo_calculo})...")
        self.modelo = WhisperModel(modelo, device=dispositivo, compute_type=tipo_calculo)
        self.idioma = idioma

    def transcrever(self, wav: Path) -> Iterator[Segmento]:
        segmentos, _info = self.modelo.transcribe(
            str(wav), language=self.idioma, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 700}, beam_size=5,
        )
        for s in segmentos:
            yield (s.start, s.end, s.text)


class MotorWhisperCpp:
    nome = "whispercpp"

    def __init__(self, cli: str, modelo_ggml: str, idioma: str, threads: int):
        caminho_cli = shutil.which(cli) or (cli if Path(cli).is_file() else None)
        if not caminho_cli:
            sys.exit(f"ERRO: whisper-cli nao encontrado: {cli}\n"
                     "  Informe o caminho completo com --whisper-cli, ex.: C:/whisper/whisper-cli.exe")
        if not Path(modelo_ggml).is_file():
            sys.exit(f"ERRO: modelo ggml nao encontrado: {modelo_ggml}\n"
                     "  Baixe, por exemplo, ggml-large-v3-turbo-q5_0.bin e informe com --modelo-ggml")
        self.cli, self.modelo, self.idioma, self.threads = caminho_cli, modelo_ggml, idioma, threads
        self.gpu_checada = False
        log(f"Motor whisper.cpp: {self.cli}  modelo: {Path(self.modelo).name}")

    def transcrever(self, wav: Path) -> Iterator[Segmento]:
        base = wav.with_suffix("")
        # Sem "-np": essa opcao tambem silencia o log de inicializacao, onde o
        # whisper.cpp informa a GPU ("ggml_vulkan: Found N Vulkan devices").
        cmd = [self.cli, "-m", self.modelo, "-f", str(wav), "-l", self.idioma,
               "-oj", "-of", str(base), "-t", str(self.threads)]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        saida_log = (proc.stderr or "") + (proc.stdout or "")
        if not self.gpu_checada:
            self.gpu_checada = True
            if "ggml_vulkan" in saida_log or "Vulkan" in saida_log:
                log("    GPU via Vulkan detectada pelo whisper.cpp.")
            else:
                log("    AVISO: nao vi sinal de Vulkan no log. Este whisper-cli pode estar rodando so na CPU.")
        if proc.returncode != 0:
            ultimas = "\n".join(saida_log.strip().splitlines()[-8:])
            raise RuntimeError(f"whisper-cli terminou com codigo {proc.returncode}:\n{ultimas}")

        arq_json = base.with_suffix(".json")
        if not arq_json.exists():
            raise RuntimeError("whisper-cli nao gerou o arquivo JSON esperado.")
        dados = json.loads(arq_json.read_text(encoding="utf-8", errors="replace"))
        for item in dados.get("transcription", []):
            off = item.get("offsets", {})
            yield (off.get("from", 0) / 1000.0, off.get("to", 0) / 1000.0, item.get("text", ""))


def segmentos_para_markdown(origem: Path, duracao: float, segmentos: Iterable[Segmento],
                            progresso: Callable[[float], None] | None = None) -> tuple[str, int]:
    linhas = [
        f"# {origem.stem}\n",
        f"> Fonte: `{origem.name}` — duracao {hhmmss(duracao)}\n",
        "\n## Transcricao\n",
    ]
    bloco: list[str] = []
    inicio_bloco = 0.0
    total = 0
    for inicio, fim, texto in segmentos:
        texto = texto.strip()
        if not texto:
            continue
        total += 1
        if not bloco:
            inicio_bloco = inicio
        bloco.append(texto)
        if fim - inicio_bloco >= PARAGRAFO_S:
            linhas.append(f"\n**[{hhmmss(inicio_bloco)}]** {' '.join(bloco)}\n")
            bloco = []
        if progresso and total % 100 == 0:
            progresso(fim)
    if bloco:
        linhas.append(f"\n**[{hhmmss(inicio_bloco)}]** {' '.join(bloco)}\n")
    return "\n".join(linhas), total


def midia_para_markdown(origem: Path, destino: Path, motor) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        extrair_audio(origem, wav)
        duracao = duracao_wav(wav)
        texto, total = segmentos_para_markdown(
            origem, duracao, motor.transcrever(wav),
            progresso=lambda t: log(f"    ... {hhmmss(t)} de {hhmmss(duracao)}"),
        )
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(texto, encoding="utf-8")
    return {"tipo": "midia", "motor": motor.nome, "duracao_s": round(duracao, 1), "segmentos": total}


# --------------------------------------------------------------------------
# Varredura, inventario e indice
# --------------------------------------------------------------------------

def varrer(origem: Path, apenas: str | None, ignorar: Path | None = None) -> list[Path]:
    alvo: set[str] = set()
    if apenas in (None, "video"):
        alvo |= EXTS_MIDIA
    if apenas in (None, "pdf"):
        alvo |= EXTS_PDF
    arquivos = []
    ign = ignorar.resolve() if ignorar is not None else None
    for p in sorted(origem.rglob("*")):
        if ign is not None and ign in p.resolve().parents:
            continue
        if p.is_file() and p.suffix.lower() in alvo and not p.name.startswith(("~$", ".")):
            arquivos.append(p)
    return arquivos


def inventariar(arquivos: list[Path]) -> None:
    por_tipo: dict[str, list[Path]] = {"video/audio": [], "pdf": []}
    for p in arquivos:
        por_tipo["pdf" if p.suffix.lower() in EXTS_PDF else "video/audio"].append(p)

    print("\n" + "=" * 66)
    print("INVENTARIO")
    print("=" * 66)
    for tipo, itens in por_tipo.items():
        tamanho = sum(p.stat().st_size for p in itens)
        print(f"  {tipo:<12} {len(itens):>5} arquivos   {humanizar_bytes(tamanho):>10}")
    total = sum(p.stat().st_size for p in arquivos)
    print(f"  {'TOTAL':<12} {len(arquivos):>5} arquivos   {humanizar_bytes(total):>10}")

    midia = sum(p.stat().st_size for p in por_tipo["video/audio"])
    if midia:
        # estimativa grosseira: ~0.7 GB por hora de video de aula em 720p
        horas = midia / (0.7 * 1024 ** 3)
        print(f"\n  Estimativa de midia: ~{horas:.0f} horas de video/audio")
        print("  Tempo de transcricao estimado (varia muito com o hardware):")
        print(f"    GPU NVIDIA (faster-whisper, large-v3) ....... ~{horas / 8:.0f} a {horas / 4:.0f} h")
        print(f"    GPU AMD/Intel (whisper.cpp Vulkan, turbo) .... ~{horas / 8:.0f} a {horas / 3:.0f} h")
        print(f"    CPU (faster-whisper, medium int8) ............ ~{horas / 1.5:.0f} a {horas * 1.5:.0f} h")
    print("=" * 66 + "\n")


def gerar_indice(destino: Path) -> None:
    linhas = ["# Indice da base de conhecimento\n"]
    atual = None
    for md in sorted(destino.rglob("*.md")):
        if md.name == "INDICE.md":
            continue
        pasta = md.parent.relative_to(destino).as_posix() or "(raiz)"
        if pasta != atual:
            linhas.append(f"\n## {pasta}\n")
            atual = pasta
        alvo = quote(md.relative_to(destino).as_posix())
        linhas.append(f"- [{md.stem}]({alvo})")
    (destino / "INDICE.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    log(f"Indice gerado: {destino / 'INDICE.md'}")


# --------------------------------------------------------------------------
# Orquestracao
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Transcreve videos e extrai PDFs para Markdown.")
    ap.add_argument("--origem", required=True, help="Pasta raiz com o conteudo")
    ap.add_argument("--destino", required=True, help="Pasta onde a base sera gravada")
    ap.add_argument("--motor", default="faster-whisper", choices=["faster-whisper", "whispercpp"],
                    help="faster-whisper (CPU/NVIDIA) ou whispercpp (GPU AMD/Intel via Vulkan)")
    ap.add_argument("--modelo", default="medium",
                    help="[faster-whisper] tiny, base, small, medium, large-v3 (padrao: medium)")
    ap.add_argument("--dispositivo", default="auto", choices=["auto", "cpu", "cuda"],
                    help="[faster-whisper] onde rodar")
    ap.add_argument("--whisper-cli", default="whisper-cli", help="[whispercpp] caminho do whisper-cli(.exe)")
    ap.add_argument("--modelo-ggml", help="[whispercpp] caminho do modelo .bin, ex.: ggml-large-v3-turbo-q5_0.bin")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) // 2),
                    help="[whispercpp] threads de CPU de apoio")
    ap.add_argument("--idioma", default="pt")
    ap.add_argument("--limite", type=int, default=0, help="Processa no maximo N arquivos (piloto)")
    ap.add_argument("--apenas", choices=["video", "pdf"], help="Processa so um tipo")
    ap.add_argument("--inventario", action="store_true", help="So lista o que existe e sai")
    args = ap.parse_args(argv)

    origem, destino = Path(args.origem).expanduser(), Path(args.destino).expanduser()
    if not origem.is_dir():
        sys.exit(f"ERRO: pasta de origem nao encontrada: {origem}")

    arquivos = varrer(origem, args.apenas, ignorar=destino)
    if not arquivos:
        sys.exit("Nenhum arquivo compativel encontrado.")

    inventariar(arquivos)
    if args.inventario:
        return

    destino.mkdir(parents=True, exist_ok=True)
    manifesto = Manifesto.carregar(destino)

    pendentes_midia = [p for p in arquivos if p.suffix.lower() in EXTS_MIDIA
                       and not manifesto.ja_feito(chave_arquivo(p))]
    motor = None
    if pendentes_midia:
        exige_ffmpeg()
        if args.motor == "whispercpp":
            if not args.modelo_ggml:
                sys.exit("ERRO: o motor whispercpp exige --modelo-ggml (caminho do arquivo .bin).")
            motor = MotorWhisperCpp(args.whisper_cli, args.modelo_ggml, args.idioma, args.threads)
        else:
            motor = MotorFasterWhisper(args.modelo, args.dispositivo, args.idioma)

    processados = pulados = falhas = 0
    inicio = time.time()

    for i, arquivo in enumerate(arquivos, start=1):
        if args.limite and processados >= args.limite:
            log(f"Limite de {args.limite} arquivo(s) atingido. Encerrando o piloto.")
            break

        chave = chave_arquivo(arquivo)
        if manifesto.ja_feito(chave):
            pulados += 1
            continue

        relativo = arquivo.relative_to(origem)
        saida = destino / relativo.with_suffix(".md")
        log(f"[{i}/{len(arquivos)}] {relativo}")

        try:
            if arquivo.suffix.lower() in EXTS_PDF:
                meta = pdf_para_markdown(arquivo, saida)
            else:
                meta = midia_para_markdown(arquivo, saida, motor)
            manifesto.marcar(chave, saida, meta)
            processados += 1
        except KeyboardInterrupt:
            log("Interrompido. O progresso ja gravado sera retomado na proxima execucao.")
            break
        except Exception as e:
            falhas += 1
            log(f"    FALHA: {type(e).__name__}: {e}")

    gerar_indice(destino)
    print()
    log(f"Concluido em {hhmmss(time.time() - inicio)} — {processados} processados, "
        f"{pulados} ja existentes, {falhas} falhas.")
    log(f"Base em: {destino}")


if __name__ == "__main__":
    main()
