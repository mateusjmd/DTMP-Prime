#!/usr/bin/env python3
"""
DTMP-Prime Web Interface — backend Flask.

Uso:
  python -m web.app        (a partir da raiz do repositório)
  # ou: python web/app.py
  # Abre em http://localhost:5000

Dependências: flask, pandas, pyyaml (ver requirements.txt).
"""
import os, sys, json, uuid, time, threading, subprocess
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, abort

# Permite executar diretamente (python web/app.py) além de python -m web.app
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from dtmp_prime import paths

# WEB_DIR: onde vive o frontend.html. ROOT: raiz do repositório, que é o
# diretório de trabalho correto para o subprocesso do pipeline.
# Antes, WORK_DIR era o diretório de web/ e servia para as duas coisas: o
# comando montado apontava para web/main.py (inexistente) e o cwd do
# subprocesso fazia todos os caminhos relativos resolverem contra web/.
WEB_DIR = Path(__file__).parent.resolve()
ROOT = paths.ROOT
JOBS_DIR = ROOT / ".dtmp_jobs"
JOBS_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False          # preserva acentos no JSON
_jobs: dict = {}
_lock = threading.Lock()


# ── Configuração ──────────────────────────────────────────────────────────────

def _base_config() -> dict:
    """Lê o config.yaml da raiz como base do config de cada job.

    É o que garante que a escolha de modelo (Produção x Completude) feita no
    config.yaml valha também para as execuções disparadas pela interface.
    """
    try:
        with open(paths.DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
            return yaml.load(fh, Loader=yaml.FullLoader) or {}
    except FileNotFoundError:
        return {}
    except yaml.YAMLError:
        return {}


# ── Pipeline runner (thread separada) ─────────────────────────────────────────

def _run(jid: str, inp: Path, cfg: Path, out: Path) -> None:
    job = _jobs[jid]
    job["status"] = "running"
    job["started"] = datetime.now().isoformat()

    # Invoca o pacote como módulo: main.py usa imports relativos e não pode
    # mais ser executado por caminho. sys.executable garante o mesmo interpretador
    # (e o mesmo ambiente virtual) que roda o Flask.
    cmd = [
        sys.executable, "-m", "dtmp_prime.main",
        "-f", str(inp), "-c", str(cfg), "-o", str(out),
    ]

    def log(msg: str):
        job["logs"].append({"t": round(time.time(), 2), "m": str(msg)})

    try:
        log("$ " + " ".join(str(x) for x in cmd))
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"          # logs em tempo real no painel
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            cwd=str(ROOT), env=env,
        )
        for line in iter(proc.stdout.readline, ""):
            log(line.rstrip("\n"))
        proc.stdout.close()
        rc = proc.wait()
        job["files"] = sorted(f.name for f in out.glob("*") if f.is_file())
        job["status"] = "done" if rc == 0 else "error"
        log(f"[Processo encerrado · código de saída {rc}]")
    except Exception as e:
        log(f"[ERRO: {e}]")
        job["status"] = "error"

    job["ended"] = datetime.now().isoformat()


# ── Serialização segura ───────────────────────────────────────────────────────

def _json_safe(obj):
    """Converte NaN/Inf e tipos numpy para algo que JSON.parse aceite.

    ``df.where(pd.notnull, None)`` não basta: em colunas float o pandas converte
    o None de volta para NaN, e o jsonify emite o token literal ``NaN``, que o
    JSON.parse do navegador rejeita — quebrando a página inteira, não só a célula.
    """
    import math
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if hasattr(obj, "item"):          # escalares numpy
        try:
            return _json_safe(obj.item())
        except Exception:
            return str(obj)
    return obj


# ── Rotas ──────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    resp = send_from_directory(str(WEB_DIR), "frontend.html")
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    return resp


@app.route("/api/run", methods=["POST"])
def api_run():
    jid  = uuid.uuid4().hex[:8]
    jdir = JOBS_DIR / jid
    jdir.mkdir()
    out  = jdir / "out"
    out.mkdir()
    mode = request.form.get("mode", "manual")

    # ─ Salvar input ─
    if mode in ("vcf", "fasta"):
        f = request.files.get("file")
        if not f or not f.filename:
            return jsonify({"error": "Nenhum arquivo enviado."}), 400
        inp = jdir / ("input.vcf" if mode == "vcf" else "input.fa")
        f.save(str(inp))

    else:  # manual
        variants = json.loads(request.form.get("variants", "[]"))
        valid = [v for v in variants if v.get("ref", "").strip() and v.get("alt", "").strip()]
        if not valid:
            return jsonify({"error": "Defina ao menos uma variante com ref e alt preenchidos."}), 400
        inp = jdir / "input.fa"
        with open(inp, "w") as fh:
            for i, v in enumerate(valid, 1):
                name = (v.get("name") or f"variante_{i}").replace(" ", "_")
                fh.write(f">{name}_ref\n{v['ref'].strip()}\n")
                fh.write(f">{name}_alt\n{v['alt'].strip()}\n")

    # ─ Salvar config ─
    def fi(k, d):
        raw = request.form.get(k, str(d))
        try: return int(raw)
        except: return d

    # Parte do config.yaml da raiz como BASE. Antes o config do job era montado
    # do zero e não continha use_transformer / transformer_model / use_encoding /
    # feature_norm_path — como o default de use_transformer é False, TODA execução
    # pela web caía no ramo XGBoost e procurava um .pkl inexistente.
    cfg_data = _base_config()
    cfg_data.update({
        "scaffold":            cfg_data.get(
            "scaffold",
            "GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGC"),
        "n_jobs":              fi("n_jobs", 4),
        "debug":               max(1, fi("debug", 6)),   # mínimo 1 obrigatório
        "min_PBS_length":      fi("min_PBS_length", 10),
        "max_PBS_length":      fi("max_PBS_length", 15),
        "min_RTT_length":      fi("min_RTT_length", 10),
        "max_RTT_length":      fi("max_RTT_length", 20),
        "min_distance_RTT5":   fi("min_distance_RTT5", 5),
        "max_ngRNA_distance":  fi("max_ngRNA_distance", 100),
        "sgRNA_length":        fi("sgRNA_length", 20),
        "offset":              fi("offset", -3),
        "PAM":                 request.form.get("PAM", "NGG"),
        "gRNA_search_space":   fi("gRNA_search_space", 200),
        "max_target_to_sgRNA": fi("max_target_to_sgRNA", 10),
    })

    # Genoma: o campo do formulário só sobrescreve a base se vier preenchido.
    genome = request.form.get("genome_fasta", "").strip()
    if genome:
        cfg_data["genome_fasta"] = genome

    cfg_path = jdir / "config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.dump(cfg_data, fh, default_flow_style=False, allow_unicode=True)

    with _lock:
        _jobs[jid] = {
            "status": "pending", "logs": [], "files": [],
            "created": datetime.now().isoformat(),
        }

    threading.Thread(target=_run, args=(jid, inp, cfg_path, out), daemon=True).start()
    return jsonify({"job_id": jid})


@app.route("/api/job/<jid>")
def api_job(jid):
    if jid not in _jobs:
        abort(404)
    j = _jobs[jid]
    since = int(request.args.get("since", 0))
    return jsonify({
        "status":    j["status"],
        "logs":      j["logs"][since:],
        "log_count": len(j["logs"]),
        "files":     j.get("files", []),
    })


@app.route("/api/results/<jid>")
def api_results(jid):
    if jid not in _jobs:
        abort(404)
    out = JOBS_DIR / jid / "out"
    try:
        import pandas as pd
        data: dict = {}

        tops = sorted(out.glob("*top*pegRNA*.csv"))
        if tops:
            df = pd.read_csv(tops[0]).where(pd.notnull, None)

            # ── Derivações (parsing, sem alterar o pipeline) ──
            # A sequência do nicking guide já está embutida no ngRNA_name,
            # como a última porção após o padrão ..._{+|-}_SEQ (mesmo padrão
            # do sgRNA_name, cuja última porção reproduz sgRNA_seq).
            def _ngrna_seq_from_name(name):
                #if not name or str(name).lower() in ("", "nan", "none"):
                if not name or str(name).lower() in ("", "nan", "none", "0"):
                    return None
                return str(name).rsplit("_", 1)[-1]

            def _pe_mode(row):
                if not _ngrna_seq_from_name(row.get("ngRNA_name")):
                    return "PE2"
                is3b = row.get("is_PE3b") in (True, "True", 1, "1", 1.0)
                return "PE3b" if is3b else "PE3"

            if "ngRNA_name" in df.columns:
                df["ngRNA_seq"] = df["ngRNA_name"].map(_ngrna_seq_from_name)
            df["pe_mode"] = df.apply(_pe_mode, axis=1)

            # Ordem enxuta (espírito PRIDICT): campos essenciais primeiro.
            preferred = ["predicted_efficiency", "pe_mode", "sgRNA_seq",
                         "PBS_seq", "RTT_seq", "PBS_length", "RTT_length",
                         "strand", "is_PE3b", "DeepSpCas9",
                         "sgRNA_distance_to_ngRNA", "ngRNA_seq"]
            present = [c for c in preferred if c in df.columns]
            rest    = [c for c in df.columns if c not in present]
            df = df[present + rest]

            top_n = int(request.args.get("top_n", 500))
            df_top = df.head(top_n) if len(df) > top_n else df
            data["columns"] = list(df.columns)
            data["rows"]    = df_top.to_dict(orient="records")
            data["total"]   = len(df)

            # ── Agrupamento por variante (para o fluxo por variante) ──
            if "variant" in df.columns:
                groups = {}
                for var, sub in df.groupby("variant", sort=False):
                    groups[str(var)] = {
                        "count": int(len(sub)),
                        "rows":  sub.to_dict(orient="records"),
                    }
                data["by_variant"] = groups
                data["variants"]   = list(groups.keys())

        summs = sorted(out.glob("*summary*.csv"))
        if summs:
            df2 = pd.read_csv(summs[0], index_col=0).where(pd.notnull, None)
            data["summary"] = df2.to_dict(orient="index")

        return jsonify(_json_safe(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/download/<jid>/<path:fname>")
def api_download(jid, fname):
    if jid not in _jobs:
        abort(404)
    fpath = JOBS_DIR / jid / "out" / fname
    if fpath.exists() and fpath.parent == JOBS_DIR / jid / "out":
        return send_from_directory(str(fpath.parent), fname, as_attachment=True)
    abort(404)


if __name__ == "__main__":
    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │   DTMP-Prime Interface              │")
    print(f"  │   → http://localhost:5000           │")
    print(f"  └─────────────────────────────────────┘\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
