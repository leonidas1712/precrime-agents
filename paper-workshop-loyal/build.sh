#!/bin/sh
# Build the workshop PDF with the repo's TinyTeX plus the local package tree in this folder.
set -e
W="$(cd "$(dirname "$0")" && pwd)"
PATH="$(ls -d "$W"/../paper-arxiv/.tools/TinyTeX/bin/*/ | head -1):$PATH"
TEXINPUTS=".:$W/../paper-arxiv:"
export TEXINPUTS PATH TEXMFHOME="$W/.texmf" TEXMFVAR="$W/.texmf-var" TEXMFCONFIG="$W/.texmf-config"
cd "$W" && python3 make_refs.py && latexmk -pdf -interaction=nonstopmode -halt-on-error "$@" main.tex
