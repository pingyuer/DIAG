#!/bin/bash
# wiki 重建：materials/proposals 同步到 docs-staging + mkdocs build 到 build/wiki
# S1后docs/不再是materials副本堆场；mkdocs docs_dir指向build/docs-staging
set -e
cd /home/tahara/DIAG
STAGING=build/docs-staging
rm -rf $STAGING && mkdir -p $STAGING/proposals
for f in materials/*.md; do cp "$f" "$STAGING/$(basename $f)"; done
for f in proposals/00*.md proposals/01*.md; do cp "$f" "$STAGING/proposals/$(basename $f)"; done
cp docs/project/diag_project_alignment.md docs/project/dpfr_diag_protocol_manual.md docs/project/ws_restructure_plan.md $STAGING/ 2>/dev/null || true
.venv/bin/python scripts/wiki_map_build.py
.venv/bin/mkdocs build -f mkdocs.yml -d build/wiki
echo REBUILT $(date +%H:%M)
