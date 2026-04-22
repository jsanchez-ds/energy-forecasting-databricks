#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/env.sh — project-local environment (source this before commands)
#
# Usage:  source scripts/env.sh
# ---------------------------------------------------------------------------

# Java (required by PySpark)
export JAVA_HOME="/c/Program Files/Eclipse Adoptium/jdk-11.0.30.7-hotspot"

# Extend PATH with the tools winget installed
export PATH="$JAVA_HOME/bin:$PATH"
export PATH="/c/Program Files/GitHub CLI:$PATH"
export PATH="/c/Users/jona2/AppData/Local/Microsoft/WinGet/Packages/Databricks.DatabricksCLI_Microsoft.Winget.Source_8wekyb3d8bbwe:$PATH"

echo "[env] JAVA_HOME=$JAVA_HOME"
echo "[env] gh      $(gh --version 2>/dev/null | head -1)"
echo "[env] java    $(java -version 2>&1 | head -1)"
echo "[env] databricks $(databricks --version 2>/dev/null | head -1)"
echo "[env] python3.11 $(py -3.11 --version 2>/dev/null)"
