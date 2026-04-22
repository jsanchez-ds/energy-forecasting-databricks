#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/env.sh — project-local environment (source this before commands)
#
# Usage:  source scripts/env.sh
# ---------------------------------------------------------------------------

# Java (required by PySpark)
export JAVA_HOME="/c/Program Files/Eclipse Adoptium/jdk-11.0.30.7-hotspot"

# Hadoop winutils (required by PySpark on Windows for local file I/O)
# The binaries live in vendor/hadoop/bin — pinned to Hadoop 3.3.5
export HADOOP_HOME="$(pwd)/vendor/hadoop"
export PATH="$HADOOP_HOME/bin:$PATH"

# PySpark needs an explicit path to the Python inside the venv for its
# worker subprocesses, otherwise on Windows the JVM can't fork them.
export PYSPARK_PYTHON="$(pwd)/.venv/Scripts/python.exe"
export PYSPARK_DRIVER_PYTHON="$(pwd)/.venv/Scripts/python.exe"

# Extend PATH with the tools winget installed
export PATH="$JAVA_HOME/bin:$PATH"
export PATH="/c/Program Files/GitHub CLI:$PATH"
export PATH="/c/Users/jona2/AppData/Local/Microsoft/WinGet/Packages/Databricks.DatabricksCLI_Microsoft.Winget.Source_8wekyb3d8bbwe:$PATH"

echo "[env] JAVA_HOME=$JAVA_HOME"
echo "[env] HADOOP_HOME=$HADOOP_HOME"
echo "[env] gh      $(gh --version 2>/dev/null | head -1)"
echo "[env] java    $(java -version 2>&1 | head -1)"
echo "[env] databricks $(databricks --version 2>/dev/null | head -1)"
echo "[env] python3.11 $(py -3.11 --version 2>/dev/null)"
