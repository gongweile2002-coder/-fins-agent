# FinTech Project - Part A

> FIRST: rename this folder to <yourZID>_projectA (for example z1234567_projectA)
> and move it into fins-agent/fins2026/. The folder name carrying your zID is your
> submission.

Part A: the data foundation (DFF Stations 1-2). See the project brief and
context/project_context.md.

## How to run

    pip install -r requirements.txt
    python scripts/run_part_a.py        # reproduces your results into results/

Load all data through src/data_access.py (see context/DATA_GUIDE.md). Never commit
data files.

## What is here

- PROJECT_BRIEF.md   the full assignment brief for your course (read this first)
- src/        your code (data_access is provided; etl.py, features.py are yours)
- scripts/    runnable scripts that reproduce your results
- results/    your outputs: figures in results/figures/, tables in results/tables/, data artifacts in results/data/
- context/    provided data guide and project context (do not edit)
- report/     your report - see report/OUTLINE.md (author in Word, submit report.pdf)
- ai/         your prompt logs and AI notes
- AGENTS.md / CLAUDE.md   replace the stub for your tool (you need just one) with your own

## Before you hand in

    python scripts/check_handin.py

Then zip this whole folder and upload the zip to Moodle.
