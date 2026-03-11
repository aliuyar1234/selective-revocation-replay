$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$required = @(
    (Join-Path $here "main.tex"),
    (Join-Path $here "refs.bib"),
    (Join-Path $here "artifact_table.tex"),
    (Join-Path $here "live_table.tex"),
    (Join-Path $here "assets\fig4_cost_vs_retention.pdf"),
    (Join-Path $here "assets\fig5_live_case_study.pdf")
)
foreach ($path in $required) {
    if (-not (Test-Path $path)) {
        throw "Missing required paper asset: $path"
    }
}

Push-Location $here
$auxFiles = @(
    "main.aux",
    "main.bbl",
    "main.blg",
    "main.fdb_latexmk",
    "main.fls",
    "main.log",
    "main.out"
)

try {
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
}
finally {
    foreach ($name in $auxFiles) {
        $path = Join-Path $here $name
        if (Test-Path $path) {
            Remove-Item $path -Force
        }
    }
}
Pop-Location
