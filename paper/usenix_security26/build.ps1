$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$required = @(
    (Join-Path $here "main.tex"),
    (Join-Path $here "refs.bib"),
    (Join-Path $here "artifact_table.tex"),
    (Join-Path $here "assets\fig4_cost_vs_retention.pdf"),
    (Join-Path $here "assets\fig5_live_case_study.pdf")
)
foreach ($path in $required) {
    if (-not (Test-Path $path)) {
        throw "Missing required paper asset: $path"
    }
}

Push-Location $here
$publicPdf = Join-Path (Split-Path $here -Parent) "selective-revocation-and-replay.pdf"
$auxFiles = @(
    "main.aux",
    "main.bbl",
    "main.blg",
    "main.fdb_latexmk",
    "main.fls",
    "main.log",
    "main.out",
    "main.pdf"
)

try {
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
    Copy-Item (Join-Path $here "main.pdf") $publicPdf -Force
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
