[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Scene,
    [int]$stride = 8
)

$allowedScenes = @("lerf_ramen", "lerf_teatime", "lerf_waldo_kitchen", "lerf_figurines")
if (-not ($allowedScenes -contains $Scene)) {
    Write-Error "不支持Scene: $Scene"
    exit 1
}

Write-Host "Running pipeline for scene: $Scene"

$verboseFlag = ""
if ($VerbosePreference -eq 'Continue') {
    $verboseFlag = "--verbose"
}


uv run bigs_langsplat.py --stride=$stride --level=1 --out --scene="$Scene" 

uv run bigs_langsplat.py --stride=$stride --level=2 --out --scene="$Scene" 

uv run bigs_langsplat.py --stride=$stride --level=3 --out --scene="$Scene" 

uv run bigs_langsplat_eval.py --stride=$stride --scene="$Scene" --pre_render --out

uv run bigs_langsplat_eval.py --stride=$stride --scene="$Scene" 