param([switch]$Check)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
$pod = Get-Content -LiteralPath (Join-Path $repoRoot 'infra/runpod/pod.json') -Raw | ConvertFrom-Json
$keyPath = Join-Path $repoRoot '.secrets/runpod_ed25519'
$knownHostsPath = Join-Path $repoRoot '.secrets/runpod_known_hosts'
$sshArgs = @('-i', $keyPath, '-p', [string]$pod.ssh_port,
    '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
    '-o', 'StrictHostKeyChecking=accept-new', '-o', "UserKnownHostsFile=$knownHostsPath")
$destination = "root@$($pod.ssh_host)"
if ($Check) {
    & ssh @sshArgs $destination 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
} else {
    & ssh @sshArgs -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 `
        -L '127.0.0.1:18188:127.0.0.1:8188' -L '127.0.0.1:18081:127.0.0.1:8085' $destination
}
exit $LASTEXITCODE
